
import logging
import os

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import scipy.stats as stats

from sdo.datasets.dimmed_sdo_dataset import DimmedSDO_Dataset
from sdo.io import format_graph_prefix
from sdo.models.autocalibration1 import Autocalibration1
from sdo.models.autocalibration2 import Autocalibration2
from sdo.pipelines.training_pipeline import TrainingPipeline
from sdo.pytorch_utilities import create_dataloader
from sdo.metrics.plotting import plot_regression


_logger = logging.getLogger(__name__)


class AutocalibrationPipeline(TrainingPipeline):
    def __init__(self, exp_name, model_version, actual_resolution, scaled_height,
                 scaled_width, device, instruments, wavelengths, subsample, batch_size_train,
                 batch_size_test, test_ratio, log_interval, results_path, num_epochs, save_interval,
                 additional_metrics_interval, continue_training, saved_model_path, saved_optimizer_path,
                 start_epoch_at, yr_range, mnt_step, day_step, h_step, min_step, dataloader_workers, scaling,
                 optimizer_weight_decay, optimizer_lr, tolerance, min_alpha):
        self.num_channels = len(wavelengths)
        self.results_path = results_path
        self.wavelengths = wavelengths
        self.tolerance = tolerance
        self.scaling = scaling

        _logger.info('Using {} channels across the following wavelengths and instruments:'.format(
            self.num_channels))
        _logger.info('Wavelengths: {}'.format(wavelengths))
        _logger.info('Instruments: {}'.format(instruments))

        _logger.info('\nSetting up training dataset:')
        train_dataset = DimmedSDO_Dataset(num_channels=self.num_channels,
                                          instr=instruments,
                                          channels=wavelengths, yr_range=yr_range,
                                          mnt_step=mnt_step, day_step=day_step,
                                          h_step=h_step, min_step=min_step,
                                          resolution=actual_resolution,
                                          subsample=subsample,
                                          normalization=0, scaling=scaling,
                                          test_ratio=test_ratio,
                                          min_alpha=min_alpha,
                                          shuffle=True)

        _logger.info('\nSetting up testing dataset:')
        test_dataset = DimmedSDO_Dataset(num_channels=self.num_channels,
                                         instr=instruments,
                                         channels=wavelengths, yr_range=yr_range,
                                         mnt_step=mnt_step, day_step=day_step,
                                         h_step=h_step, min_step=min_step,
                                         resolution=actual_resolution,
                                         subsample=subsample,
                                         normalization=0, scaling=scaling,
                                         test_ratio=test_ratio, min_alpha=min_alpha,
                                         shuffle=True, test=True)

        train_loader = create_dataloader(train_dataset, batch_size_train,
                                         dataloader_workers, train=True)
        test_loader = create_dataloader(test_dataset, batch_size_test,
                                        dataloader_workers, train=False)

        if model_version == 1:
            model = Autocalibration1(input_shape=[self.num_channels, scaled_height,
                                                  scaled_width], output_dim=self.num_channels)
        elif model_version == 2:
            model = Autocalibration2(input_shape=[self.num_channels, scaled_height,
                                                  scaled_width], output_dim=self.num_channels,
                                     increase_dim=2)
        else:
            # Note: For other model_versions, simply instantiate whatever class
            # you want to test your experiment for. You will have to update the code
            # here to reference that class, preferably in sdo.models.*, such as
            # sdo.models.Autocalibration2.
            raise Exception('Unknown model version: {}'.format(model_version))

        model.cuda(device)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=optimizer_weight_decay,
                                     lr=optimizer_lr)

        super(AutocalibrationPipeline, self).__init__(
            exp_name=exp_name,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            train_loader=train_loader,
            test_loader=test_loader,
            batch_size_train=batch_size_train,
            batch_size_test=batch_size_test,
            model=model,
            optimizer=optimizer,
            log_interval=log_interval,
            results_path=results_path,
            num_epochs=num_epochs,
            device=device,
            save_interval=save_interval,
            additional_metrics_interval=additional_metrics_interval,
            continue_training=continue_training,
            saved_model_path=saved_model_path,
            saved_optimizer_path=saved_optimizer_path,
            start_epoch_at=start_epoch_at,
            scaling=scaling)

    def show_sample(self, loader):
        """ Show some samples for debugging purposes before training/testing. """
        _logger.info('Showing a single sample across all channels at start for debugging purposes:')
        # Get a single sample from the dataset, with all of its channels.
        dimmed_img, dim_factors, orig_img = loader.dataset[0]

        _logger.info('\nScaling: {}'.format(self.scaling))

        _logger.info('\nDimmed image:')
        _logger.info('\tMax value: {}, min value: {}'.format(torch.max(dimmed_img),
                                                             torch.min(dimmed_img)))
        _logger.info('\tShape: {}'.format(dimmed_img.shape))
        _logger.info('\tDtype: {}'.format(dimmed_img.dtype))

        _logger.info('\nDimming factors:')
        _logger.info('\t{}'.format(dim_factors))

        _logger.info('\nOriginal undimmed image:')
        _logger.info('\tMax value: {}, min value: {}'.format(torch.max(orig_img),
                                                             torch.min(orig_img)))
        _logger.info('\tShape: {}'.format(orig_img.shape))
        _logger.info('\tDtype: {}'.format(orig_img.dtype))

    def get_loss_func(self, output, gt_output):
        """ Return the loss function this pipeline is using. """
        # Both the output and gt_output should be a vector num_channels wide, where each vector entry is the
        # brightness dimming factor.
        return nn.MSELoss()(output, gt_output)

    def calculate_primary_metric(self, epoch, output, gt_output):
        """
        Given some predicted output from a network and some ground truth, this method
        calculates the binary frequency of correct cases, where a case is considerated 
        correct if the real and predicted value differ equal less than the tolerance. 
        The mean over the batch and the channels is returned as single metric value.
        """
        diff = torch.abs(output - gt_output)
        batch_size = diff.shape[0]
        n_tensor_elements = batch_size * self.num_channels
        primary_metric = (torch.sum(diff < self.tolerance, dtype=torch.float32) /
                          n_tensor_elements)
        return primary_metric.cpu()

    def is_higher_better_primary_metric(self):
        return True

    def get_primary_metric_name(self):
        return 'Frequency of binary success (tol={})'.format(self.tolerance)

    def generate_supporting_metrics(self, orig_img, output, input_data, gt_output, epoch,
                                    train):
        """ Print debugging details on the final batch per epoch during training or testing. """
        super(AutocalibrationPipeline, self).generate_supporting_metrics(
            orig_img, output, input_data, gt_output, epoch, train)

        # Generate some extra metric details that are specific to autocalibration.
        _logger.info('\n\nDetails with sample from final batch:')

        sample = orig_img[0].cpu().numpy()
        sample_dimmed = input_data[0].cpu().numpy()

        scale_min = 0
        # Make sure that our visualization aren't too affected by very large outliers.
        # TODO: Scale the divisor by our different channel wavelengths somehow instead
        # of having the fixed value of 10.
        scale_max = sample.max() / 10.0

        # TODO move the figure below into a plotting function
        fig = plt.figure()
        pos = 0
        for i, (channel_orig, channel_dimmed) in enumerate(zip(sample, sample_dimmed)):
            pos += 1
            pred_channel_dim_factor = float(output[0, i])
            # Reconstructed means we want to apply some transformation to the dimmed image
            # to get back the original undimmed image.
            reconstructed_channel = channel_dimmed / pred_channel_dim_factor

            ax1 = fig.add_subplot(self.num_channels, 4, pos)
            fig.subplots_adjust(top=3.0)
            ax1.set_title(
                'Channel: {} (col 1: original, col 2: dimmed, col 3: reconstructed, col 4: difference)\n'
                'Dimming (true): {}, dimming (predicted): {}'.format(i+1, gt_output[0, i],
                                                                     pred_channel_dim_factor))
            ax1.axis('off')
            ax1.imshow(channel_orig, norm=None, cmap='hot', vmin=scale_min, vmax=scale_max)

            pos += 1
            ax2 = fig.add_subplot(self.num_channels, 4, pos)
            ax2.axis('off')
            ax2.imshow(channel_dimmed, norm=None, cmap='hot', vmin=scale_min, vmax=scale_max)

            pos += 1
            ax3 = fig.add_subplot(self.num_channels, 4, pos)
            ax3.axis('off')
            ax3.imshow(reconstructed_channel, norm=None, cmap='hot', vmin=scale_min,
                       vmax=scale_max)

            # See the difference of how well we reconstructed a dimmed image vs. the actual original
            # non-degraded image.
            pos += 1
            ax4 = fig.add_subplot(self.num_channels, 4, pos)
            ax4.axis('off')
            channel_diff = channel_orig - reconstructed_channel
            ax4.imshow(channel_diff, norm=None, cmap='hot')

        img_file = os.path.join(self.results_path, '{}_debug_sample_{}.png'.format(
            format_graph_prefix(epoch, self.exp_name), 'train' if train else 'test'))
        plt.savefig(img_file, bbox_inches='tight')
        plt.close()
        _logger.info('Debug sample saved to {}'.format(img_file))

        # TODO move dimming values figure below into a plotting function if we still need it
        # or decide to remove it
        fig = plt.figure()
        dim_factors_numpy = gt_output[0].view(-1).cpu().numpy()
        plt.scatter(range(1, self.num_channels + 1), dim_factors_numpy,
                    label='Dimming factors (true)')
        output_numpy = output[0].detach().view(-1).cpu().numpy()
        plt.scatter(range(1, self.num_channels + 1), output_numpy,
                    label='Dimming factors (predicted)')
        title = 'training dimming factors' if train else 'testing dimming factors'
        plt.title(title)
        plt.xlabel("Channel")
        plt.ylabel("Dimming factor")
        plt.legend()
        img_file = os.path.join(self.results_path, '{}_dimming_factors_graph_{}.png'.format(
            format_graph_prefix(epoch, self.exp_name), 'train' if train else 'test'))
        plt.savefig(img_file, bbox_inches='tight')
        plt.close()
        _logger.info('Dimming factors graph saved to {}'.format(img_file))

        # Plotting regression plot between Ground Truth Dimm factor vs. Predicted Dimm Factors
        # this plot includes histograms of the variables
        output_numpy = output.detach().cpu().numpy()
        gt_output_numpy = gt_output.detach().cpu().numpy()
        title = 'GT Dimmed Factor vs. Predicted Dimmed Factor - {}'.format(
                'Training' if train else 'Testing')
        x_label = "Predicted Dimmed Factor"
        y_label = "Ground Truth Dimmed Factor"
        img_file = os.path.join(self.results_path, '{}_GTvsPR_plot_metric_{}.png'.format(
            format_graph_prefix(epoch, self.exp_name), 'train' if train else 'test'))
        # all channels are plotted together
        plot_regression(output_numpy.flatten(), gt_output_numpy.flatten(),
                        title, x_label, y_label, img_file)
        _logger.info('Regression plot saved to {}'.format(img_file))

        # Pearson correlation values last batch
        pr_coeff = []
        for i, channel in enumerate(self.wavelengths):
            pr_coeff.append(stats.pearsonr(
                output_numpy[i], gt_output_numpy[i])[0])
        df_pr_coeff = pd.DataFrame(
            dict(zip(self.wavelengths, pr_coeff)), index=[0])
        _logger.info('\n\nPearson coefficient values by channel \n {}'
                     .format(df_pr_coeff))
        _logger.info('Mean Pearson coefficient {}'.format(np.mean(pr_coeff)))
        _logger.info('\n')
