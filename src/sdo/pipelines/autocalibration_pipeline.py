import logging
import os

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import scipy.stats as stats
import pandas

from sdo.datasets.dimmed_sdo_dataset import DimmedSDO_Dataset
from sdo.io import format_graph_prefix
from sdo.models.autocalibration1 import Autocalibration1
from sdo.models.autocalibration2 import Autocalibration2
from sdo.pipelines.training_pipeline import TrainingPipeline
from sdo.pytorch_utilities import create_dataloader


_logger = logging.getLogger(__name__)


class AutocalibrationPipeline(TrainingPipeline):
    def __init__(self, exp_name, model_version, actual_resolution, scaled_height,
                 scaled_width, device, instruments, wavelengths, subsample, batch_size_train,
                 batch_size_test, log_interval, results_path, num_epochs, save_interval,
                 continue_training, saved_model_path, saved_optimizer_path, start_epoch_at,
                 yr_range, mnt_step, day_step, h_step, min_step, dataloader_workers, scaling,
                 return_random_dim, norm_by_orig_img_max, norm_by_dimmed_img_max):
        self.num_channels = len(wavelengths)
        self.results_path = results_path
        self.norm_by_orig_img_max = norm_by_orig_img_max
        self.norm_by_dimmed_img_max = norm_by_dimmed_img_max

        _logger.info('Using {} channels across the following wavelengths and instruments:'.format(
            self.num_channels))
        _logger.info('Wavelengths: {}'.format(wavelengths))
        _logger.info('Instruments: {}'.format(instruments))

        assert yr_range is not None and len(yr_range) > 0, \
            'The AutocalibrationPipeline requires a yr_range: {}'.format(yr_range)
        _logger.info('Using following year range for both training and testing: {}'.format(
            yr_range))

        _logger.info('\nSetting up training dataset:')
        train_dataset = DimmedSDO_Dataset(self.num_channels,
                                          instr=instruments,
                                          channels=wavelengths, yr_range=yr_range,
                                          mnt_step=mnt_step, day_step=day_step,
                                          h_step=h_step, min_step=min_step,
                                          resolution=actual_resolution,
                                          subsample=subsample,
                                          normalization=0, scaling=scaling,
                                          return_random_dim=return_random_dim,
                                          norm_by_orig_img_max=norm_by_orig_img_max,
                                          norm_by_dimmed_img_max=norm_by_dimmed_img_max)

        _logger.info('\nSetting up testing dataset:')
        test_dataset = DimmedSDO_Dataset(self.num_channels,
                                         instr=instruments,
                                         channels=wavelengths, yr_range=yr_range,
                                         mnt_step=mnt_step, day_step=day_step,
                                         h_step=h_step, min_step=min_step,
                                         resolution=actual_resolution,
                                         subsample=subsample,
                                         normalization=0, scaling=scaling,
                                         return_random_dim=return_random_dim,
                                         norm_by_orig_img_max=norm_by_orig_img_max,
                                         norm_by_dimmed_img_max=norm_by_dimmed_img_max,
                                         test=True)

        # TODO: Calculate global mean/std across brightness adjusted data.
        # Apply this global mean/std across the data to normalize it in the
        # loader. Note that we might not want to apply the std as this might
        # remove our brightness correlations.

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
        optimizer = torch.optim.Adam(model.parameters())

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

        _logger.info('\nNormalization flags:')
        _logger.info('norm_by_orig_img_max: {}'.format(
            self.norm_by_orig_img_max))
        _logger.info('norm_by_dimmed_img_max: {}'.format(
            self.norm_by_dimmed_img_max))

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
        calculates a scalar on how "well" we are doing for a given problem to gauge
        progress during different experiments and during training. Note that we
        already calculate and print out the loss outside of this method, so this
        method is appropriate for other kinds of scalar values indicating progress
        you'd like to use.
        """
        primary_metric = torch.mean(torch.mean(torch.abs(output - gt_output), dim=1), dim=0)
        primary_metric = float(primary_metric.cpu())

        # Note: lower values are better for our primary metric here.
        return primary_metric

    def is_higher_better_primary_metric(self):
        return False

    def generate_supporting_metrics(self, normed_orig_data, output, input_data, gt_output, epoch,
                                    train):
        """ Print debugging details on the final batch per epoch during training or testing. """
        super(AutocalibrationPipeline, self).generate_supporting_metrics(
            normed_orig_data, output, input_data, gt_output, epoch, train)

        # Generate some extra metric details that are specific to autocalibration.
        _logger.info('\n\nDetails with sample from final batch:')

        scale_min = 0
        scale_max = normed_orig_data.max()

        sample = normed_orig_data[0].cpu().numpy()
        sample_dimmed = input_data[0].cpu().numpy()

        # TODO: These images don't look correct; our channel 1 original image is dim, which is wrong.
        # our channel. Col 2 dimming image looks completely black. Things don't look great.
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
            ax4.imshow(channel_diff, norm=None, cmap='hot', vmin=scale_min, vmax=scale_max)

        img_file = os.path.join(self.results_path, '{}_debug_sample_{}.png'.format(
            format_graph_prefix(epoch, self.exp_name), 'train' if train else 'test'))
        plt.savefig(img_file, bbox_inches='tight')
        plt.close()
        _logger.info('Debug sample saved to {}'.format(img_file))

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
        
        # This is a regression plot between Ground Truth Dimm factor vs.  Predicted Dimm Factors
        
        output_numpy = output.detach().cpu().numpy()
        gt_output_numpy = gt_output.detach().cpu().numpy()

        ax = sns.jointplot(x=output_numpy, y=gt_output_numpy,kind='reg')
        title = 'GT Dimmed Factor vs Predicted Dimmed Factor - Training' \
            if train else 'GT Dimmed Factor vs Predicted Dimmed Factor - Testing'
        ax.set_axis_labels("Predicted Dimmed Factor","Ground Truth Dimmed Factor")
        plt.title(title)
        plt.tight_layout()
        ax.annotate(stats.pearsonr)
        img_file = os.path.join(self.results_path, '{}_GTvsPR_plot_metric_{}.png'.format(
            format_graph_prefix(epoch, self.exp_name), 'train' if train else 'test'))
        plt.savefig(img_file, bbox_inches='tight')
        plt.close()

        # TODO: Either speed this up by doing it in torch or print it out less often.
        # It's becomming a bottleneck now that things are faster elsewhere.
        num_subsample = 3 # For the final batch, the number of entries to subsample to print out for debugging.
        column_labels = ['Pred', 'GT', 'Mean Delta']
        pretty_results = np.zeros((min(num_subsample, len(output)), len(column_labels)),
                                  dtype=np.float32)

        output = output.detach().cpu().numpy()
        gt_output = gt_output.detach().cpu().numpy()

        # The mean channel prediction across each row of the batch results.
        pretty_results[:, 0] = np.round(output.mean(axis=1)[:num_subsample], decimals=2)

        # The mean channel ground truth across each row of the batch results.
        pretty_results[:, 1] = np.round(gt_output.mean(axis=1)[:num_subsample], decimals=2)

        # The mean difference btw prediction and grouth truth across each row of the batch results.
        pretty_results[:, 2] = np.round(np.abs(gt_output - output).mean(axis=1)[:num_subsample],
                                        decimals=2)

        df = pandas.DataFrame(pretty_results, columns=column_labels)
        _logger.info("\n\nRandom sample of mean predictions across channels, "
                     "where each row is a sample in the training batch:\n")
        _logger.info(df.to_string(index=False))
        _logger.info('\n')