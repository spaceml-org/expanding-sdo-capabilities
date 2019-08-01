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
                 batch_size_test, log_interval, results_path, num_epochs, save_interval,
                 additional_metrics_interval, continue_training, saved_model_path, saved_optimizer_path, 
                 start_epoch_at, yr_range, mnt_step, day_step, h_step, min_step, dataloader_workers, scaling,
                 normalization, return_random_dim, tol=0.1):
        self.num_channels = len(wavelengths)
        self.results_path = results_path
        self.normalization_by_max = normalization
        self.wavelengths = wavelengths
        self.tol = tol

        _logger.info('Using {} channels across the following wavelengths and instruments:'.format(
            self.num_channels))
        _logger.info('Wavelengths: {}'.format(wavelengths))
        _logger.info('Instruments: {}'.format(instruments))

        assert yr_range is not None and len(yr_range) > 0, \
            'The AutocalibrationPipeline requires a yr_range: {}'.format(
                yr_range)
        _logger.info('Using following year range for both training and testing: {}'.format(
            yr_range))

        _logger.info('\nSetting up training dataset:')
        train_dataset = DimmedSDO_Dataset(self.num_channels, self.normalization_by_max,
                                          instr=instruments,
                                          channels=wavelengths, yr_range=yr_range,
                                          mnt_step=mnt_step, day_step=day_step,
                                          h_step=h_step, min_step=min_step,
                                          resolution=actual_resolution,
                                          subsample=subsample,
                                          normalization=0, scaling=scaling,
                                          return_random_dim=return_random_dim)

        _logger.info('\nSetting up testing dataset:')
        test_dataset = DimmedSDO_Dataset(self.num_channels, self.normalization_by_max,
                                         instr=instruments,
                                         channels=wavelengths, yr_range=yr_range,
                                         mnt_step=mnt_step, day_step=day_step,
                                         h_step=h_step, min_step=min_step,
                                         resolution=actual_resolution,
                                         subsample=subsample,
                                         normalization=0, scaling=scaling,
                                         return_random_dim=return_random_dim,
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
            additional_metrics_interval=additional_metrics_interval,
            continue_training=continue_training,
            saved_model_path=saved_model_path,
            saved_optimizer_path=saved_optimizer_path,
            start_epoch_at=start_epoch_at,
            scaling=scaling)

    def show_sample(self, loader):
        """ Show some samples for debugging purposes before training/testing. """
        _logger.info('\nUndimmed channels for single time slice:\n')
        _, _, item = loader.dataset[0]
        _logger.info('Max value: {}, min value: {}'.format(
            torch.max(item), torch.min(item)))
        _logger.info('Shape: {}'.format(item.shape))
        _logger.info('Dtype: {}'.format(item.dtype))
        fig, ax = plt.subplots(1, self.num_channels,
                               figsize=(10, 10), sharey=True)
        for c in range(self.num_channels):
            ax[c].title.set_text('Channel {}'.format(c + 1))
            ax[c].imshow(item[c].cpu().numpy(), cmap='gray')
        img_file = os.path.join(self.results_path, '{}_debug_sample.png'.format(
            format_graph_prefix(0, self.exp_name)))
        plt.savefig(img_file, bbox_inches='tight')
        plt.close()
        _logger.info('Debug sample saved to {}'.format(img_file))

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
        you'd like to use. The primary metric currently chosen is the binary frequency
        of correct cases, where a case is considerated correct if the real and predicted
        value differ equal less than the tol.
        """
        diff = torch.abs(output - gt_output)
        diff_np = diff.cpu().detach().numpy()
        primary_metric = (diff_np <= self.tol).sum() / (diff_np.shape[0]*diff_np.shape[1])
        return primary_metric

    def is_higher_better_primary_metric(self):
        return True
    
    def get_primary_metric_name(self):
        return 'Frequency of binary succes (tol={})'.format(self.tol)

    def generate_supporting_metrics(self, orig_data, output, input_data, gt_output, epoch, train):
        """ Print debugging details on the final batch per epoch during training or testing. """
        super(AutocalibrationPipeline, self).generate_supporting_metrics(
            orig_data, output, input_data, gt_output, epoch, train)

        # Generate some extra metric details that are specific to autocalibration.
        _logger.info('\n\nDetails with sample from final batch:')
        data_min, data_max = torch.min(orig_data), torch.max(orig_data)
        sample = orig_data[0].cpu().numpy()
        sample_dimmed = input_data[0].cpu().numpy()
        
        #TODO move the figure below into a plotting function
        fig = plt.figure()
        pos = 0
        for i, (channel, channel_dimmed) in enumerate(zip(sample, sample_dimmed)):
            pos += 1
            ax1 = fig.add_subplot(self.num_channels, 3, pos)
            fig.subplots_adjust(top=3.0)
            ax1.set_title(
                'Channel: {} (left: original, middle: dimmed, right: undimmed)\n'
                'Dimming (true): {}, dimming (predicted): {}'.format(i+1, gt_output[0, i],
                                                                     output[0, i]))
            ax1.axis('off')
            ax1.imshow(channel, norm=None, cmap='hot',
                       vmin=data_min, vmax=data_max)

            pos += 1
            ax2 = fig.add_subplot(self.num_channels, 3, pos)
            ax2.axis('off')
            ax2.imshow(channel_dimmed, norm=None, cmap='hot',
                       vmin=data_min, vmax=data_max)

            pos += 1
            ax3 = fig.add_subplot(self.num_channels, 3, pos)
            ax3.axis('off')
            ax3.imshow(channel_dimmed / float(output[0, i]), norm=None, cmap='hot', vmin=data_min,
                       vmax=data_max)
        img_file = os.path.join(self.results_path, '{}_debug_sample_{}.png'.format(
            format_graph_prefix(epoch, self.exp_name), 'train' if train else 'test'))
        plt.savefig(img_file, bbox_inches='tight')
        plt.close()
        _logger.info('Debug sample saved to {}'.format(img_file))

        #TODO move dimming values figure below into a plotting function if we still need it
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
        # this plot includes histrograms of the variables
        output_numpy = output.detach().cpu().numpy()
        gt_output_numpy = gt_output.detach().cpu().numpy()
        if train:
            title = 'GT Dimmed Factor vs Predicted Dimmed Factor - Training'
        else:
            title = 'GT Dimmed Factor vs Predicted Dimmed Factor - Testing'
        x_label = "Predicted Dimmed Factor"
        y_label =  "Ground Truth Dimmed Factor"
        img_file = os.path.join(self.results_path, '{}_GTvsPR_plot_metric_{}.png'.format(
            format_graph_prefix(epoch, self.exp_name), 'train' if train else 'test'))
        # all channels are plotted together
        plot_regression(output_numpy.flatten(), gt_output_numpy.flatten(), 
                        title, x_label, y_label, img_file)
        _logger.info('Regression plot saved to {}'.format(img_file))
            
        # Pearson correlation values last batch
        pr_coeff = []
        for i, channel in enumerate(self.wavelengths):
            pr_coeff.append(stats.pearsonr(output_numpy[i], gt_output_numpy[i])[0])
        df_pr_coeff = pd.DataFrame(dict(zip(self.wavelengths, pr_coeff)), index=[0])
        _logger.info('\n\nPearson coefficient values by channel \n {}'
                     .format(df_pr_coeff))
        _logger.info('Mean Pearson coefficient {}'.format(np.mean(pr_coeff)))  
        _logger.info('\n')
