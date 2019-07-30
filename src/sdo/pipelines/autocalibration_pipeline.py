import logging
import os

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import pandas

from sdo.datasets.dimmed_sdo_dataset import DimmedSDO_Dataset
from sdo.io import format_epoch
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
                 yr_range, mnt_step, day_step, h_step, min_step, dataloader_workers):
        self.num_channels = len(wavelengths)
        # TODO Change resluts_path to a path that is shared
        self.results_path = results_path

        _logger.info('Using {} channels across the following wavelengths and instruments:'.format(
            self.num_channels))
        _logger.info('Wavelengths: {}'.format(wavelengths))
        _logger.info('Instruments: {}'.format(instruments))

        assert yr_range is not None and len(yr_range) > 0, \
            'The AutocalibrationPipeline requires a yr_range: {}'.format(yr_range)
        _logger.info('Using following year range for both training and testing: {}'.format(
            yr_range))

        # TODO: We are using 'scaling=True' below, and are currently also scaling by the max()
        # pixel value inside of dimmed_sdo_dataset. This might cause issues. However, we know
        # that training requires our values to be roughly between 0 and 1. Resolve where
        # and how to do this scaling.
        _logger.info('\nSetting up training dataset:')
        train_dataset = DimmedSDO_Dataset(self.num_channels, instr=instruments,
                                          channels=wavelengths, yr_range=yr_range,
                                          mnt_step=mnt_step, day_step=day_step,
                                          h_step=h_step, min_step=min_step,
                                          resolution=actual_resolution,
                                          subsample=subsample,
                                          normalization=0, scaling=False)

        _logger.info('\nSetting up testing dataset:')
        test_dataset = DimmedSDO_Dataset(self.num_channels, instr=instruments,
                                         channels=wavelengths, yr_range=yr_range,
                                         mnt_step=mnt_step, day_step=day_step,
                                         h_step=h_step, min_step=min_step,
                                         resolution=actual_resolution,
                                         subsample=subsample,
                                         normalization=0, scaling=False,
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
            start_epoch_at=start_epoch_at)

    def show_sample(self, loader):
        """ Show some samples for debugging purposes before training/testing. """
        _logger.info('\nUndimmed channels for single time slice:\n')
        _, _, item = loader.dataset[0]
        _logger.info('Max value: {}, min value: {}'.format(torch.max(item), torch.min(item)))
        _logger.info('Shape: {}'.format(item.shape))
        _logger.info('Dtype: {}'.format(item.dtype))
        fig, ax = plt.subplots(1, self.num_channels, figsize=(10,10), sharey=True)
        for c in range(self.num_channels):
            ax[c].title.set_text('Channel {}'.format(c + 1))
            ax[c].imshow(item[c].cpu().numpy(), cmap='gray')
        img_file = os.path.join(self.results_path, '{}_debug_sample.png'.format(
            format_epoch(0)))
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
        you'd like to use.
        """
        primary_metric = torch.mean(torch.mean(torch.abs(output - gt_output), dim=1), dim=0)
        primary_metric = float(primary_metric.cpu())

        # Note: lower values are better for our primary metric here.
        return primary_metric

    def is_higher_better_primary_metric(self):
        return False

    def generate_supporting_metrics(self, orig_data, output, input_data, gt_output, epoch, train):
        """ Print debugging details on the final batch per epoch during training or testing. """
        super(AutocalibrationPipeline, self).generate_supporting_metrics(
            orig_data, output, input_data, gt_output, epoch, train)

        # Generate some extra metric details that are specific to autocalibration.
        _logger.info('\n\nDetails with sample from final batch:')
        data_min, data_max = torch.min(orig_data), torch.max(orig_data)
        sample = orig_data[0].cpu().numpy()
        sample_dimmed = input_data[0].cpu().numpy()

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
            ax1.imshow(channel, norm=None, cmap='hot', vmin=data_min, vmax=data_max)

            pos += 1
            ax2 = fig.add_subplot(self.num_channels, 3, pos)
            ax2.axis('off')
            ax2.imshow(channel_dimmed, norm=None, cmap='hot', vmin=data_min, vmax=data_max)

            pos += 1
            ax3 = fig.add_subplot(self.num_channels, 3, pos)
            ax3.axis('off')
            ax3.imshow(channel_dimmed / float(output[0, i]), norm=None, cmap='hot', vmin=data_min,
                       vmax=data_max)
        img_file = os.path.join(self.results_path, '{}_debug_sample_{}.png'.format(
            format_epoch(epoch), 'train' if train else 'test'))
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
            format_epoch(epoch), 'train' if train else 'test'))
        plt.savefig(img_file, bbox_inches='tight')
        plt.close()
        _logger.info('Dimming factors graph saved to {}'.format(img_file))

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