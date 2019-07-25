import logging
import os

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas

from sdo.dimmed_sdo_dataset import DimmedSDO_Dataset
from sdo.io import format_epoch
from sdo.models.autocalibration import Autocalibration
from sdo.pipelines.training_pipeline import TrainingPipeline


_logger = logging.getLogger(__name__)


class AutocalibrationPipeline(TrainingPipeline):
    def __init__(self, num_channels, height, width, device, instruments, wavelengths,
                 subsample, batch_size_train, batch_size_test, log_interval,
                 results_path, num_epochs, save_interval, continue_training,
                 saved_model_path, saved_optimizer_path, start_epoch_at,
                 yr_range, pct_close=0.15):
        self.num_channels = num_channels
        self.results_path = results_path

        assert yr_range is not None and len(yr_range) > 0, \
            'The AutocalibrationPipeline requires a yr_range: {}'.format(yr_range)
        _logger.info('Using following year range for both training and testing: {}'.format(
            yr_range))

        train_dataset = DimmedSDO_Dataset(num_channels, device, instr=instruments,
                                          channels=wavelengths, yr_range=yr_range,
                                          subsample=subsample,
                                          normalization=0, scaling=True)
        test_dataset = DimmedSDO_Dataset(num_channels, device, instr=instruments,
                                         channels=wavelengths, yr_range=yr_range,
                                         subsample=subsample,
                                         normalization=0, scaling=True,
                                         test=True)

        # TODO: Calculate global mean/std across brightness adjusted data.
        # Apply this global mean/std across the data to normalize it in the
        # loader. Note that we might not want to apply the std as this might
        # remove our brightness correlations.

        train_loader = DataLoader(train_dataset, batch_size=batch_size_train,
                                  shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_test,
                                 shuffle=True)

        model = Autocalibration(input_shape=[num_channels, height, width],
                                output_dim=num_channels)
        model.cuda(device)
        optimizer = torch.optim.Adam(model.parameters())

        # If a channel brightness prediction is within this percentage of the ground truth then we
        # consider that prediction correct.
        self.pct_close = pct_close

        super(AutocalibrationPipeline, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            train_loader=train_loader,
            test_loader=test_loader,
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
        _logger.info('Debug sample saved to {}'.format(img_file))

    def get_loss_func(self, output, gt_output):
        """ Return the loss function this pipeline is using. """
        return nn.MSELoss()(output, gt_output)

    def print_final_batch_details(self, orig_data, output, input_data, gt_output, epoch, train):
        """ Print debugging details on the final batch per epoch during training or testing. """
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
        _logger.info('Debug sample saved to {}'.format(img_file))

        fig = plt.figure()
        dim_factors_numpy = gt_output[0].view(-1).cpu().numpy()
        plt.plot(dim_factors_numpy, label='Dimming factors (true)')
        output_numpy = output[0].detach().view(-1).cpu().numpy()
        plt.plot(output_numpy, label='Dimming factors (predicted)')
        title = 'training dimming factors' if train else 'testing dimming factors'
        plt.title(title)
        plt.legend()
        img_file = os.path.join(self.results_path, '{}_dimming_factors_graph_{}.png'.format(
            format_epoch(epoch), 'train' if train else 'test'))
        plt.savefig(img_file, bbox_inches='tight')
        _logger.info('Dimming factors graph saved to {}'.format(img_file))

        num_subsample = 3 # For the final batch, the number of entries to subsample to print out for debugging.
        column_labels = ['Pred', 'GT', 'Delta', 'Pct Correct']
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

        # Percentage correct across all the channels for a given batch row?
        per_channel_diff = np.abs(gt_output - output)
        per_channel_correct = per_channel_diff <= np.abs(self.pct_close * gt_output)
        correct_per_channel = np.sum(np.where(per_channel_correct, 1, 0), axis=1, keepdims=True,
                                     dtype=np.int)
        pct_correct_per_channel = correct_per_channel / self.num_channels
        pretty_results[:, 3] = np.round(pct_correct_per_channel * 100.0,
                                        decimals=0)[:num_subsample, 0].astype(np.int)

        df = pandas.DataFrame(pretty_results, columns=column_labels)
        _logger.info("\n\nRandom sample of mean predictions across channels, "
                     "where each row is a sample in the training batch:\n")
        _logger.info(df.to_string(index=False))

    def get_correct_count(self, output, gt_output):
        """ Given some predictions and ground truth, calculate how many are 'correct' over the batch.
            What is considered 'correct' differs based on what a pipeline is trying to do. """
        preds = output.detach().cpu().numpy()
        targets = gt_output.detach().cpu().numpy()

        per_channel_diff = np.abs(targets - preds)
        per_channel_correct = per_channel_diff <= np.abs(self.pct_close * targets)
        correct_per_channel = np.sum(np.where(per_channel_correct, 1, 0), axis=1, keepdims=True,
                                     dtype=np.int)
        pct_correct_per_channel = correct_per_channel / self.num_channels

        # Which batch results have _all_ of their channel predictions fully correct?
        num_fully_correct_all_channels = np.where(pct_correct_per_channel == 1.0, 1, 0).sum()

        correct = num_fully_correct_all_channels
        return correct