import logging
import os

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import seaborn as sns

from sklearn.metrics import mean_squared_error
import scipy.stats as stats

import pandas

from sdo.datasets.virtual_telescope_sdo_dataset import VirtualTelescopeSDO_Dataset
from sdo.io import format_graph_prefix
from sdo.metrics.azimuth_metric import azimuthal_average, compute_2Dpsd
from sdo.metrics.extended_vt_metrics import structural_sim, pixel_sim
from sdo.models.vt_encoder_decoder import VT_EncoderDecoder
from sdo.models.vt_basic_encoder import VT_BasicEncoder
from sdo.pipelines.training_pipeline import TrainingPipeline
from sdo.pytorch_utilities import create_dataloader


_logger = logging.getLogger(__name__)


class VirtualTelescopePipeline(TrainingPipeline):
    def __init__(self, exp_name, model_version, actual_resolution, scaled_height,
                 scaled_width, device, instruments, wavelengths, subsample, batch_size_train,
                 batch_size_test, test_ratio, log_interval, results_path, num_epochs, save_interval,
                 additional_metrics_interval, continue_training, saved_model_path, saved_optimizer_path,
                 start_epoch_at, yr_range, mnt_step, day_step, h_step, min_step, dataloader_workers,
                 scaling, optimizer_weight_decay, optimizer_lr):
        self.num_channels = len(wavelengths)

        _logger.info('Using {} channels across the following wavelengths and instruments:'.format(
            self.num_channels))
        _logger.info('Wavelengths: {}'.format(wavelengths))
        _logger.info('Instruments: {}'.format(instruments))

        _logger.info('\nSetting up training dataset:')
        train_dataset = VirtualTelescopeSDO_Dataset(
                                    num_channels=self.num_channels,
                                    instr=instruments,
                                    channels=wavelengths, yr_range=yr_range,
                                    mnt_step=mnt_step, day_step=day_step,
                                    h_step=h_step, min_step=min_step,
                                    resolution=actual_resolution,
                                    subsample=subsample,
                                    normalization=0, scaling=scaling,
                                    test_ratio=test_ratio)

        _logger.info('\nSetting up testing dataset:')
        test_dataset = VirtualTelescopeSDO_Dataset(
                                   num_channels=self.num_channels,
                                   instr=instruments,
                                   channels=wavelengths, yr_range=yr_range,
                                   mnt_step=mnt_step, day_step=day_step,
                                   h_step=h_step, min_step=min_step,
                                   resolution=actual_resolution,
                                   subsample=subsample,
                                   normalization=0, scaling=scaling,
                                   test_ratio=test_ratio, test=True)

        train_loader = create_dataloader(train_dataset, batch_size_train,
                                         dataloader_workers, train=True)
        test_loader = create_dataloader(test_dataset, batch_size_test,
                                        dataloader_workers, train=False)

        if model_version == 1:
            # TODO add hidden_dim to the pipeline parameters
            model = VT_EncoderDecoder(input_shape=[self.num_channels - 1, scaled_height,
                                                   scaled_width], hidden_dim=512)
        elif model_version == 2:
            model = VT_BasicEncoder(input_shape=[self.num_channels - 1, scaled_height,
                                                 scaled_width])
        else:
            # Note: For other model_versions, simply instantiate whatever class
            # you want to test your experiment for. You will have to update the code
            # here to reference that class, preferably in sdo.models.*, such as
            # sdo.models.Autocalibration2.
            raise Exception('Unknown model version: {}'.format(model_version))

        model.cuda(device)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=optimizer_weight_decay,
                                     lr=optimizer_lr)

        super(VirtualTelescopePipeline, self).__init__(
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
        input_data, gt_output, _ = loader.dataset[0]

        # Print each of the input channels on their own line.
        fig = plt.figure()
        pos = 0
        for i in range(self.num_channels - 1):
            pos += 1

            ax = fig.add_subplot(self.num_channels, 1, pos)
            fig.subplots_adjust(top=3.0)
            ax.set_title('Input Channel {}'.format(i + 1))
            ax.axis('off')
            ax.imshow(input_data[i].detach().numpy(), vmin=0, vmax=1)

        pos += 1

        ax = fig.add_subplot(self.num_channels, 1, pos)
        fig.subplots_adjust(top=3.0)
        ax.set_title('Output Channel {}'.format(self.num_channels - 1))
        ax.axis('off')
        ax.imshow(gt_output[0].detach().numpy(), vmin=0, vmax=1)

        img_file = os.path.join(self.results_path, '{}_initial_sample.png'.format(
            format_graph_prefix(0, self.exp_name)))
        plt.savefig(img_file, bbox_inches='tight')
        plt.close()
        _logger.info('Initial sample saved to {}'.format(img_file))

    def get_loss_func(self, output, gt_output):
        """ Return the loss function this pipeline is using. """
        distance = nn.MSELoss()
        return torch.sqrt(distance(output, gt_output))

    def calculate_primary_metric(self, epoch, output, gt_output):
        """
        Given some predicted output from a network and some ground truth, this method
        calculates a scalar on how "well" we are doing for a given problem to gauge
        progress during different experiments and during training. Note that we
        already calculate and print out the loss outside of this method, so this
        method is appropriate for other kinds of scalar values indicating progress
        you'd like to use.
        """
        # TODO: Do this all on the GPU via Torch, rather than the CPU via Numpy.
        prediction = output.detach().cpu().numpy()
        ground_truth = gt_output.detach().cpu().numpy()

        # TODO: Can also include the filtering components.
        psd_1Dpred = azimuthal_average(compute_2Dpsd(prediction[0, 0, :, :]))
        psd_1Dtruth = azimuthal_average(compute_2Dpsd(ground_truth[0, 0, :, :]))

        primary_metric = mean_squared_error(psd_1Dtruth, psd_1Dpred)

        # Note: lower values are better for our primary metric here.
        return primary_metric

    def is_higher_better_primary_metric(self):
        return False

    def get_primary_metric_name(self):
        return 'MSE of azimuthal average'

    def generate_supporting_metrics(self, optional_debug_data, output, input_data, gt_output, epoch,
                                    train):
        """ Print debugging details on the final batch per epoch during training or testing. """
        super(VirtualTelescopePipeline, self).generate_supporting_metrics(
            optional_debug_data, output, input_data, gt_output, epoch, train)

        input_data = input_data.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        gt_output = gt_output.detach().cpu().numpy()

        # Print each of the input channels on their own line.
        fig = plt.figure()
        pos = 0
        for i in range(self.num_channels - 1):
            pos += 1

            ax = fig.add_subplot(self.num_channels + 1, 1, pos)
            fig.subplots_adjust(top=3.0)
            ax.set_title('Input Channel {}'.format(i + 1))
            ax.axis('off')
            ax.imshow(input_data[0][i], vmin=0, vmax=1)

        pos += 1

        ax = fig.add_subplot(self.num_channels + 1, 1, pos)
        fig.subplots_adjust(top=3.0)
        ax.set_title('Predicted Output Channel {}'.format(self.num_channels - 1))
        ax.axis('off')
        ax.imshow(output[0][0], vmin=0, vmax=1)

        pos += 1

        ax = fig.add_subplot(self.num_channels + 1, 1, pos)
        fig.subplots_adjust(top=3.0)
        ax.set_title('Ground Truth Output Channel {}'.format(self.num_channels - 1))
        ax.axis('off')
        ax.imshow(gt_output[0][0], vmin=0, vmax=1)

        img_file = os.path.join(self.results_path, '{}_debug_sample.png'.format(
            format_graph_prefix(epoch, self.exp_name)))
        plt.savefig(img_file, bbox_inches='tight')
        plt.close()
        _logger.info('Debug sample saved to {}'.format(img_file))

        # Get the similarity values between the predicted and ground truth outputs.
        # TODO: Do all these operations on the GPU with Torch, not the CPU via Numpy.
        struc_sim = structural_sim(output[0][0], gt_output[0][0])
        pix_sim = pixel_sim(output[0][0], gt_output[0][0])
        #   sift_sim = sift_sim(output[0][0], gt_output[0][0])
        #  emd = earth_movers_distance(output[0][0], gt_output[0][0])
        _logger.info('Structural similarity for single sample: {}'.format(struc_sim))
        _logger.info('Pixel similarity for single sample: {}'.format(pix_sim))
        