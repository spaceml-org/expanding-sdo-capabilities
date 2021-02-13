import logging
from math import sqrt
import os

from contexttimer import Timer
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from sdo.datasets.virtual_telescope_sdo_dataset import VirtualTelescopeSDO_Dataset
from sdo.io import format_graph_prefix
from sdo.metrics.ssim_metric import SSIM, ssim
from sdo.metrics.extended_vt_metrics import structural_sim, pixel_sim
from sdo.models.vt_models.vt_encoders import (
    VT_EncoderDecoder,
    VT_BasicEncoder
)
from sdo.models.vt_models.vt_linear import linearRegression
from sdo.models.vt_models.vt_unet import (
    VT_UnetGenerator,
    VT_UnetGenerator2
    )
from sdo.pipelines.training_pipeline import TrainingPipeline
from sdo.pytorch_utilities import create_dataloader
from sdo.viz.plot_vt_outputs import plot_vt_sample, plot_2d_hist, plot_difference


_logger = logging.getLogger(__name__)


class VirtualTelescopePipeline(TrainingPipeline):
    def __init__(self, exp_name, model_version, actual_resolution, scaled_height,
                 scaled_width, device, data_basedir, data_inventory, instruments,
                 wavelengths, subsample, batch_size_train, batch_size_test,
                 test_ratio, log_interval, results_path, num_epochs, save_interval,
                 additional_metrics_interval, continue_training, saved_model_path, saved_optimizer_path,
                 start_epoch_at, d_events, datetime_range,yr_range, mnt_step, day_step, h_step, min_step, dataloader_workers,
                 scaling, optimizer_weight_decay, optimizer_lr, loss, unet_depth):
        self.num_channels = len(wavelengths)
        self.loss = loss
        if model_version != 3:
            _logger.warning('Unet depth is only implemented in model 3, this parameter will be ignored.')
        else:
            self.unet_depth = unet_depth

        _logger.info('Using {} channels across the following wavelengths and instruments:'.format(
            self.num_channels))
        _logger.info('Wavelengths: {}'.format(wavelengths))
        _logger.info('Instruments: {}'.format(instruments))

        _logger.info('\nSetting up training dataset:')
        with Timer() as train_dataset_perf:
            train_dataset = VirtualTelescopeSDO_Dataset(
                                        data_basedir=data_basedir,
                                        data_inventory=data_inventory,
                                        num_channels=self.num_channels,
                                        instr=instruments,
                                        channels=wavelengths, d_events=d_events,
                                        datetime_range=datetime_range, yr_range=yr_range,
                                        mnt_step=mnt_step, day_step=day_step,
                                        h_step=h_step, min_step=min_step,
                                        resolution=actual_resolution,
                                        subsample=subsample,
                                        normalization=0, scaling=scaling,
                                        test_ratio=test_ratio)
        _logger.info('Total time to load training dataset: {:.1f} s'.format(
          train_dataset_perf.elapsed))

        _logger.info('\nSetting up testing dataset:')
        with Timer() as test_dataset_perf:
            test_dataset = VirtualTelescopeSDO_Dataset(
                                       data_basedir=data_basedir,
                                       data_inventory=data_inventory,
                                       num_channels=self.num_channels,
                                       instr=instruments,
                                       channels=wavelengths,  d_events=d_events,
                                       datetime_range=datetime_range, yr_range=yr_range,
                                       mnt_step=mnt_step, day_step=day_step,
                                       h_step=h_step, min_step=min_step,
                                       resolution=actual_resolution,
                                       subsample=subsample,
                                       normalization=0, scaling=scaling,
                                       test_ratio=test_ratio, test=True)
        _logger.info('Total time to load testing dataset: {:.1f} s'.format(
          test_dataset_perf.elapsed))

        _logger.info('\nTotal time to load both train & test dataset: {:.1f} s\n'.format(
          train_dataset_perf.elapsed + test_dataset_perf.elapsed))

        train_loader = create_dataloader(train_dataset, batch_size_train,
                                         dataloader_workers, train=True)
        test_loader = create_dataloader(test_dataset, batch_size_test,
                                        dataloader_workers, train=False)

        model = self.create_model(model_version, scaled_height, scaled_width)
        
        if torch.cuda.device_count() > 1:
            _logger.info("{} GPUs are available".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            # in order to use DataParallel "module must have its parameters and buffers on device cuda:0"
            device = torch.device('cuda', 0)

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

    def create_model(self, model_version, scaled_height, scaled_width):
        """
        Create the right model version for this experiment.
        """
        if model_version == 1:
            # TODO add hidden_dim to the pipeline parameters
            return VT_EncoderDecoder(input_shape=[self.num_channels - 1, scaled_height,
                                                  scaled_width], hidden_dim=512)
        elif model_version == 2:
            return VT_BasicEncoder(input_shape=[self.num_channels - 1, scaled_height,
                                                scaled_width])
        elif model_version == 3:
            return VT_UnetGenerator(input_shape=[self.num_channels - 1, scaled_height,
                                                 scaled_width], num_filter=64, LR_neg_slope=0.2, 
                                                 depth=self.unet_depth)
        elif model_version == 4:
            return VT_UnetGenerator2(input_shape=[self.num_channels - 1, scaled_height,
                                                  scaled_width], num_filter=64, LR_neg_slope=0.2)
        elif model_version == 5:
            return linearRegression(input_shape=[self.num_channels - 1, scaled_height,
                                                 scaled_width])

        else:
            # Note: For other model_versions, simply instantiate whatever class
            # you want to test your experiment for. You will have to update the code
            # here to reference that class, preferably in sdo.models.*, such as
            # sdo.models.VT_model2.
            raise Exception('Unknown model version: {}'.format(model_version))

    def show_sample(self, loader):
        """ Show some samples for debugging purposes before training/testing. """
        input_data, gt_output, timestamp = loader.dataset[0]

        # Print each of the input channels on their own line.
        fig = plt.figure()
        pos = 0
        for i in range(self.num_channels - 1):
            pos += 1

            ax = fig.add_subplot(self.num_channels, 1, pos)
            fig.subplots_adjust(top=3.0)
            ax.set_title('Input Channel {} at time {}'.format((i + 1), timestamp))
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
        if self.loss == 'rmse':
            mse_loss = nn.MSELoss()
            return torch.sqrt(mse_loss(output, gt_output))
        elif self.loss == 'ssim':
            ssim_loss = SSIM()
            # TODO ssim can be lower than 1, to investigate if the sign matters
            # but we cannot have a flipping sign into the loss function
            return (1 - abs(ssim_loss(output, gt_output)))
        elif self.loss == 'smoothL1':
            L1_loss = nn.SmoothL1Loss()
            return L1_loss(output, gt_output)
        else:
            _logger.error('Required loss not implemented')

    def calculate_primary_metric(self, epoch, output, gt_output):
        """
        Given some predicted output from a network and some ground truth, this method
        calculates a scalar on how "well" we are doing for a given problem to gauge
        progress during different experiments and during training. Note that we
        already calculate and print out the loss outside of this method, so this
        method is appropriate for other kinds of scalar values indicating progress
        you'd like to use.
        """
        metric1 = torch.sqrt(torch.mean((output - gt_output).pow(2)))
        metric2 = torch.abs(1-ssim(output, gt_output))
        primary_metric = (metric1 + metric2)/2.
        return primary_metric.detach().cpu().item()

    def is_higher_better_primary_metric(self):
        return False

    def get_primary_metric_name(self):
        return '(RMSE + abs(1-SSIM))/2'

    def generate_supporting_metrics(self, optional_debug_data, output, input_data, gt_output, epoch,
                                    train):
        """ Print debugging details on the final batch per epoch during training or testing. """
        super(VirtualTelescopePipeline, self).generate_supporting_metrics(
            optional_debug_data, output, input_data, gt_output, epoch, train)

         # TODO: Do this all on the GPU via Torch, rather than the CPU via Numpy.
        input_data = input_data.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        gt_output = gt_output.detach().cpu().numpy()
        # np array of shape array([[year, month, day, hour, min]])
        timestamp = optional_debug_data.numpy()

        # the following plots and metrics will be computed on one single batch index
        index = 0
        img_file = os.path.join(self.results_path, '{}_debug_sample_{}.png'.format(
            format_graph_prefix(epoch, self.exp_name), ' train' if train else 'test'))
        plot_vt_sample(self.num_channels, input_data, output, gt_output, img_file, timestamp,
                       index=index)
        _logger.info('Debug sample saved to {}'.format(img_file))
        
        img_file = os.path.join(self.results_path, '{}_2dhist_{}.png'.format(
            format_graph_prefix(epoch, self.exp_name), 'train' if train else 'test'))
        # output has shape (3, 1, H, W)
        # timestamp has shape (3, 5)
        plot_2d_hist(gt_output[index][0], output[index][0], img_file, timestamp[index])
        _logger.info('2D histogram saved to {}'.format(img_file))
        
        img_file = os.path.join(self.results_path, '{}_diff_hist_map_{}.png'.format(
            format_graph_prefix(epoch, self.exp_name), 'train' if train else 'test'))
        plot_difference(gt_output[index][0], output[index][0], img_file, timestamp[index])
        
        # Get the similarity values between the predicted and ground truth outputs.
        # TODO: Do all these operations on the GPU with Torch, not the CPU via Numpy.
        struc_sim = structural_sim(output[index][0], gt_output[index][0])
        pix_sim = pixel_sim(output[index][0], gt_output[index][0])
        _logger.info('Structural similarity for single sample: {}'.format(struc_sim))
        _logger.info('Pixel similarity for single sample: {}'.format(pix_sim))
        
        # TODO: The metric below produces huge values, fix
        # Note: lower values are better for the psd metric here.
        #psd_1Dpred = azimuthal_average(compute_2Dpsd(output[index, 0, :, :]))
        #psd_1Dtruth = azimuthal_average(compute_2Dpsd(gt_output[index, 0, :, :]))
        #psd_metric = mean_squared_error(psd_1Dtruth, psd_1Dpred)
        #_logger.info('PSD metric for single sample: {}'.format(psd_metric))

    def save_predictions(self, epoch, gt_outputs, outputs, l_optional_data, train=True):
        """
        This method saves the ground truth and the predictions for
        post-modelling analysis, including the timestamps. It is called at
        least once for training and once for test.
        """
        predictions_filename, stacked_factors, timestamps_np = \
            super(VirtualTelescopePipeline, self).save_predictions(
            epoch, gt_outputs, outputs, l_optional_data, train)
        # stacked factors have shape (len_dataset, 1, 2*size_image, size_image)
        # timestamps have shape  (len_dataset, 5)
        # In the third column, the first half are the ground truth image, the second half
        # the predicted ones. This format is maintened for retro-compatibility
        _logger.info('Saving ground truths and predictions to {}...'.
                     format(predictions_filename))
        np.save(predictions_filename, stacked_factors)
        
        # reverse stack
        n_timestamps, _, _, size = stacked_factors.shape
        gt_outputs_np = stacked_factors.reshape(n_timestamps, 2, size, size)[:, 0, :, :]
        outputs_np = stacked_factors.reshape(n_timestamps, 2, size, size)[:, 1, :, :]
        
        # save binary file that contains also the timestamps
        time_predictions_filename = predictions_filename.split('.')[0] + '_timestamps.npz' 
        _logger.info('Saving timestamped ground truths and predictions to {}...'.
                     format(time_predictions_filename))
        np.savez(time_predictions_filename, gt_img=gt_outputs_np, img=outputs_np, 
                 ts=timestamps_np)
        