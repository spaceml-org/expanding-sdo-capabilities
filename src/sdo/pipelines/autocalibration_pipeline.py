import logging
import os
from contexttimer import Timer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.stats as stats
from sdo.datasets.dimmed_sdo_dataset import DimmedSDO_Dataset
from sdo.io import format_graph_prefix
from sdo.models.autocalibration_models import (
  Autocalibration1,
  Autocalibration2,
  Autocalibration3,
  Autocalibration4,
  Autocalibration5,
  Autocalibration6,
  Autocalibration7,
  Autocalibration8,
  Autocalibration9,
  Autocalibration10,
  Autocalibration11,
  Autocalibration12,
  Autocalibration13,
  )
from sdo.pipelines.training_pipeline import TrainingPipeline
from sdo.pytorch_utilities import create_dataloader
from sdo.metrics.plotting import plot_regression
from sdo.metrics.hsced_loss import HeteroscedasticLoss


_logger = logging.getLogger(__name__)


class AutocalibrationPipeline(TrainingPipeline):
    def __init__(self, exp_name, model_version, actual_resolution, scaled_height,
                 scaled_width, device, data_basedir, data_inventory, instruments,
                 wavelengths, subsample, batch_size_train, batch_size_test,
                 test_ratio, log_interval, results_path, num_epochs, save_interval,
                 additional_metrics_interval, continue_training, saved_model_path, saved_optimizer_path,
                 start_epoch_at, yr_range, mnt_step, day_step, h_step, min_step, dataloader_workers, scaling, apodize,
                 optimizer_weight_decay, optimizer_lr, tolerance, min_alpha, max_alpha, noise_image,
                 threshold_black, threshold_black_value, flip_test_images, sigmoid_scale, loss):
        self.num_channels = len(wavelengths)
        self.results_path = results_path
        self.wavelengths = wavelengths
        self.tolerance = tolerance
        self.scaling = scaling
        self.loss = loss
        self.apodize = apodize

        _logger.info('Using {} channels across the following wavelengths and instruments:'.format(
            self.num_channels))
        _logger.info('Wavelengths: {}'.format(wavelengths))
        _logger.info('Instruments: {}'.format(instruments))
        _logger.info('Apodize: {}'.format(apodize))

        _logger.info('\nSetting up training dataset:')
        with Timer() as train_dataset_perf:
          train_dataset = DimmedSDO_Dataset(data_basedir=data_basedir,
                                            data_inventory=data_inventory,
                                            num_channels=self.num_channels,
                                            instr=instruments,
                                            channels=wavelengths, yr_range=yr_range,
                                            mnt_step=mnt_step, day_step=day_step,
                                            h_step=h_step, min_step=min_step,
                                            resolution=actual_resolution,
                                            subsample=subsample,
                                            normalization=0, scaling=scaling,
                                            apodize=apodize,
                                            test_ratio=test_ratio,
                                            min_alpha=min_alpha,
                                            max_alpha=max_alpha,
                                            scaled_height=scaled_height,
                                            scaled_width=scaled_width,
                                            noise_image=noise_image,
                                            threshold_black=threshold_black,
                                            threshold_black_value=threshold_black_value,
                                            shuffle=True)
        _logger.info('Total time to load training dataset: {:.1f} s'.format(
          train_dataset_perf.elapsed))

        _logger.info('\nSetting up testing dataset:')
        with Timer() as test_dataset_perf:
          test_dataset = DimmedSDO_Dataset(data_basedir=data_basedir,
                                           data_inventory=data_inventory,
                                           num_channels=self.num_channels,
                                           instr=instruments,
                                           channels=wavelengths, yr_range=yr_range,
                                           mnt_step=mnt_step, day_step=day_step,
                                           h_step=h_step, min_step=min_step,
                                           resolution=actual_resolution,
                                           subsample=subsample,
                                           normalization=0, scaling=scaling,
                                           apodize=apodize,
                                           test_ratio=test_ratio,
                                           min_alpha=min_alpha,
                                           max_alpha=max_alpha,
                                           scaled_height=scaled_height,
                                           scaled_width=scaled_width,
                                           noise_image=noise_image,
                                           threshold_black=threshold_black,
                                           threshold_black_value=threshold_black_value,
                                           flip_test_images=flip_test_images,
                                           shuffle=True, test=True)
        _logger.info('Total time to load testing dataset: {:.1f} s'.format(
          test_dataset_perf.elapsed))

        _logger.info('\nTotal time to load both train & test dataset: {:.1f} s\n'.format(
          train_dataset_perf.elapsed + test_dataset_perf.elapsed))

        train_loader = create_dataloader(train_dataset, batch_size_train,
                                         dataloader_workers, shuffle=True, train=True)
        test_loader = create_dataloader(test_dataset, batch_size_test,
                                        dataloader_workers, shuffle=True, train=False)

        model = self.create_model(model_version, scaled_height, scaled_width, device,
                                  sigmoid_scale)

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

    def create_model(self, model_version, scaled_height, scaled_width, device, sigmoid_scale):
        """
        Create the right model version for this experiment.
        """
        if model_version == 1:
            return Autocalibration1(input_shape=[self.num_channels, scaled_height,
                                                 scaled_width], output_dim=self.num_channels)
        elif model_version == 2:
            # This model allows the dimension of all the parameters to be increased by
            # 'increase_dim'
            return Autocalibration2(input_shape=[self.num_channels, scaled_height,
                                                 scaled_width], output_dim=self.num_channels,
                                    increase_dim=2)
        elif model_version == 3:
            # Uses a leaky relu instead of a sigmoid as its final activation function.
            return Autocalibration3(input_shape=[self.num_channels, scaled_height,
                                                 scaled_width],
                                    output_dim=self.num_channels)
        elif model_version == 4:
            # Scales free parameters by the size of the resolution, as well as uses
            # a leaky relu at the end.
            return Autocalibration4(input_shape=[self.num_channels, scaled_height,
                                                 scaled_width],
                                    output_dim=self.num_channels,
                                    scaled_resolution=scaled_height)
        elif model_version == 5:
            # Add more convolutional layers.
            return Autocalibration5(input_shape=[self.num_channels, scaled_height,
                                                 scaled_width],
                                    output_dim=self.num_channels,
                                    scaled_resolution=scaled_height)
        elif model_version == 6:
            # How simple can we get our network to be and still have single
            # channel input perform well?
            return Autocalibration6(input_shape=[self.num_channels, scaled_height,
                                                 scaled_width],
                                    output_dim=self.num_channels)
        elif model_version == 7:
            # How simple can we get our network to be and still have single
            # channel input perform well?
            return Autocalibration7(input_shape=[self.num_channels, scaled_height,
                                                 scaled_width],
                                    output_dim=self.num_channels,
                                    device=device)
        elif model_version == 8:
            # Take Autocalibration6 and replace the final sigmoid with a simple,
            # plain vanilla ReLU to deal with regressed brightnesses > 1.0.
            return Autocalibration8(input_shape=[self.num_channels, scaled_height,
                                                 scaled_width],
                                    output_dim=self.num_channels)
        elif model_version == 9:
            # Keep sigmoid activation function but scale its value.
            return Autocalibration9(input_shape=[self.num_channels, scaled_height,
                                                 scaled_width],
                                    output_dim=self.num_channels,
                                    sigmoid_scale=sigmoid_scale)
        elif model_version == 10:
            # Same as Autocalibration6, but use a ReLU6 activation function as
            # the final function replacing the sigmoid to get a clipped relu value.
            return Autocalibration10(input_shape=[self.num_channels, scaled_height,
                                                 scaled_width],
                                     output_dim=self.num_channels)
        elif model_version == 11:
            # Same as Autocalibration6, it implements heteroscedastic regression.
            return Autocalibration11(input_shape=[self.num_channels, scaled_height,
                                                  scaled_width],
                                     output_dim=self.num_channels)
        elif model_version == 12:
            # Same as Autocalibration11, but mean and logvar go through a different
            # fully connected layer
            return Autocalibration12(input_shape=[self.num_channels, scaled_height,
                                                  scaled_width],
                                     output_dim=self.num_channels)
        elif model_version == 13:
            # Same as Autocalibration12, but ELU with alpha=0.1 used as activation
            # for log_var
            return Autocalibration13(input_shape=[self.num_channels, scaled_height,
                                                  scaled_width], output_dim=self.num_channels)

        else:
            # Note: For other model_versions, simply instantiate whatever class
            # you want to test your experiment for. You will have to update the code
            # here to reference that class, preferably in sdo.models.*, such as
            # sdo.models.Autocalibration2.
            raise Exception('Unknown model version: {}'.format(model_version))

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
        if self.loss == 'mse':
            return nn.MSELoss()(output, gt_output)
        elif self.loss == 'hsced':
            # the NN outputs the mean and the log_var when running a heteroscedastic regression
            if output.shape[0] == 2:
                return HeteroscedasticLoss()(output, gt_output)
            else:
                _logger.error('Heteroscedastic regression requires to output two values'
                              'for each channel (mean, log_var). Look at Autocalibration11 for an'
                              'example of model compatible with this loss.')
        else:
            _logger.error('Required loss not implemented')

    def calculate_primary_metric(self, epoch, output, gt_output):
        """
        Given some predicted output from a network and some ground truth, this method
        calculates the binary frequency of correct cases, where a case is considerated 
        correct if the real and predicted value differ equal less than the tolerance. 
        The mean over the batch and the channels is returned as single metric value.
        """
        # the NN outputs the mean and the log_var when running a heteroscedastic regression
        # but only the mean is relevant for this primary metric
        if output.shape[0] == 2:
            output = output[0]
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
        # the NN outputs the mean and the log_var when running a heteroscedastic regression
        # but only the mean is relevant for the following metrics
        if output.shape[0] == 2:
            log_var = output[1]
            output = output[0]
        else:
            log_var = None
        super(AutocalibrationPipeline, self).generate_supporting_metrics(
            orig_img, output, input_data, gt_output, epoch, train)

        # Generate some extra metric details that are specific to autocalibration.
        _logger.info('\n\nDetails with sample from final batch:')

        _logger.info('\noutput[:3]: {}'.format(output[:3]))
        _logger.info('\ngt_output[:3]: {}'.format(gt_output[:3]))

        rand_idx = np.random.randint(len(input_data))
        sample = orig_img[rand_idx].cpu().numpy()
        sample_dimmed = input_data[rand_idx].cpu().numpy()

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
            pred_channel_dim_factor = float(output[rand_idx, i])
            # Reconstructed means we want to apply some transformation to the dimmed image
            # to get back the original undimmed image.
            reconstructed_channel = channel_dimmed / pred_channel_dim_factor

            ax1 = fig.add_subplot(self.num_channels, 4, pos)
            fig.subplots_adjust(top=3.0)
            ax1.set_title(
                'Channel: {} (col 1: original, col 2: dimmed, col 3: reconstructed, col 4: difference)\n'
                'Dimming (true): {}, dimming (predicted): {}'.format(i+1, gt_output[rand_idx, i],
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

        # TODO move dimming values figure below into a plotting function
        fig = plt.figure()
        dim_factors_numpy = gt_output[rand_idx].view(-1).cpu().numpy()
        plt.scatter(range(1, self.num_channels + 1), dim_factors_numpy,
                    label='Dimming factors (true)')
        output_numpy = output[rand_idx].detach().view(-1).cpu().numpy()
        if log_var is not None:
            log_var_numpy = log_var[rand_idx].detach().view(-1).cpu().numpy()
            sigma_numpy = np.sqrt(np.exp(log_var_numpy))
            plt.errorbar(range(1, self.num_channels + 1), output_numpy,
                         yerr=sigma_numpy, label='Dimming factors (predicted)')
        else:
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
        for c in range(self.num_channels):
            pr_coeff.append(stats.pearsonr(
                output_numpy[:, c], gt_output_numpy[:, c])[0])
        df_pr_coeff = pd.DataFrame(
            dict(zip(self.wavelengths, pr_coeff)), index=[0])
        _logger.info('\n\nPearson coefficient values by channel \n {}'
                     .format(df_pr_coeff))
        _logger.info('Mean Pearson coefficient {}'.format(np.mean(pr_coeff)))
        _logger.info('\n')

    def save_predictions(self, epoch, gt_outputs, outputs, optional_debug_data, train=True):
        """
        This method saves the ground truth and the predictions for
        post-modelling analysis,. It is called at
        least once for training and once for test.
        """
        predictions_filename, stacked_factors, _ = super(AutocalibrationPipeline, self).save_predictions(
            epoch, gt_outputs, outputs, optional_debug_data, train=train)
        _logger.info('Saving ground truths and predictions to {}...'.
                     format(predictions_filename))
        np.save(predictions_filename, stacked_factors)
