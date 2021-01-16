from abc import ABCMeta, abstractmethod
import math
import logging
import os
from contexttimer import Timer
from operator import itemgetter 
import numpy as np
import torch
from sdo.io import format_graph_prefix
from sdo.metrics.plotting import plot_loss, plot_primary_metric


_logger = logging.getLogger(__name__)


class TrainingPipeline(object):
    __metaclass__ = ABCMeta

    def __init__(self, exp_name, train_dataset, test_dataset, train_loader, test_loader,
                 batch_size_train, batch_size_test, model, optimizer, log_interval,
                 results_path, num_epochs, device, save_interval, additional_metrics_interval,
                 continue_training, saved_model_path, saved_optimizer_path, start_epoch_at, scaling):
        self.exp_name = exp_name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.model = model
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.results_path = results_path
        self.num_epochs = num_epochs
        self.device = device
        self.save_interval = save_interval
        self.additional_metrics_interval = additional_metrics_interval
        self.start_epoch_at = start_epoch_at
        self.scaling = scaling

        if continue_training:
            self.load_saved_checkpoint(model, saved_model_path,
                                       optimizer, saved_optimizer_path,
                                       start_epoch_at)

    @abstractmethod
    def show_sample(self, loader):
        """ Override to show some sample data at the beginning of the training process.
            to aid debugging. """
        pass

    @abstractmethod
    def get_loss_func(self, output, gt_output):
        """ Override to provide a loss function. """
        pass

    @abstractmethod
    def calculate_primary_metric(self, epoch, output, gt_output):
        """
        Given some predicted output from a network and some ground truth, this method
        calculates a scalar on how "well" we are doing for a given problem to gauge
        progress during different experiments and during training. Note that we
        already calculate and print out the loss outside of this method, so this
        method is appropriate for other kinds of scalar values indicating progress
        you'd like to use.
        """
        pass

    @abstractmethod
    def is_higher_better_primary_metric(self):
        """
        For the value returned from calculate_primary_metric(), a True value from
        this method indicates that higher values are better, while a False return
        result indicates that lower values are better.
        """
        pass

    @abstractmethod
    def get_primary_metric_name(self):
        """
        This method returns a string containing the name of the metric suitable for
        printing out into a log giving a very brief name for what the primary metric
        is tracking.
        """
        return 'Primary metric'

    def generate_supporting_metrics(self, optional_debug_data, output, input_data, gt_output,
                                    epoch, train):
        """
        Every log_interval pass during training or testing in an epoch, we might want to
        calculate supporting metrics and graphs to know how we are doing.
        """
        # TODO: There are a fair number of standard metrics across different projects we
        # can calculate here. Subclasses can also override this to add some extra metrics
        # that are specific to sub-projects that they also want to generate.
        pass

    def train(self, epoch, next_to_last_epoch=False, final_epoch=False):
        with Timer() as epoch_perf:
            _logger.info('\n\n')
            _logger.info("===================================\n\n\n\n")
            _logger.info("Training epoch {}\n".format(epoch))
            # Indicate to PyTorch that we are in training mode.
            self.model.train()
            losses = []
            total_primary_metrics = []
            gt_outputs = []
            outputs = []
            l_optional_data =[]
            for batch_idx, (input_data,
                            gt_output,
                            optional_debug_data) in enumerate(self.train_loader):
                with Timer() as iteration_perf:
                    self.optimizer.zero_grad()
                    # Send the entire batch to the GPU as one to increase efficiency.
                    input_data = input_data.to(self.device)
                    gt_output = gt_output.to(self.device)
                    output = self.model(input_data)
                    output = output.to(self.device)
                    loss = self.get_loss_func(output, gt_output)
                    loss.backward()
                    self.optimizer.step()

                    # Make sure that saving outputs don't end up with gradient tracking.
                    gt_outputs.append(gt_output.detach().clone())
                    outputs.append(output.detach().clone())
                    losses.append(float(loss))
                    l_optional_data.append(optional_debug_data)

                if batch_idx % self.log_interval == 0:
                    with torch.no_grad():
                        primary_metric = self.calculate_primary_metric(
                            epoch, output, gt_output)
                        total_primary_metrics.append(primary_metric)
                        self.print_epoch_details(epoch, batch_idx, self.batch_size_train,
                                                 self.train_dataset, loss, primary_metric,
                                                 iteration_perf.elapsed,
                                                 next_to_last_epoch=False, train=True)

        with torch.no_grad():
            self.print_epoch_details(epoch, batch_idx, self.batch_size_train,
                                     self.train_dataset, np.mean(losses), primary_metric,
                                     epoch_perf.elapsed, next_to_last_epoch=True, train=True)

            # Generate extra metrics useful for debugging and analysis.
            # TODO: We are incorrectly doing this on the final batch; fix.
            # TODO: We seem to be doing this an interval too soon redundantly
            if (epoch % self.additional_metrics_interval == 0) or next_to_last_epoch:
                with Timer() as generate_support_perf:
                    self.generate_supporting_metrics(optional_debug_data, output, input_data,
                                                     gt_output, epoch, train=True)
                _logger.info('\nTotal time to generate supporting training metrics: {:.1f} s'.format(
                    generate_support_perf.elapsed))

            if (epoch % self.save_interval == 0) or final_epoch:
                with Timer() as saving_results_perf:
                    self.save_training_results(epoch)
                    self.save_predictions(epoch, gt_outputs, outputs, l_optional_data, train=True)
                _logger.info('\nTotal time to save training results: {:.1f} s'.format(
                    saving_results_perf.elapsed))

            return np.mean(losses), np.mean(total_primary_metrics)

    def test(self, epoch, next_to_last_epoch=False, final_epoch=False):
        _logger.info("\n\nTesting epoch {}\n".format(epoch))
        with Timer() as epoch_perf, torch.no_grad():
            # Indicate to PyTorch that we are in testing mode.
            self.model.eval()
            losses = []
            total_primary_metrics = []
            times = []
            gt_outputs = []
            outputs = []
            l_optional_data =[]
            correct = 0
            for batch_idx, (input_data,
                            gt_output,
                            optional_debug_data) in enumerate(self.test_loader):
                with Timer() as iteration_perf:
                    # Send the entire batch to the GPU as one to increase efficiency.
                    input_data = input_data.to(self.device)
                    gt_output = gt_output.to(self.device)
                    gt_outputs.append(gt_output)
                    output = self.model(input_data)
                    outputs.append(output)
                    output = output.to(self.device)
                    loss = self.get_loss_func(output, gt_output)
                    losses.append(float(loss))
                    l_optional_data.append(optional_debug_data)

                if batch_idx % self.log_interval == 0:
                    primary_metric = self.calculate_primary_metric(
                        epoch, output, gt_output)
                    total_primary_metrics.append(primary_metric)
                    self.print_epoch_details(epoch, batch_idx, self.batch_size_test,
                                             self.test_dataset, loss, primary_metric,
                                             iteration_perf.elapsed,
                                             next_to_last_epoch=False, train=False)

        self.print_epoch_details(epoch, batch_idx, self.batch_size_test,
                                 self.test_dataset, np.mean(
                                     losses), primary_metric,
                                 epoch_perf.elapsed,
                                 next_to_last_epoch=True, train=False)

        # TODO: We are incorrectly doing this on the final batch; fix.
        # Generate extra metrics useful for debugging and analysis.
        # TODO: We seem to be doing this an interval too soon redundantly
        if (epoch % self.additional_metrics_interval == 0) or next_to_last_epoch:
            self.generate_supporting_metrics(optional_debug_data, output,
                                             input_data, gt_output,
                                             epoch, train=False)

        if (epoch % self.save_interval == 0) or final_epoch:
            self.save_training_results(epoch)
            self.save_predictions(epoch, gt_outputs, outputs, l_optional_data, train=False)
        return np.mean(losses), np.mean(total_primary_metrics)

    def print_epoch_details(self, epoch, batch_idx, batch_size, dataset, loss,
                            primary_metric, time_s, next_to_last_epoch, train):
        """
        During epochs, this method prints some details every `log_interval` steps.
        """
        num_batches = math.ceil(len(dataset) / batch_size)
        if next_to_last_epoch:
            epoch_percentage_done = 100.0
            data_idx = len(dataset)
        else:
            epoch_percentage_done = int(
                100.0 * float(batch_idx + 1) / num_batches)
            data_idx = (batch_idx + 1) * batch_size
        batch_info = '{}/{} ({}%)'.format(
            data_idx,
            len(dataset),
            epoch_percentage_done,
        )

        if next_to_last_epoch:
            _logger.info('Summary: {} Epoch: {} [{}]\t'
                         'Mean loss: {:.6f}, {}: {:.2f}, '
                         'Time to run: {:.1f} s'.format(
                'Train' if train else 'Test',
                epoch,
                batch_info,
                float(loss),
                self.get_primary_metric_name(),
                primary_metric,
                time_s))
        else:
            _logger.info('{} Epoch: {} [{}]\t'
                         'Loss: {:.6f}, {}: {:.2f}, '
                         'Time to run: {:.1f} s'.format(
                'Train' if train else 'Test',
                epoch,
                batch_info,
                float(loss),
                self.get_primary_metric_name(),
                primary_metric,
                time_s))

    def load_saved_checkpoint(self, model, model_path, optimizer, optimizer_path,
                              start_epoch_at):
        """ Given some saved model and optimizer, load and return them. """
        model_state_dict = torch.load(model_path)
        model.load_state_dict(model_state_dict)

        optimizer_state_dict = torch.load(optimizer_path)
        optimizer.load_state_dict(optimizer_state_dict)

        _logger.info('Restarting training at epoch {}, using model: {}, optimizer: {}'.format(
            start_epoch_at, model_path, optimizer_path))

    def save_training_results(self, epoch):
        model_filename = os.path.join(
                self.results_path, '{}_model.pth'.format(
                format_graph_prefix(epoch, self.exp_name)))
        _logger.info('\nSaving model to {}...'.format(model_filename))
        torch.save(self.model.state_dict(), model_filename)

        optimizer_filename = os.path.join(
                self.results_path, '{}_optimizer.pth'.format(
                format_graph_prefix(epoch, self.exp_name)))
        _logger.info('Saving optimizer to {}...'.format(optimizer_filename))
        torch.save(self.optimizer.state_dict(), optimizer_filename)

    def save_predictions(self, epoch, gt_outputs, outputs, l_optional_debug_data, train=True):
        """
        This method saves the ground truth and the predictions for
        post-modelling analysis. It is called at least once for training
        and once for test.
        """
        if train:
            predictions_filename = os.path.join(
                self.results_path, '{}_train_predictions.npy'.format(
                format_graph_prefix(epoch, self.exp_name)))
        else:
            predictions_filename = os.path.join(
                self.results_path, '{}_test_predictions.npy'.format(
                format_graph_prefix(epoch, self.exp_name)))
        # the NN outputs the mean and the log_var when running a heteroscedastic regression
        gt_outputs_np = torch.cat(gt_outputs).detach().cpu().numpy()
        optional_debug_data_np = torch.cat(l_optional_debug_data).detach().cpu().numpy()
        # TODO make this if more robust, checking the shape could lead to errors
        # in case we are running other models with 2 batches per epoch.
        if len(outputs[0]) == 2:
            mean_outputs = list( map(itemgetter(0), outputs)) 
            log_var_outputs = list( map(itemgetter(1), outputs))
            log_var_np = torch.cat(log_var_outputs).detach().cpu().numpy()
            outputs_np = torch.cat(mean_outputs).detach().cpu().numpy()
            # for autocal stacked factors will have shape (len_dataset, len_channels, 3)
            # In the last column, the first element is the ground truth, the 
            # second element is the predicted, the third element is the log of
            # the variance.
            stacked_factors = np.dstack((gt_outputs_np, outputs_np, log_var_np))
        else:   
            outputs_np = torch.cat(outputs).detach().cpu().numpy()
            # for autocal stacked factors will have shape (len_dataset, len_channels, 2)
            # In the last column, the first element is the ground truth, the 
            # second element is the predicted.  See save_predictions in virtual telescope 
            # pipeline for dimensions there
            stacked_factors = np.dstack((gt_outputs_np, outputs_np))
        # optional_debug_data_np has not been stacked here to not alter the autocal output
        return predictions_filename, stacked_factors, optional_debug_data_np

    def print_final_details(self, total_perf, train_losses, test_losses,
                            train_primary_metrics, test_primary_metrics):
        _logger.info('\n\nTotal testing/training time: {}'.format(
            total_perf.elapsed))

        # Print some final aggregate details at the complete end all epochs of training/testing.
        _logger.info('\nFinal training loss after {} epochs: {}'.format(
            self.num_epochs, train_losses[-1]))
        _logger.info('Final testing loss after {} epochs: {}'.format(
            self.num_epochs, test_losses[-1]))

        _logger.info('\nFinal best training loss: {}, encountered at epoch: {}'.format(
            np.min(train_losses), np.array(train_losses).argmin() + 1))
        _logger.info('Final best testing loss: {}, encountered at epoch: {}'.format(
            np.min(test_losses), np.array(test_losses).argmin() + 1))

        if self.is_higher_better_primary_metric():
            best = np.max
            best_arg = np.argmax
        else:
            best = np.min
            best_arg = np.argmin

        _logger.info('\nFinal best training {}: {}, encountered at epoch: {}'.format(
            self.get_primary_metric_name(),
            best(train_primary_metrics),
            best_arg(train_primary_metrics) + 1))
        _logger.info('Final best testing {}: {}, encountered at epoch: {}'.format(
            self.get_primary_metric_name(),
            best(test_primary_metrics),
            best_arg(test_primary_metrics) + 1))

    def get_ending_epoch_details(self, epoch):
        """
        Determine if we are at the next to last final epoch or the final epoch.
        We want to do the final calculations on the next to last final epoch
        to determine metrics and graphs, since the truly final epoch might not
        have a full batch, which could bias our calcuations. We also need
        to know the truly final epoch in order to save final results
        and predictions to disk.
        """
        next_to_last_epoch = (epoch == (self.start_epoch_at + self.num_epochs - 2))
        last_epoch = (epoch == (self.start_epoch_at + self.num_epochs - 1))
        return next_to_last_epoch, last_epoch

    def run(self):
        """ Actually does the train/test cycle for num_epochs. """
        self.show_sample(self.train_loader)

        train_losses = []
        test_losses = []
        train_primary_metrics = []
        test_primary_metrics = []
        with Timer() as total_perf:
            for epoch in range(self.start_epoch_at, self.start_epoch_at + self.num_epochs):
                next_to_last_epoch, last_epoch = self.get_ending_epoch_details(epoch)

                loss, primary_metric = self.train(epoch, next_to_last_epoch, last_epoch)
                train_losses.append(loss)
                train_primary_metrics.append(primary_metric)

                loss, primary_metric = self.test(epoch, next_to_last_epoch, last_epoch)
                test_losses.append(loss)
                test_primary_metrics.append(primary_metric)

                if (epoch % self.additional_metrics_interval == 0) or next_to_last_epoch:
                    plot_loss(epoch, train_losses, test_losses, self.results_path,
                              self.exp_name)
                    plot_primary_metric(epoch, train_primary_metrics, test_primary_metrics,
                                        self.results_path, self.exp_name,
                                        self.get_primary_metric_name())
                if (epoch % self.save_interval == 0) or last_epoch:
                    # We want to save at regular intervals the best values so far
                    self.print_final_details(total_perf, train_losses, test_losses,
                                 train_primary_metrics, test_primary_metrics)
