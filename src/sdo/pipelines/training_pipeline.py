from abc import ABCMeta, abstractmethod
import math
import logging
import os

import matplotlib.pyplot as plt

from contexttimer import Timer

import numpy as np

import torch

from sdo.io import format_epoch


_logger = logging.getLogger(__name__)


class TrainingPipeline(object):
    __metaclass__ = ABCMeta

    def __init__(self, exp_name, train_dataset, test_dataset, train_loader, test_loader,
                 batch_size_train, batch_size_test, model, optimizer, log_interval,
                 results_path, num_epochs, device, save_interval, continue_training,
                 saved_model_path, saved_optimizer_path, start_epoch_at):
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
        self.start_epoch_at = start_epoch_at

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
    def print_final_batch_details(self, orig_data, output, input_data, gt_output, epoch, train):
        """ Override to print some debugging details on the final batch for a given epoch
            of either the training or testing loop. """
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

    def generate_supporting_metrics(self, orig_data, output, input_data, gt_output, epoch,
                                    train):
        """
        Every log_interval pass during training or testing in an epoch, we might want to
        calculate supporting metrics and graphs to know how we are doing.
        """
        # TODO: There are a fair number of standard metrics across different projects we
        # can calculate here. Subclasses can also override this to add some extra metrics
        # that are specific to sub-projects that they also want to generate.
        pass

    def train(self, epoch, final_epoch=False):
        _logger.info('\n\n')
        _logger.info("===================================\n\n\n\n")
        _logger.info("Training epoch {}\n".format(epoch))
        self.model.train() # Indicate to PyTorch that we are in training mode.
        losses = []
        total_primary_metrics = []
        times = []
        for batch_idx, (input_data, gt_output, orig_data) in enumerate(self.train_loader):
            with Timer() as t:
                self.optimizer.zero_grad()
                # Send the entire batch to the GPU as one to increase efficiency.
                input_data = input_data.to(self.device)
                gt_output = gt_output.to(self.device)
                output = self.model(input_data)
                loss = self.get_loss_func(output, gt_output)
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss))

            times.append(t.elapsed)

            if batch_idx % self.log_interval == 0:
                primary_metric = self.calculate_primary_metric(epoch, output, gt_output)
                total_primary_metrics.append(primary_metric)
                self.print_epoch_details(epoch, batch_idx, self.batch_size_train,
                                         self.train_dataset, loss, primary_metric,
                                         t.elapsed, final_batch=False, train=True)

        # Generate extra metrics useful for debugging and analysis.
        self.generate_supporting_metrics(orig_data, output, input_data, gt_output, epoch,
                                       train=True)


        # TODO: Have a timer for the entire batch loop rather than a mean for the total
        # epoch time.
        self.print_epoch_details(epoch, batch_idx, self.batch_size_train, self.train_dataset,
                                 np.mean(losses), primary_metric, np.mean(times),
                                 final_batch=True, train=True)

        if epoch % self.save_interval == 0 or final_epoch:
            self.save_training_results(epoch)

        return np.mean(losses), np.mean(total_primary_metrics)

    def test(self, epoch):
        _logger.info("\n\nTesting epoch {}\n".format(epoch))
        with torch.no_grad():
            self.model.eval() # Indicate to PyTorch that we are in testing mode.
            losses = []
            total_primary_metrics = []
            times = []
            correct = 0
            for batch_idx, (input_data, gt_output, orig_data) in enumerate(self.test_loader):
                with Timer() as t:
                    # Send the entire batch to the GPU as one to increase efficiency.
                    input_data = input_data.to(self.device)
                    gt_output = gt_output.to(self.device)
                    output = self.model(input_data)
                    output = output.to(self.device)
                    loss = self.get_loss_func(output, gt_output)
                    losses.append(float(loss))

                times.append(t.elapsed)

                if batch_idx % self.log_interval == 0:
                    primary_metric = self.calculate_primary_metric(epoch, output, gt_output)
                    total_primary_metrics.append(primary_metric)
                    self.print_epoch_details(epoch, batch_idx, self.batch_size_test,
                                             self.test_dataset, loss, primary_metric,
                                             t.elapsed, final_batch=False, train=False)

            # Generate extra metrics useful for debugging and analysis.
            self.generate_supporting_metrics(orig_data, output, input_data, gt_output, epoch,
                                           train=False)

            # TODO: Have a timer for the entire batch loop rather than a mean for the total
            # epoch time.
            self.print_epoch_details(epoch, batch_idx, self.batch_size_test,
                                     self.test_dataset, np.mean(losses), primary_metric,
                                     t.elapsed, final_batch=True, train=False)

            return np.mean(losses), np.mean(total_primary_metrics)

    def print_epoch_details(self, epoch, batch_idx, batch_size, dataset, loss, primary_metric, time_s,
                            final_batch, train):
        """
        During epochs, this method prints some details every `log_interval` steps.
        """
        num_batches = math.ceil(len(dataset) / batch_size)
        if final_batch:
            epoch_percentage_done = 100.0
            data_idx = len(dataset)
        else:
            epoch_percentage_done = int(100.0 * float(batch_idx + 1) / num_batches)
            data_idx = (batch_idx + 1) * batch_size
        batch_info = '{}/{} ({}%)'.format(
            data_idx,
            len(dataset),
            epoch_percentage_done,
            )

        if final_batch:
            _logger.info('Summary: {} Epoch: {} [{}]\tMean loss: {:.6f}, Mean primary metric: {:.2f}, Time to run: {:.1f} s'.format(
                'Train' if train else 'Test',
                epoch,
                batch_info,
                float(loss),
                primary_metric,
                time_s))
        else:
            _logger.info('{} Epoch: {} [{}]\tLoss: {:.6f}, Primary metric: {:.2f}, Time to run: {:.1f} s'.format(
                'Train' if train else 'Test',
                epoch,
                batch_info,
                float(loss),
                primary_metric,
                time_s))

    def load_saved_checkpoint(self, model, model_path, optimizer, optimizer_path, start_epoch_at):
        """ Given some saved model and optimizer, load and return them. """
        model_state_dict = torch.load(model_path)
        model.load_state_dict(model_state_dict)

        optimizer_state_dict = torch.load(optimizer_path)
        optimizer.load_state_dict(optimizer_state_dict)

        _logger.info('Restarting training at epoch {}, using model: {}, optimizer: {}'.format(
            start_epoch_at, model_path, optimizer_path))

    def save_training_results(self, epoch):
        model_filename = os.path.join(self.results_path,
                                      'model_epoch_{}.pth'.format(epoch))
        _logger.info('\nSaving model to {}...'.format(model_filename))
        torch.save(self.model.state_dict(), model_filename)

        optimizer_filename = os.path.join(self.results_path,
                                          'optimizer_epoch_{}.pth'.format(epoch))
        _logger.info('Saving optimizer to {}...'.format(optimizer_filename))
        torch.save(self.optimizer.state_dict(), optimizer_filename)

    def run(self):
        """ Actually does the train/test cycle for num_epochs. """
        self.show_sample(self.train_loader)

        train_losses = []
        test_losses = []

        train_primary_metrics = []
        test_primary_metrics = []
        for epoch in range(self.start_epoch_at, self.start_epoch_at + self.num_epochs):
            final_epoch = True if epoch == (self.start_epoch_at + self.num_epochs - 1) else False

            loss, primary_metric = self.train(epoch, final_epoch)
            train_losses.append(loss)
            train_primary_metrics.append(primary_metric)

            loss, primary_metric = self.test(epoch)
            test_losses.append(loss)
            test_primary_metrics.append(primary_metric)

            # TODO: For all the graphs, make sure the epoch number comes before the
            # experiment name in the filename.
            # TODO: Put the experiment name into the title of the graph or as a
            # small subtitle somewhere.
            fig = plt.figure()
            plt.plot(train_losses, label='Training Loss')
            plt.plot(test_losses, label='Testing Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training/testing loss after {} epochs'.format(epoch))
            img_file = os.path.join(self.results_path, '{}_{}_loss_graph.png'.format(
                self.exp_name,
                format_epoch(epoch)))
            plt.legend()
            plt.savefig(img_file, bbox_inches='tight')
            plt.close()
            _logger.info('\nTraining/testing loss graph for epoch {} saved to {}'.format(
                epoch, img_file))

            fig = plt.figure()
            plt.plot(train_primary_metrics, label='Training Primary Metric')
            plt.plot(test_primary_metrics, label='Testing Primary Metric')
            plt.xlabel('Epoch')
            plt.ylabel('Primary metric')
            plt.title('Training/testing primary metric after {} epochs'.format(epoch))
            img_file = os.path.join(self.results_path, '{}_{}_primary_metric_graph.png'.format(
                self.exp_name,
                format_epoch(epoch)))
            plt.legend()
            plt.savefig(img_file, bbox_inches='tight')
            plt.close()
            _logger.info('Training/testing primary metric graph for epoch {} saved to {}'.format(
                epoch, img_file))
        
        # TODO: Print the total aggregate training/testing time.

        # Print some final aggregate details at the complete end all epochs of training/testing.
        _logger.info('\n\nFinal training loss after {} epochs: {:.6f}'.format(self.num_epochs, train_losses[-1]))
        _logger.info('Final testing loss after {} epochs: {:.6f}'.format(self.num_epochs, test_losses[-1]))

        _logger.info('\nFinal best training loss: {}, encountered at epoch: {}'.format(
            np.round(np.min(train_losses), decimals=4), np.array(train_losses).argmin() + 1))
        _logger.info('Final best testing loss: {}, encountered at epoch: {}'.format(
            np.round(np.min(test_losses), decimals=4), np.array(test_losses).argmin() + 1))

        if self.is_higher_better_primary_metric():
            best = np.max
            best_arg = np.argmax
        else:
            best = np.min
            best_arg = np.argmin

        # TODO: Show more decimals here.
        _logger.info('\nFinal best training primary metric: {}, encountered at epoch: {}'.format(
            np.round(best(train_primary_metrics), decimals=1), best_arg(train_primary_metrics) + 1))
        _logger.info('Final best testing primary metric: {}, encountered at epoch: {}'.format(
            np.round(best(test_primary_metrics), decimals=1), best_arg(test_primary_metrics) + 1))
