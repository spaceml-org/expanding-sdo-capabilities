from abc import ABCMeta, abstractmethod
import logging
import os

import matplotlib.pyplot as plt

import numpy as np

import torch

from sdo.io import format_epoch


_logger = logging.getLogger(__name__)


class TrainingPipeline(object):
    __metaclass__ = ABCMeta

    def __init__(self, train_dataset, test_dataset, train_loader, test_loader,
                 batch_size_train, batch_size_test, model, optimizer, log_interval,
                 results_path, num_epochs, device, save_interval, continue_training,
                 saved_model_path, saved_optimizer_path, start_epoch_at):
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
    def calculate_progress(self, epoch, output, gt_output):
        """
        Given some predicted output from a network and some ground truth, this method
        calculates a scalar on how "well" we are doing for a given problem to gauge
        progress during different experiments and during training. Note that we
        already calculate and print out the loss outside of this method, so this
        method is appropriate for other kinds of scalar values indicating progress
        you'd like to use. Also, note that the value returned should assume that
        higher values are 'good' and lower values are 'bad' (i.e. a progress of 5
        is 'better' than a progress of 0).
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
        # TODO: IO from the SDO_Dataset is causing a bottleneck for training, making this slow.
        _logger.info('\n\n')
        _logger.info("===================================\n\n\n\n")
        _logger.info("Training epoch {}\n".format(epoch))
        self.model.train()
        losses = []
        total_progress = []
        for batch_idx, (input_data, gt_output, orig_data) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            input_data = input_data.to(self.device)
            gt_output = gt_output.to(self.device)
            output = self.model(input_data)
            loss = self.get_loss_func(output, gt_output)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss))

            if batch_idx % self.log_interval == 0:
                progress = self.calculate_progress(epoch, output, gt_output)
                total_progress.append(progress)
                self.print_epoch_details(self.batch_size_train, orig_data, loss, progress,
                                         train=True)

        # Generate extra metrics useful for debugging and analysis.
        self.generate_supporting_metrics(orig_data, output, input_data, gt_output, epoch,
                                       train=True)


        self.print_epoch_details(self.batch_size_train, orig_data, np.mean(losses), 100.0,
                                 train=True)

        if epoch % self.save_interval == 0 or final_epoch:
            self.save_training_results(epoch)

        return np.mean(losses), np.mean(total_progress)

    def test(self, epoch):
        _logger.info("\n\nTesting epoch {}\n".format(epoch))
        with torch.no_grad():
            self.model.eval()
            losses = []
            total_progress = []
            correct = 0
            for batch_idx, (input_data, gt_output, orig_data) in enumerate(self.test_loader):
                input_data = input_data.to(self.device)
                gt_output = gt_output.to(self.device)
                output = self.model(input_data)
                output = output.to(self.device)
                loss = self.get_loss_func(output, gt_output)
                losses.append(float(loss))

                if batch_idx % self.log_interval == 0:
                    progress = self.calculate_progress(epoch, output, gt_output)
                    total_progress.append(progress)
                    self.print_epoch_details(self.batch_size_test, orig_data, loss, progress,
                                             train=False)

            # Generate extra metrics useful for debugging and analysis.
            self.generate_supporting_metrics(orig_data, output, input_data, gt_output, epoch,
                                           train=False)

            self.print_epoch_details(self.batch_size_test, orig_data, np.mean(losses), 100.0,
                                     train=False)

            return np.mean(losses), np.mean(total_progress)

    def print_epoch_details(self, batch_size, orig_data, loss, progress, train):
        """
        During epochs, this method prints some details every `log_interval` steps.
        """
        num_batches = len(orig_data) / batch_size
        epoch_percentage_done = int(100.0 * float(batch_idx) / num_batches)
        _logger.info('{} Epoch: {} [{}% ({:.2f})]\tLoss: {:.6f}'.format(
            'Train' if train else 'Test', epoch, epoch_percentage_done, progress,
            float(loss)))

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

        train_progress = []
        test_progress = []
        for epoch in range(self.start_epoch_at, self.start_epoch_at + self.num_epochs):
            final_epoch = True if epoch == (self.start_epoch_at + self.num_epochs - 1) else False

            loss, progress = self.train(epoch, final_epoch)
            train_losses.append(loss)
            train_progress.append(progress)

            loss, progress = self.test(epoch)
            test_losses.append(loss)
            test_progress.append(progress)

            fig = plt.figure()
            plt.plot(train_losses, label='Training Loss')
            plt.plot(test_losses, label='Testing Loss')
            plt.title('Training/testing loss after {} epochs'.format(epoch))
            img_file = os.path.join(self.results_path, '{}_loss_graph.png'.format(
                format_epoch(epoch)))
            plt.legend()
            plt.savefig(img_file, bbox_inches='tight')
            plt.close()
            _logger.info('\nTraining/testing loss graph for epoch {} saved to {}'.format(
                epoch, img_file))

            fig = plt.figure()
            plt.plot(train_progress, label='Training Progress')
            plt.plot(test_progress, label='Testing Progress')
            plt.title('Training/testing progress after {} epochs'.format(epoch))
            img_file = os.path.join(self.results_path, '{}_progress_graph.png'.format(
                format_epoch(epoch)))
            plt.legend()
            plt.savefig(img_file, bbox_inches='tight')
            plt.close()
            _logger.info('Training/testing progress graph for epoch {} saved to {}'.format(
                epoch, img_file))
          
        # Print some final aggregate details at the complete end all epochs of training/testing.
        _logger.info('\n\nFinal training loss after {} epochs: {}'.format(self.num_epochs, train_losses[-1])
        _logger.info('Final mean testing loss after {} epochs: {}'.format(self.num_epochs, test_losses[-1])

        _logger.info('\nFinal best training loss: {}, encountered at epoch: {}'.format(
            np.round(np.min(train_loss), decimals=4), np.array(train_loss).argmin() + 1))
        _logger.info('Final best testing loss: {}, encountered at epoch: {}'.format(
            np.round(np.min(test_loss), decimals=4), np.array(test_loss).argmin() + 1))

        _logger.info('\nFinal best training progress: {}, encountered at epoch: {}'.format(
            np.round(np.max(train_progress), decimals=1), np.array(train_progress).argmax() + 1))
        _logger.info('Final best testing progress: {}, encountered at epoch: {}'.format(
            np.round(np.max(test_progress), decimals=1), np.array(test_progress).argmax() + 1))
