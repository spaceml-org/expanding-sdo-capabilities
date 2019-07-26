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
                 model, optimizer, log_interval, results_path, num_epochs, device,
                 save_interval, continue_training, saved_model_path,
                 saved_optimizer_path, start_epoch_at):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_loader = train_loader
        self.test_loader = test_loader
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
    def get_correct_count(self, output, gt_output):
        """ Given some output and ground truth output, tries to return a notion of how
            'correct' the given predictions are so that we can compute an accuracy score. """
        pass

    def train(self, epoch, final_epoch=False):
        # TODO: IO from the SDO_Dataset is causing a bottleneck for training, making this slow.
        _logger.info('\n\n')
        _logger.info("===================================\n\n\n\n")
        _logger.info("Training epoch {}\n".format(epoch))
        self.model.train()
        losses = []
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
                _logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(orig_data), len(self.train_loader.dataset),
                    100.0 * (batch_idx / len(self.train_loader)), float(loss)))

        # Print extra debug output on the final batch.
        self.print_final_batch_details(orig_data, output, input_data, gt_output, epoch,
                                       train=True)


        _logger.info('\nAt end of train epoch {}, loss min: {}, max: {}, mean: {}'.format(epoch,
            min(losses), max(losses), np.mean(losses)))

        if epoch % self.save_interval == 0 or final_epoch:
            self.save_training_results(epoch)

        return np.mean(losses)

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

    def test(self, epoch):
        _logger.info("\n\nTesting epoch {}\n".format(epoch))
        with torch.no_grad():
            self.model.eval()
            losses = []
            correct = 0
            for batch_idx, (input_data, gt_output, orig_data) in enumerate(self.test_loader):
                input_data = input_data.to(self.device)
                gt_output = gt_output.to(self.device)
                output = self.model(input_data)
                output = output.to(self.device)
                loss = self.get_loss_func(output, gt_output)
                losses.append(float(loss))

                correct += self.get_correct_count(output, gt_output)

                if batch_idx % self.log_interval == 0:
                    _logger.info('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(orig_data), len(self.test_loader.dataset),
                        100.0 * (batch_idx / len(self.test_loader)), float(loss)))

            # Print extra debug output on the final batch.
            self.print_final_batch_details(orig_data, output, input_data, gt_output, epoch,
                                           train=False)

            accuracy = 100.0 * (correct / len(self.test_loader.dataset))
            _logger.info('\n\nEpoch {}, test set: avg. loss: {:.8f}, Accuracy: {} correct/{} total test size ({:.0f}%)'.format(
                  epoch, np.mean(losses), correct, len(self.test_loader.dataset), accuracy))

            return np.mean(losses), accuracy

    def run(self):
        """ Actually does the train/test cycle for num_epochs. """
        self.show_sample(self.train_loader)

        train_losses = []
        test_losses = []
        test_accuracies = []
        for epoch in range(self.start_epoch_at, self.start_epoch_at + self.num_epochs):
            final_epoch = True if epoch == (self.start_epoch_at + self.num_epochs - 1) else False

            train_loss = self.train(epoch, final_epoch)
            train_losses.append(train_loss)

            test_loss, test_accuracy = self.test(epoch)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            fig = plt.figure()
            plt.plot(train_losses, label='Training Loss')
            plt.plot(test_losses, label='Testing Loss')
            plt.title('Training/testing loss after {} epochs'.format(epoch))
            img_file = os.path.join(self.results_path, '{}_loss_graph.png'.format(
                format_epoch(epoch)))
            plt.savefig(img_file, bbox_inches='tight')
            plt.close()
            _logger.info('\nTraining/testing loss graph for epoch {} saved to {}'.format(
                epoch, img_file))

            fig = plt.figure()
            plt.plot(test_accuracies, label='Test Accuracies')
            plt.title('Testing accuracy after {} epochs'.format(epoch))
            img_file = os.path.join(self.results_path, '{}_training_accuracy_graph.png'.format(
                format_epoch(epoch)))
            plt.savefig(img_file, bbox_inches='tight')
            plt.close()
            _logger.info('\nTesting accuracy graph for epoch {} saved to {}'.format(
                epoch, img_file))
          
        _logger.info('\n\nFinal mean training loss after {} epochs: {}'.format(
            self.num_epochs, np.mean(train_losses)))
        _logger.info('Final mean testing loss after {} epochs: {}'.format(
            self.num_epochs, np.mean(test_losses)))
        _logger.info('Final best accuracy: {}%, encountered at epoch: {}'.format(
            round(max(test_accuracies), 1), np.array(test_accuracies).argmax() + 1))
