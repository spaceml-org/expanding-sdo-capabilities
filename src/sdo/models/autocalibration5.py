"""
Defines our autocalibration architecture.
"""
import logging

import torch
import torch.nn as nn


_logger = logging.getLogger(__name__)

# Add more convolutional layers.
class Autocalibration5(nn.Module):
    def __init__(self, input_shape, output_dim, scaled_resolution):
        super.__init__()
        if len(input_shape) != 3:
            raise ValueError('Expecting an input_shape representing dimensions CxHxW')
        self()._input_channels = input_shape[0]
        _logger.info('input_channels: {}'.format(self._input_channels))

        conv1_in = self._input_channels
        conv1_out = int(scaled_resolution / 2)
        self._conv2d1 = nn.Conv2d(in_channels=conv1_in,
                                  out_channels=conv1_out,
                                  kernel_size=3)
        _logger.info('\nCNN1, in_channels: {}'.format(conv1_in))
        _logger.info('CNN1, out_channels: {}'.format(conv1_out))

        conv2_in = int(scaled_resolution / 2)
        conv2_out = scaled_resolution
        self._conv2d2 = nn.Conv2d(in_channels=conv2_in,
                                  out_channels=conv2_out,
                                  kernel_size=3)
        _logger.info('\nCNN2, in_channels: {}'.format(conv2_in))
        _logger.info('CNN2, out_channels: {}'.format(conv2_out))

        conv3_in = scaled_resolution
        conv3_out = scaled_resolution * 2
        self._conv2d3 = nn.Conv2d(in_channels=conv3_in,
                                  out_channels=conv3_out,
                                  kernel_size=3)
        _logger.info('\nCNN3, in_channels: {}'.format(conv3_in))
        _logger.info('CNN3, out_channels: {}'.format(conv3_out))

        cnn_output_dim = torch.zeros(input_shape).unsqueeze(0)
        _logger.info('\ncnn_output_dim.shape: {}'.format(cnn_output_dim.shape))
        cnn_output_nelems = self._cnn(cnn_output_dim).nelement()
        _logger.info('cnn_output_nelems: {}'.format(cnn_output_nelems))

        fc1_in = cnn_output_nelems
        fc1_out = scaled_resolution * 2
        self._fc1 = nn.Linear(fc1_in, fc1_out)
        _logger.info('\nFCN1, in_channels: {}'.format(fc1_in))
        _logger.info('FCN1, out_channels: {}'.format(fc1_out))

        fc2_in = scaled_resolution * 2
        fc2_out = output_dim
        self._fc2 = nn.Linear(fc2_in, fc2_out)
        _logger.info('FCN2, in_channels: {}'.format(fc2_in))
        _logger.info('FCN2, out_channels: {}'.format(fc2_out))

    def _cnn(self, x):
        x = self._conv2d1(x)
        x = torch.relu(x)
        x = nn.MaxPool2d(kernel_size=3)(x)

        x = self._conv2d2(x)
        x = torch.relu(x)
        x = nn.MaxPool2d(kernel_size=3)(x)

        # Note: no relu on the final convolution.
        x = self._conv2d3(x)
        x = nn.MaxPool2d(kernel_size=3)(x)

        return x
    
    def forward(self, x):
        batch_dim = x.shape[0]
        x = self._cnn(x).view(batch_dim, -1)

        x = self._fc1(x)
        x = torch.relu(x)

        x = self._fc2(x)
        x = torch.nn.LeakyReLU()(x)

        return x