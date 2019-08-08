"""
Defines our autocalibration architecture.
"""
import logging

import torch
import torch.nn as nn


_logger = logging.getLogger(__name__)

# TODO: Put all the autocalibration model versions into a single file.
class Autocalibration1(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        if (len(input_shape) != 3):
            raise ValueError('Expecting an input_shape representing dimensions CxHxW')
        self._input_channels = input_shape[0]
        _logger.info('input_channels: {}'.format(self._input_channels))
        self._conv2d1 = nn.Conv2d(in_channels=self._input_channels, out_channels=64, kernel_size=3)
        self._conv2d1_maxpool = nn.MaxPool2d(kernel_size=3)
        self._conv2d2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self._conv2d2_maxpool = nn.MaxPool2d(kernel_size=3)
        self._cnn_output_dim = self._cnn(torch.zeros(input_shape).unsqueeze(0)).nelement()
        _logger.info('cnn_output_dim: {}'.format(self._cnn_output_dim))
        self._fc1 = nn.Linear(self._cnn_output_dim, 256)
        self._fc2 = nn.Linear(256, output_dim)
        
    def _cnn(self, x):
        x = self._conv2d1(x)
        x = torch.relu(x)
        #x = nn.MaxPool2d(kernel_size=3)(x)
        x = self._conv2d1_maxpool(x)
        x = self._conv2d2(x)
        #x = nn.MaxPool2d(kernel_size=3)(x)
        x = self._conv2d2_maxpool(x)
        return x
    
    def forward(self, x):
        batch_dim = x.shape[0]
        x = self._cnn(x).view(batch_dim, -1)
        x = self._fc1(x)
        x = torch.relu(x)
        x = self._fc2(x)
        x = torch.sigmoid(x)
        return x