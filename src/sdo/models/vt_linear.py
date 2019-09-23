"""
This module contains a simple linear regression model. It is used as baseline for the virtual telescope.
"""
import logging
import torch.nn as nn

_logger = logging.getLogger(__name__)

class linearRegression(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(input_shape, output_shape)

    def forward(self, x):
        out = self.linear(x)
        return out
    