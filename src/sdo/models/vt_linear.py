"""
This module contains a simple linear regression model. It is used as baseline for the virtual telescope.
"""
import logging
import torch.nn as nn

_logger = logging.getLogger(__name__)

class linearRegression(nn.Module):
    def __init__(self, input_shape=[3, 512, 512]):
        super(linearRegression, self).__init__()
        self.in_dim = input_shape[0]
        self.width = input_shape[1]
        self.height = input_shape[2]
        self.out_dim = 1
        self.linear = nn.Linear(in_features=self.in_dim, 
                                out_features=self.out_dim, 
                                bias=True)

    def forward(self, x):
        batch_dim = x.shape[0]
        x = x.view(batch_dim, self.width, self.height, self.in_dim)
        out = self.linear(x)
        out = out.view(batch_dim, self.out_dim, self.width, self.height)
        return out
    