"""
Defines our autocalibration architecture.
"""
import logging

import torch
import torch.nn as nn


_logger = logging.getLogger(__name__)

# Can we create a separate column for processing every wavelength independently,
# combining the final filter banks from all wavelengths feeding into a single
# fully connected layer so the channels can influence each other in the final
# regression results?
class Autocalibration7(nn.Module):
    def __init__(self, input_shape, output_dim, device):
        super().__init__()
        if (len(input_shape) != 3):
            raise ValueError('Expecting an input_shape representing dimensions CxHxW')
        self._input_channels = input_shape[0]
        _logger.info('input_channels: {}'.format(self._input_channels))

        self.device = device

        num_channels = input_shape[0]
        scaled_height = input_shape[1]
        scaled_width = input_shape[2]

        # Create an independent column of CNN layers for each wavelength channel;
        # flatten each of their outputs, then feed them all into a single final
        # fully connected layer.
        self._columns = nn.ModuleList([
            self._create_column() for c in range(num_channels)
        ])

        fake_x = torch.zeros([1, scaled_height, scaled_width]).unsqueeze(0)
        cnn_output_dim = self._columns[0](fake_x).nelement()
    
        self._fc1 = nn.Linear(cnn_output_dim * num_channels, 256 * 3)
        self._fc2 = nn.Linear(256 * 3, output_dim)

    def _create_column(self):
        # Note: pytorch has 'magic' on members of the actual class, shifting them
        # into and out of the correct device, so put these column values on ourself.
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(),
        )
    
    def forward(self, x):
        batch_dim = x.shape[0]
        num_channels = x.shape[1]

        col_outs = []
        for c in range(num_channels):
            # TODO: Technically these can all run in parallel, rather than
            # serially, for better performance.
            per_channel_in = x[:, c].unsqueeze(1)
            per_channel_out = self._columns[c](per_channel_in)

            # Since the values coming out of each wavelength column are
            # unbalanced (i.e. some channels can have larger values),
            # force normalize them over the batch so that later processing
            # is not dominated by the larger channels.
            per_channel_out = per_channel_out / per_channel_out.max()
            col_outs.append(per_channel_out.view(batch_dim, -1))

        x = torch.cat(col_outs, dim=1)

        x = self._fc1(x)
        x = torch.nn.LeakyReLU()(x)

        x = self._fc2(x)
        x = torch.nn.LeakyReLU()(x)

        return x