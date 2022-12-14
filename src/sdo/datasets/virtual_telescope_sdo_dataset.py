"""
A version of the SDO dataset suitable for the virtual telescope
that returns data samples as (input_data_image, gt_image).
"""
import torch

from sdo.datasets.sdo_dataset import SDO_Dataset


class VirtualTelescopeSDO_Dataset(SDO_Dataset):
    def __init__(self, num_channels, *args, **kwargs):
        super(VirtualTelescopeSDO_Dataset, self).__init__(*args, **kwargs)
        self.num_channels = num_channels

    def __getitem__(self, idx):
        data_with_timestamps = super(VirtualTelescopeSDO_Dataset, self).__getitem__(idx)
        # Note: Shape is (num_channels, height, width)
        data = data_with_timestamps[0]
        timestamp = data_with_timestamps[1]

        assert data.shape[0] == self.num_channels, \
            'orig data has incorrect size: {}'.format(data.shape[0])

        last_channel_idx = self.num_channels - 1

        # Have the input image consist of n - 1 channels.
        img = data[:last_channel_idx, :, :]

        # Have the ground truth image be the very last channel we want to
        # reproduce in our encoder/decoder network.
        truth = data[last_channel_idx, :, :]
        truth = truth.unsqueeze(0)

        assert img.shape[0] == self.num_channels - 1, \
            'input img shape has incorrect size: {}'.format(img.shape[0])
        assert truth.shape[0] == 1, \
            'truth output shape has incorrect size: {}'.format(truth.shape[0])
        return img, truth, timestamp
