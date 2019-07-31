"""
A version of the SDO dataset that can synthetically dim its brightness
to aid training.
"""

import torch

from sdo.datasets.sdo_dataset import SDO_Dataset


class DimmedSDO_Dataset(SDO_Dataset):
    def __init__(self, num_channels, normalization_by_max, *args, **kwargs):
        super(DimmedSDO_Dataset, self).__init__(*args, **kwargs)
        self.num_channels = num_channels
        self.normalization_by_max = normalization_by_max
    
        # TODO: Compute mean and std across dataset, and normalize them.
    
    def __getitem__(self, idx):
        orig_imgs = super(DimmedSDO_Dataset, self).__getitem__(idx)    
        dimmed_imgs = orig_imgs.clone()

        dim_factor = torch.rand(self.num_channels)
        for c in range(self.num_channels):
            dimmed_imgs[c] *= dim_factor[c]

        # TODO: Do an experiment using the max() that sdo_dataset has calculated
        # across the full data instead of our own, because this is currently by
        # image below.
        # TODO: Allow this 'max' scaling to happen in the SDODataset class rather
        # than externally here.
        if self.normalization_by_max:
            # Scale the images between [0.0, 1.0]
            dimmed_imgs = dimmed_imgs / dimmed_imgs.max()
            orig_imgs = orig_imgs / orig_imgs.max()

        # Note: For efficiency reasons, don't send each item to the GPU;
        # rather, later, send the entire batch to the GPU.
        return dimmed_imgs, dim_factor, orig_imgs