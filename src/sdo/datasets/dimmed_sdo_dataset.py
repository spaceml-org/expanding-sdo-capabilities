"""
A version of the SDO dataset that can synthetically dim its brightness
to aid training.
"""
import torch

from sdo.datasets.sdo_dataset import SDO_Dataset


class DimmedSDO_Dataset(SDO_Dataset):
    def __init__(self, num_channels, min_alpha, *args, **kwargs):
        super(DimmedSDO_Dataset, self).__init__(*args, **kwargs)
        self.num_channels = num_channels
        self.min_alpha = min_alpha

    def __getitem__(self, idx):
        orig_img = super(DimmedSDO_Dataset, self).__getitem__(idx)
        # Note: If scaling==True, then orig_img is already scaled here.
        dimmed_img = orig_img.clone()

        dim_factor = torch.zeros(self.num_channels)
        while any(dim_factor < self.min_alpha):
            dim_factor = torch.rand(self.num_channels)

        for c in range(self.num_channels):
            dimmed_img[c] = dimmed_img[c] * dim_factor[c]

        # Note: For efficiency reasons, don't send each item to the GPU;
        # rather, later, send the entire batch to the GPU.
        return dimmed_img, dim_factor, orig_img
