"""
A version of the SDO dataset that can synthetically dim its brightness
to aid training.
"""

import torch

from sdo.datasets.sdo_dataset import SDO_Dataset


class DimmedSDO_Dataset(SDO_Dataset):
  def __init__(self, num_channels, *args, **kwargs):
    super(DimmedSDO_Dataset, self).__init__(*args, **kwargs)
    self.num_channels = num_channels
    
    # TODO: Compute mean and std across dataset, and normalize them.
    
  def __getitem__(self, idx):
    imgs = super(DimmedSDO_Dataset, self).__getitem__(idx)    
    # Scale the image to between [0.0, 1.0]
    # Note: if we don't do this scaling, training and testing don't work!

    # TODO: Use the max() that sdo_dataset has calculated across the full
    # data instead of our own, because this is currently by image below.
    imgs = imgs / imgs.max()
    dimmed_imgs = imgs.clone()
    # TODO: Degrade the brightness _before_ scaling the values.
    dim_factor = torch.rand(self.num_channels)
    for c in range(self.num_channels):
      dimmed_imgs[c] *= dim_factor[c]
    
    # Note: For efficiency reasons, don't send each item to the GPU;
    # rather, later, send the entire batch to the GPU.
    return dimmed_imgs, dim_factor, imgs