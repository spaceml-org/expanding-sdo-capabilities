"""
A version of the SDO dataset that can synthetically dim its brightness
to aid training.
"""

import torch

from sdo.sdo_dataset import SDO_Dataset


class DimmedSDO_Dataset(SDO_Dataset):
  def __init__(self, num_channels, device, *args, **kwargs):
    super(DimmedSDO_Dataset, self).__init__(device, *args, **kwargs)
    self.num_channels = num_channels
    self.device = device
    
    # TODO: Compute mean and std across dataset, and normalize them.
    
  def __getitem__(self, idx):
    imgs = super(DimmedSDO_Dataset, self).__getitem__(idx)    
    # Scale the image to between [0.0, 1.0]
    # Note: if we don't do this scaling, training and testing don't work!
    # TODO: This functionality has now been replaced inside of SDO_Dataset
    # itsel; adapt to use that instead.
    imgs = imgs / imgs.max()
    dimmed_imgs = imgs.clone().to(self.device)
    dim_factor = torch.rand(self.num_channels).to(self.device)    
    for c in range(self.num_channels):
      dimmed_imgs[c] *= dim_factor[c]
      
    return dimmed_imgs, dim_factor, imgs