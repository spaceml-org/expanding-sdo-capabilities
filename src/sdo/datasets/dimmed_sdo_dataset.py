"""
A version of the SDO dataset that can synthetically dim its brightness
to aid training.
"""
import torch

from sdo.datasets.sdo_dataset import SDO_Dataset


class DimmedSDO_Dataset(SDO_Dataset):
    def __init__(self, num_channels, return_random_dim,
                 norm_by_orig_img_max, norm_by_dimmed_img_max,
                 min_alpha, *args, **kwargs):
        super(DimmedSDO_Dataset, self).__init__(*args, **kwargs)
        self.num_channels = num_channels
        self.return_random_dim = return_random_dim
        self.norm_by_orig_img_max = norm_by_orig_img_max
        self.norm_by_dimmed_img_max = norm_by_dimmed_img_max
        self.min_alpha = min_alpha

    def __getitem__(self, idx):
        orig_img = super(DimmedSDO_Dataset, self).__getitem__(idx)
        # If scaling==True, then orig_img is already scaled here.
        dimmed_img = orig_img.clone()

        dim_factor = torch.zeros(self.num_channels)
        while any(dim_factor < self.min_alpha):
            dim_factor = torch.rand(self.num_channels)

        dim_factor = torch.rand(self.num_channels)
        for c in range(self.num_channels):
            dimmed_img[c] *= dim_factor[c]

        # TODO: Allow this 'max' scaling to happen in the SDODataset class rather
        # than externally here.

        # Should we divide our image by the max of the original image or
        # by the max of the dimmed image?
        max_value = None
        if self.norm_by_orig_img_max:
            assert not self.norm_by_dimmed_img_max, \
                'You can not have both norm_by_orig_img_max and norm_by_dimmed_img_max True'
            max_value = orig_img.max()
        elif self.norm_by_dimmed_img_max:
            assert not self.norm_by_orig_img_max, \
                'You can not have both norm_by_orig_img_max and norm_by_dimmed_img_max True'
            max_value = dimmed_img.max()

        if max_value is None:
            # Scaling already happened in the superclass with scaling=True.
            normed_dimmed_img = dimmed_img
            normed_orig_img = orig_img
        else:
            # Scale the images roughly between [0.0, 1.0]
            normed_dimmed_img = dimmed_img / max_value
            normed_orig_img = orig_img / max_value

        # TODO: Remove this flag as the experiment is done.
        if self.return_random_dim:
            dim_factor = torch.rand(self.num_channels)

        # Note: For efficiency reasons, don't send each item to the GPU;
        # rather, later, send the entire batch to the GPU.
        return normed_dimmed_img, dim_factor, normed_orig_img