"""
A version of the SDO dataset that can synthetically dim its brightness
to aid training.
"""
import logging
import torch
from sdo.datasets.mock_data import create_noise_image
from sdo.datasets.sdo_dataset import SDO_Dataset


_logger = logging.getLogger(__name__)

class DimmedSDO_Dataset(SDO_Dataset):
    def __init__(self, num_channels, min_alpha, max_alpha,
                 scaled_height, scaled_width, noise_image=False,
                 threshold_black=False, threshold_black_value=0,
                 flip_test_images=False, *args, **kwargs):
        super(DimmedSDO_Dataset, self).__init__(*args, **kwargs)
        self.num_channels = num_channels
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.noise_image = noise_image
        self.scaled_height = scaled_height
        self.scaled_width = scaled_width
        self.threshold_black = threshold_black
        self.threshold_black_value = threshold_black_value
        self.flip_test_images = flip_test_images

        if self.noise_image:
            _logger.info('WARNING: We are generating random noise images!')

    def __getitem__(self, idx):
        # Note: If scaling==True, then orig_img is already scaled when
        # fetched from the superclass.
        orig_img = super(DimmedSDO_Dataset, self).__getitem__(idx)[0]

        if self.threshold_black:
            orig_img[orig_img <= self.threshold_black_value] = 0

        # If this is being used as the test dataset, in some cases we might want to
        # flip the images to ensure no stray pixels are sitting around influencing
        # the results between the training and testing datasets.
        if self.flip_test_images:
            orig_img = torch.flip(orig_img, [2])

        if self.noise_image:
            dimmed_img = create_noise_image(self.num_channels,
                                            self.scaled_height,
                                            self.scaled_width)
        else:
            dimmed_img = orig_img.clone()

        dim_factor = torch.zeros(self.num_channels)
        while any(dim_factor < self.min_alpha):
            dim_factor = self.max_alpha * torch.rand(self.num_channels)

        for c in range(self.num_channels):
            dimmed_img[c] = dimmed_img[c] * dim_factor[c]

        # Note: For efficiency reasons, don't send each item to the GPU;
        # rather, later, send the entire batch to the GPU.
        return dimmed_img, dim_factor, orig_img
