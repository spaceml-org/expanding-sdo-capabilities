"""
Utilities to create fake mock data.
"""
import logging
import math

import numpy as np

import torch


_logger = logging.getLogger(__name__)


def create_noise_image(num_channels, scaled_height, scaled_width,
                       base_val=0.5, percent_jitter=0.20):
    """
    Generate a random image composed of just noise. Note that the values
    given as defaults for base_val and percent_jitter are quick and dirty
    experimentally chosen values to test how the deep net works with
    random noise images.
    """
    rand_max = torch.rand(1)
    results = rand_max * torch.rand(num_channels, scaled_height, scaled_width,
                                    dtype=torch.float32)

    low = base_val * (1.0 - percent_jitter)
    high = base_val * (1.0 + percent_jitter)
    value = (high - low) * np.random.rand() + low
    results[:, math.floor(scaled_height / 2), math.floor(scaled_width / 2)] = value

    return results