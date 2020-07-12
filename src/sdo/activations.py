import torch
import torch.nn as nn


def positive_elu(x, alpha):
    """An elu translated of +1. This provides an activation function that:
    - is not capped at 1
    - never returns negative values
    - never returns exactly 0
    """
    return torch.nn.ELU(alpha)(x) + 1


class PositiveELU(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return positive_elu(x, self.alpha)
