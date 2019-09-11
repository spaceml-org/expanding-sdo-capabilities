"""
This module contains the basic blocks used by the U-net architecture in unet_generator.py
modified from
https://github.com/GunhoChoi/Kind-PyTorch-Tutorial/blob/master/12_Semantic_Segmentation/UNet.py
"""
import torch.nn as nn


def conv_block(in_dim, out_dim, act_fn):
    """Block of convolution followed by activation function"""
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        act_fn,
    )
    return model


def conv_block_2(in_dim, out_dim, act_fn):
    """Sequence of 2 convolution layers, this is used for down-sampling in the
    contraction phase"""
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
    )
    return model


def conv_trans_block(in_dim, out_dim, act_fn):
    """Block for Transpose Convolution, this is used for the up-sampling
    in the expansion phase, in combination with the pooled layer from the
    contraction phase"""
    model = nn.Sequential(nn.ConvTranspose2d(
        in_dim,out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        act_fn)
    return model


def maxpool():
    """this is used for downsampling in the contraction phase"""
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool



