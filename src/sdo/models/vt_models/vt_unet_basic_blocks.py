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
        in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        act_fn)
    return model


def maxpool():
    """this is used for downsampling in the contraction phase"""
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_2_sym(in_dim, out_dim, act_fn):
    """Sequence of 2 convolution layers where both the layers are followed by the 
    activation function, this is used for down-sampling in the contraction phase in 
    model 4"""
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        act_fn,
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        act_fn
    )
    return model


def up_conv(in_ch, out_ch, act_fn):
    model = nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        act_fn
    )
    return model


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out
