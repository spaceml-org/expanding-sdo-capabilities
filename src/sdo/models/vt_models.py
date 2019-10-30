"""
Torch architecture models for the virtual telescope.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import operator

from sdo.models.vt_unet_basic_blocks import (maxpool,
                                             conv_block_2,
                                             conv_trans_block)


_logger = logging.getLogger(__name__)


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


class AutoEncoder(nn.Module):
    """
    This is an autoencoder that reconstructs the input (input size = output size).
    """
    def __init__(self, input_shape=[3, 128, 128], hidden_dim=512):
        super(AutoEncoder, self).__init__()
        num_channels = input_shape[0]
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)

        self.dconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3)
        self.dconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3)
        self.dconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3)
        self.dconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3)
        self.dconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=num_channels, kernel_size=3)

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

        sample_encoder_input = torch.zeros(input_shape)
        sample_encoder_output = self._encoder(sample_encoder_input.unsqueeze(0))[0]
        sample_encoder_output_dim = sample_encoder_output.nelement()

        self.lin1 = nn.Linear(sample_encoder_output_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, sample_encoder_output_dim)

        print('Autoencoder architecture:')
        print('Input shape: {}'.format(input_shape))
        print('Input dim  : {}'.format(prod(input_shape)))
        print('Encoded dim: {}'.format(sample_encoder_output_dim))
        print('Hidden dim : {}'.format(hidden_dim))
        print('Learnable params: {}'.format(sum([p.numel() for p in self.parameters()])))

    def _encoder(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x, indices1 = self.pool(x)
        x = self.conv3(x)
        x, indices2 = self.pool(x)
        x = self.conv4(x)
        x, indices3 = self.pool(x)
        x = self.conv5(x)
        x, indices4 = self.pool(x)
        x = F.relu(x)
        return x, indices1, indices2, indices3, indices4

    def _decoder(self, x, indices1, indices2, indices3, indices4):
        x = self.unpool(x, indices4)
        x = self.dconv5(x)
        x = F.relu(x)
        x = self.unpool(x, indices3)
        x = self.dconv4(x)
        x = F.relu(x)
        x = self.unpool(x, indices2)
        x = self.dconv3(x)
        x = F.relu(x)
        x = self.unpool(x, indices1)
        x = self.dconv2(x)
        x = F.relu(x)
        x = self.dconv1(x)
        x = torch.relu(x)
        return x

    def forward(self, x):
        batch_size = x.shape[0]
        x, indices1, indices2, indices3, indices4 = self._encoder(x)
        Shap = x.shape
        x = x.view(batch_size, -1)
        x = self.lin1(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = x.reshape(Shap)
        x = self._decoder(x, indices1, indices2, indices3, indices4)
        return x


class VT_EncoderDecoder(nn.Module):
    """A Encoder Decoder module that is used to create one channel (e.g. AIA 211) from
       three other AIA channels (e.g. AIA 94, 193, and 171).
    """
    
    def __init__(self, input_shape=[3, 128, 128], hidden_dim=512):
        """
        Params: 
        1. input_shape: of the form of (n_channels x width x height)
        2. hidden_dim = number of hidden layers for the fully connected network.
        """
        super(VT_EncoderDecoder, self).__init__()
        num_channels = input_shape[0]
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)

        self.dconv5 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3)
        self.dconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3)
        self.dconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3)
        self.dconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3)
        self.dconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3)
        
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)# in_channels, stride
        self.unpool = nn.MaxUnpool2d(2, 2)
        
        sample_encoder_input = torch.zeros(input_shape)
        sample_encoder_output = self._encoder(sample_encoder_input.unsqueeze(0))[0]
        sample_encoder_output_dim = sample_encoder_output.nelement()
        
        self.lin1 = nn.Linear(sample_encoder_output_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, sample_encoder_output_dim)
    
        
        print('VT_EncoderDecoder architecture:')
        print('Input shape: {}'.format(input_shape))
        print('Input dim  : {}'.format(prod(input_shape)))
        print('Encoded dim: {}'.format(sample_encoder_output_dim))
        print('Hidden dim : {}'.format(hidden_dim))
        print('Learnable params: {}'.format(sum([p.numel() for p in self.parameters()])))

    def _encoder(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x,indices1 = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x, indices2 = self.pool(x)
        x = self.conv4(x)
        x = F.relu(x)
        x, indices3 = self.pool(x)
        x = self.conv5(x)
        x = F.relu(x)
        x, indices4 = self.pool(x) #
        return x, indices1, indices2, indices3, indices4

    def _decoder(self, x, indices1, indices2, indices3, indices4):
        x = self.unpool(x, indices4)#
        x =  self.dconv5(x)
        x = F.relu(x)
        x = self.unpool(x, indices3)
        x =  self.dconv4(x)
        x = F.relu(x)
        x = self.unpool(x, indices2)
        x =  self.dconv3(x)
        x = F.relu(x)
        x = self.unpool(x, indices1)
        x = self.dconv2(x)
        x = F.relu(x)
        x = self.dconv1(x)
        x = torch.relu(x)
        return x

    def forward(self, x):
        batch_size = x.shape[0]
        x, indices1, indices2, indices3, indices4 = self._encoder(x)
        Shap=x.shape
        x = x.view(batch_size, -1)
        x = self.lin1(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = x.reshape(Shap)
        x = self._decoder(x, indices1, indices2, indices3, indices4)
        return x


class VT_BasicEncoder(nn.Module):
    """A Encoder Decoder module that is used to create one channel (e.g. AIA 211) from
       three other AIA channels (e.g. AIA 94, 193, and 171).
    """
    
    def __init__(self, input_shape=[3, 128, 128]):
        """
        Params: 
        input_shape: of the form of (n_channels x width x height)
        """
        super(VT_BasicEncoder, self).__init__()
        num_channels = input_shape[0]
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0)
        
        print('VT_EncoderDecoder architecture:')
        print('Input shape: {}'.format(input_shape))
        print('Input dim  : {}'.format(prod(input_shape)))
    
        print('Learnable params: {}'.format(sum([p.numel() for p in self.parameters()])))

    def _encoder(self, x):
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = F.relu(x)
        x = self.conv2(x)
        #print(x.shape)
        x = F.relu(x)
        x = self.conv3(x)
        #print(x.shape)
        x = F.relu(x)
        return x 

    def forward(self,x):
        batch_size = x.shape[0]
        x = self._encoder(x)
        return x


class VT_UnetGenerator(nn.Module):
    """
    modified from https://github.com/GunhoChoi/Kind-PyTorch-Tutorial/blob/master/12_Semantic_Segmentation/UNet.py
    """
    def __init__(self, input_shape=[3, 128, 128], num_filter=64, LR_neg_slope=0.2):
        super(VT_UnetGenerator, self).__init__()
        self.in_dim = input_shape[0]
        self.out_dim = 1
        self.num_filter = num_filter
        act_fn = nn.LeakyReLU(LR_neg_slope, inplace=True)

        _logger.info("\n------Initiating U-Net------\n")
        # contracting phase
        self.down_1 = conv_block_2(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = conv_block_2(
            self.num_filter*1, self.num_filter*2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = conv_block_2(
            self.num_filter*2, self.num_filter*4, act_fn)
        self.pool_3 = maxpool()
        self.down_4 = conv_block_2(
            self.num_filter*4, self.num_filter*8, act_fn)
        self.pool_4 = maxpool()

        self.bridge = conv_block_2(
            self.num_filter*8, self.num_filter*16, act_fn)

        # expansion phase
        self.trans_1 = conv_trans_block(
            self.num_filter*16, self.num_filter*8, act_fn)
        self.up_1 = conv_block_2(self.num_filter*16, self.num_filter*8, act_fn)
        self.trans_2 = conv_trans_block(
            self.num_filter*8, self.num_filter*4, act_fn)
        self.up_2 = conv_block_2(self.num_filter*8, self.num_filter*4, act_fn)
        self.trans_3 = conv_trans_block(
            self.num_filter*4, self.num_filter*2, act_fn)
        self.up_3 = conv_block_2(self.num_filter*4, self.num_filter*2, act_fn)
        self.trans_4 = conv_trans_block(
            self.num_filter*2, self.num_filter*1, act_fn)
        self.up_4 = conv_block_2(self.num_filter*2, self.num_filter*1, act_fn)

        self.out = nn.Sequential(
            nn.Conv2d(self.num_filter, self.out_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, data_input):
        # the variables in the contracting phase are capture in order to have an handle
        # to create skip connections during the expansion phase
        down_1 = self.down_1(data_input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_4], dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_3], dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_2], dim=1)
        up_3 = self.up_3(concat_3)
        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4, down_1], dim=1)
        up_4 = self.up_4(concat_4)
        out = self.out(up_4)
        return out
