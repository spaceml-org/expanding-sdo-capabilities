"""
This is almost the same of vt_unet_generator.py. The only difference is the convolutional block used in every 
phase. The block here contains an activation function after the second convolutional layer. One good reason
to include this activation function is making sure the concatenation step include all values went through the
same activation function.
"""
import logging
import torch
import torch.nn as nn

from sdo.models.unet_basic_blocks import (maxpool,
                                          conv_block_2,
                                          conv_block_2_sym,
                                          conv_trans_block)

_logger = logging.getLogger(__name__)


class VT_UnetGenerator2(nn.Module):

    def __init__(self, input_shape=[3, 128, 128], num_filter=64, LR_neg_slope=0.2):
        super(VT_UnetGenerator2, self).__init__()
        self.in_dim = input_shape[0]
        self.out_dim = 1
        self.num_filter = num_filter
        act_fn = nn.LeakyReLU(LR_neg_slope, inplace=True)

        _logger.info("\n------Initiating U-Net------\n")
        # contracting phase
        self.down_1 = conv_block_2_sym(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = conv_block_2_sym(
            self.num_filter*1, self.num_filter*2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = conv_block_2_sym(
            self.num_filter*2, self.num_filter*4, act_fn)
        self.pool_3 = maxpool()
        self.down_4 = conv_block_2_sym(
            self.num_filter*4, self.num_filter*8, act_fn)
        self.pool_4 = maxpool()

        self.bridge = conv_block_2_sym(
            self.num_filter*8, self.num_filter*16, act_fn)

        # expansion phase
        self.trans_1 = conv_trans_block(
            self.num_filter*16, self.num_filter*8, act_fn)
        self.up_1 = conv_block_2_sym(self.num_filter*16, self.num_filter*8, act_fn)
        self.trans_2 = conv_trans_block(
            self.num_filter*8, self.num_filter*4, act_fn)
        self.up_2 = conv_block_2_sym(self.num_filter*8, self.num_filter*4, act_fn)
        self.trans_3 = conv_trans_block(
            self.num_filter*4, self.num_filter*2, act_fn)
        self.up_3 = conv_block_2_sym(self.num_filter*4, self.num_filter*2, act_fn)
        self.trans_4 = conv_trans_block(
            self.num_filter*2, self.num_filter*1, act_fn)
        self.up_4 = conv_block_2_sym(self.num_filter*2, self.num_filter*1, act_fn)

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
