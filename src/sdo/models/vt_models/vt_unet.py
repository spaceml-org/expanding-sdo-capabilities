import logging

import torch
import torch.nn as nn
from sdo.models.vt_models.vt_unet_basic_blocks import (maxpool, conv_block_2, 
                                                       conv_block_2_sym, conv_trans_block)

_logger = logging.getLogger(__name__)


class VT_UnetGenerator(nn.Module):
    """
    Unet with parametrizable depth of the network. 
    Base architecture taken from from https://github.com/GunhoChoi/Kind-PyTorch-Tutorial/blob/master/12_Semantic_Segmentation/UNet.py
    """

    def __init__(self, input_shape=[3, 128, 128],
                 num_filter=64, LR_neg_slope=0.2,
                 depth=4):
        super(VT_UnetGenerator, self).__init__()
        self.in_dim = input_shape[0]
        self.out_dim = 1
        self.num_filter = num_filter
        self.depth = depth
        act_fn = nn.LeakyReLU(LR_neg_slope, inplace=True)

        _logger.info("\n------Initiating U-Net------\n")
        # contracting phase
        self.down = [conv_block_2(self.in_dim, self.num_filter, act_fn)]
        self.pool = [maxpool()]
        alpha = [1, 2]
        for i in range(1, self.depth):
            down = conv_block_2(self.num_filter*alpha[0], self.num_filter*alpha[1], act_fn)
            # the self.pool list could be eliminated given each element is the same
            pool = maxpool()
            self.down.append(down)
            self.pool.append(pool)
            alpha = [k * 2 for k in alpha]
        self.bridge = conv_block_2(self.num_filter*alpha[0], self.num_filter*alpha[1], act_fn)
        # expansion phase
        self.trans = []
        self.up = []
        for i in range(0, self.depth):
            trans = conv_trans_block(
                int(self.num_filter*alpha[1]), int(self.num_filter*alpha[0]), act_fn)
            up = conv_block_2(int(self.num_filter*alpha[1]), int(self.num_filter*alpha[0]), act_fn)
            self.trans.append(trans)
            self.up.append(up)
            alpha = [k / 2 for k in alpha]
        self.down = nn.ModuleList(self.down)
        self.pool = nn.ModuleList(self.pool)
        self.trans = nn.ModuleList(self.trans)
        self.up = nn.ModuleList(self.up)
        self.out = nn.Sequential(
            nn.Conv2d(self.num_filter, self.out_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, data_input):
        # the variables in the contracting phase are capture in order to have an handle
        # to create skip connections during the expansion phase
        down = []
        pool = []
        down.append(self.down[0](data_input))
        pool.append(self.pool[0](down[0]))

        for i in range(1, self.depth):
            down.append(self.down[i](pool[i-1]))
            pool.append(self.pool[i](down[i]))

        bridge = self.bridge(pool[self.depth - 1])

        trans = self.trans[0](bridge)
        concat = torch.cat([trans, down[self.depth - 1]], dim=1)
        up = self.up[0](concat)
        down_i = self.depth - 2
        for i in range(1, self.depth):
            trans = self.trans[i](up)
            concat = torch.cat([trans, down[down_i]], dim=1)
            up = self.up[i](concat)
            down_i = down_i - 1

        out = self.out(up)
        return out

class VT_UnetGenerator2(nn.Module):
    """
    This is almost the same of VT_UnetGenerator with depth 4. The only difference is the convolutional block used in every
    phase. The block here contains an activation function after the second convolutional layer. One good reason
    to include this activation function is making sure the concatenation step include all values went through the
    same activation function.
    """

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

class VT_UnetGenerator_Alternate(nn.Module):
    """
    Modified from https://github.com/GunhoChoi/Kind-PyTorch-Tutorial/blob/master/12_Semantic_Segmentation/UNet.py
    
    This architecture was implemented by Brad in 2019. This architecture is necessary to load the
    saved model for channel 94.
    """

    def __init__(self, input_shape=[3, 128, 128], num_filter=64, LR_neg_slope=0.2):
        super(VT_UnetGenerator_Alternate, self).__init__()
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
