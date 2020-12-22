import logging
import torch
import torch.nn as nn
from sdo.models.vt_models.vt_unet_basic_blocks import (maxpool, conv_block_2,  conv_block_2_sym,
                                                       conv_trans_block, up_conv, Attention_block)

_logger = logging.getLogger(__name__)

# """https://jinglescode.github.io/2019/12/08/biomedical-image-segmentation-u-net-attention/"""


class UNet_Attention(nn.Module):
    def __init__(self,  input_shape=[3, 128, 128], num_filter=64,
                 LR_neg_slope=0.2, depth=4):
        super(UNet_Attention, self).__init__()
        self.in_dim = input_shape[0]
        self.out_dim = 1
        self.num_filter = num_filter
        self.depth = depth
        act_fn = nn.LeakyReLU(LR_neg_slope, inplace=True)

        _logger.info("\n------Initiating Attention U-Net------\n")
        # contracting phase
        self.down = [conv_block_2_sym(self.in_dim, self.num_filter, act_fn)]
        self.pool = [maxpool()]
        alpha = [1, 2]
        for i in range(1, self.depth):
            down = conv_block_2_sym(self.num_filter * alpha[0], self.num_filter * alpha[1], act_fn)
            # the self.pool list could be eliminated given each element is the same
            pool = maxpool()
            self.down.append(down)
            self.pool.append(pool)
            alpha = [k * 2 for k in alpha]
        self.up = []
        self.att = []
        self.up_conv = []
        for i in range(0, self.depth):
            up = up_conv(int(self.num_filter*alpha[1]), int(self.num_filter*alpha[0]),
                              act_fn)
            att = Attention_block(F_g=int(self.num_filter*alpha[0]), F_l=int(self.num_filter*alpha[0]),
                                  F_int=int(self.num_filter*alpha[0])/2
                                  )
            up_conv = conv_block_2_sym(self.num_filter * alpha[1], self.num_filter * alpha[0], act_fn)
            self.up.append(up)
            self.att.append(att)
            self.up_conv.append(up_conv)
            alpha = [k / 2 for k in alpha]

        self.down = nn.ModuleList(self.down)
        self.pool = nn.ModuleList(self.pool)
        self.up = nn.ModuleList(self.up)
        self.att = nn.ModuleList(self.att)
        self.up_conv = nn.ModuleList(self.up_conv)
        self.out = nn.Sequential(
            nn.Conv2d(self.num_filter, self.out_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )


    def forward(self, data_input):
        down = []
        pool = []
        down.append(self.down[0](data_input))
        pool.append(self.pool[0](down[0]))

        for i in range(1, self.depth):
            down.append(self.down[i](pool[i - 1]))
            pool.append(self.pool[i](down[i]))

        up_conv =  # handle bridge here
        for i in range(1, self.depth):
            up = self.up[i](up_conv)
            att = self.att[i](down[i-1], up)
            concat = torch.cat([up, att], dim=1)
            up_conv = self.up_conv[i](concat)

        out = self.out(up_conv)
        return out
