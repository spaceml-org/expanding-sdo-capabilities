import logging
import torch
import torch.nn as nn
from torch.autograd import Variable

from sdo.models.vt_models.vt_unet_basic_blocks import (maxpool, conv_block_2_sym,
                                                       up_conv_block, Attention_block)
 #from sdo.models.utils import HookBasedFeatureExtractor

_logger = logging.getLogger(__name__)

"""Modified from https://jinglescode.github.io/2019/12/08/biomedical-image-segmentation-u-net-attention/"""


class UNet_Attention(nn.Module):
    def __init__(self,  input_shape=(3, 128, 128), num_filter=64,
                 LR_neg_slope=0.2, depth=4):
        # the depth corresponds to the number of convolution blocks applied
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
            pool = maxpool()
            self.down.append(down)
            self.pool.append(pool)
            alpha = [k * 2 for k in alpha]
        self.bridge = conv_block_2_sym(self.num_filter * alpha[0], self.num_filter * alpha[1], act_fn)
        # expansion phase
        self.up = []
        self.att = []
        self.up_conv = []
        for i in range(0, self.depth):
            up = up_conv_block(int(self.num_filter*alpha[1]), int(self.num_filter*alpha[0]), act_fn)
            att = Attention_block(F_g=int(self.num_filter*alpha[0]), F_l=int(self.num_filter*alpha[0]),
                                  F_int=int(self.num_filter*alpha[0]/2)
                                 )
            up_conv = conv_block_2_sym(int(self.num_filter * alpha[1]), int(self.num_filter * alpha[0]),
                                       act_fn)
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

        bridge = self.bridge(pool[i])
        up = self.up[0](bridge)
        att = self.att[0](down[i], up)
        concat = torch.cat([att, up], dim=1)
        up_conv = self.up_conv[0](concat)
        down_i = i - 1
        for i in range(1, self.depth):
            up = self.up[i](up_conv)
            att = self.att[i](down[down_i], up)
            concat = torch.cat([up, att], dim=1)
            up_conv = self.up_conv[i](concat)
            down_i = down_i - 1
        out = self.out(up_conv)
        return out

 #    def get_feature_maps(self, layer_name, upscale):
  #       feature_extractor = HookBasedFeatureExtractor(self.net, layer_name, upscale)
   #      return feature_extractor.forward(Variable(self.input))
