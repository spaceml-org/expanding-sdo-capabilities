import logging

import torch
import torch.nn as nn
from sdo.models.vt_models.vt_unet_basic_blocks import (maxpool, conv_block_2, 
                                                       conv_block_2_sym, conv_trans_block)

_logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

"""https://jinglescode.github.io/2019/12/08/biomedical-image-segmentation-u-net-attention/"""

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x
    
class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

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

class UNet_Attention(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, num_filter=64):
        super(UNet_Attention, self).__init__()

        n1 = 64
        filters = [num_filter, num_filter * 2, num_filter * 4, 
                   num_filter * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)

        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out
    
    
class VT_UnetGeneratorAttention(nn.Module):
    """
    Unet with parametrizable depth of the network and attention gate.
    Base architecture copied from
    https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/networks/unet_grid_attention_3D.py
    See https://arxiv.org/pdf/1804.03999.pdf for more details
    """

    def __init__(self, input_shape=[3, 128, 128],
                 num_filter=64, LR_neg_slope=0.2,
                 depth=4):
        super(VT_UnetGeneratorAttention, self).__init__()
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


