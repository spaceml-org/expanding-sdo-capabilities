from collections import namedtuple, defaultdict
import math
import random
import os
import shutil
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from functools import reduce
import operator

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

class VT_BasicEncoder(nn.Module):
    """A Encoder Decoder module that is used to create one channel (e.g. AIA 211) from
       three other AIA channels (e.g. AIA 94, 193, and 171).
    """
    
    def __init__(self, input_shape=[3, 128, 128]):
        """
        Params: 
        input_shape: of the form of (n_channels x width x height)
        """
        super(VT_BasicEncoder,self).__init__()
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
