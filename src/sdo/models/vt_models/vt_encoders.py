"""
Torch architecture models for the virtual telescope.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import operator

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
