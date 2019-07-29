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
from torchvision.utils import save_image
from scipy import fftpack
import numpy as np
import scipy
from sklearn.metrics import mean_squared_error
from PIL import Image 
import sdo
import logging
import sys
from sdo.sdo_dataset import SDO_Dataset
from torch.utils.data import DataLoader
from sdo.metrics.azimuth_metric import azimuthal_average, compute_2Dpsd
from sdo.models.encoder_decoder import EncoderDecoder

output_path = '/gpfs/gpfs_gl4_16mb/b9p111/b9p111ar/results_VT/'
#just a way to get nice logging messages from the sdo package
logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S")
subsample = 4
original_ratio = 512
img_shape = int(original_ratio/subsample)
instr = ['AIA', 'AIA', 'AIA','AIA']
channels = ['0094','0171','0193','0211']
input_channels = len(channels)- 1


#some cuda initialization
torch.backends.cudnn.enabled = True
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available! Unable to continue")
device = torch.device("cuda")
print("Using device {} for training, current device: {}, total devices: {}".format(
device, torch.cuda.current_device(), torch.cuda.device_count()))


train_data = SDO_Dataset(device=device,instr=instr, channels=channels, yr_range=[2011, 2013], 
                         mnt_step=1, day_step=1, h_step=1, min_step=6, subsample=subsample, 
                         test_ratio= 0.3, normalization=0, scaling=True)
test_data = SDO_Dataset(device=device,instr=instr, channels=channels, yr_range=[2011, 2013], 
                        mnt_step=1, day_step=1, h_step=1, min_step=6, subsample=subsample, 
                        test_ratio= 0.3, normalization=0, scaling=True, test=True)
train_data_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=128, shuffle=True)


#defining some params
num_epochs = 2000 
model = EncoderDecoder(input_shape=[input_channels,img_shape,img_shape]).cuda(device)
distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5,lr=0.0001)


####Training the data##########
loss1   = torch.empty(num_epochs)
metrics1= torch.empty(num_epochs)

for epoch in range(num_epochs):
    loss_batch = []
    for batch_index, batch in enumerate(train_data_loader):
        data = batch.cuda(device,async=True)
        img = data[:,0:input_channels,:,:]
        truth = data[:,input_channels,:,:]
        truth = truth.unsqueeze(1)
        optimizer.zero_grad()
        output = model(img)
        output = output.float()
        loss = torch.sqrt(distance(output, truth))
        loss_batch.append(float(loss.cpu()))
        # ===================backward====================
        loss.backward()
        optimizer.step()
        print('loss: {}'.format(loss_batch[batch_index]))
        
    #The snippet below is used for the metric, i.e. Power Spectral Density (PSD)
    loss1[epoch] = np.array(loss_batch).mean()
    truth = truth.cpu()
    output = output.cpu()
    prediction = output.detach().numpy()
    ground_truth = truth.detach().numpy()
    # TODO Can also include the filtering components    
    psd_1Dpred = azimuthal_average(compute_2Dpsd(prediction[0,0,:,:]))
    psd_1Dtruth = azimuthal_average(compute_2Dpsd(ground_truth[0,0,:,:]))
    
    metrics1[epoch]=mean_squared_error(np.log(psd_1Dtruth),np.log(psd_1Dpred)) 
    # ===================log========================
    print('epoch [{}/{}]  loss: {}'.format(epoch+1, num_epochs, loss1))

torch.save(model.state_dict(), output_path+'virtual_telescope_model_001.pt')


###### Testing #######
tloss=0
outs=torch.empty([1,1,img_shape,img_shape])
tests=torch.empty([1,1,img_shape,img_shape])

for batch_index, batch in enumerate(test_data_loader):
    data = batch.cuda(cuda_device,async=True)
    img = data[:,0:input_channels,:,:]
    truth = data[:,input_channels,:,:]
    test = test.unsqueeze(1)
    # ===================forward=====================
    output = model(img)
    output = output.float()
    outs = torch.cat([outs,output.cpu()],dim=0)
    tests = torch.cat([tests,test.cpu()],dim=0)
    loss = torch.sqrt(distance(output, test))
    tloss=tloss+loss
    # ===================log========================
    #print('Image [{}/{}]  loss: {}'.format(i+1, len(x_test), loss))

aveloss=tloss/len(data)
print(aveloss)

ground_truth_test = torch.save(outs, output_path+'Test_ground_truth_model001.pt')
pred_test = torch.save(tests, output_path+'Test_model001.pt')

