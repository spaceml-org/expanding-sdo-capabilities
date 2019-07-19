import os
import logging
import sys
import torch
from torchvision import datasets, transforms
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import torchvision.utils as vutils
from sdo.models.gan import Generator, Discriminator, Encoder
from sdo.sdo_dataset import SDO_Dataset
from torch.utils.data import DataLoader

from matplotlib.pyplot import imshow

save_image_dir = '/gpfs/gpfs_gl4_16mb/b9p111/b9p111ap/results_BiGAN/plots'
save_model_dir = '/gpfs/gpfs_gl4_16mb/b9p111/b9p111ap/results_BiGAN/models'

# these are used in the original repo, do they make sense for us?
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)

def tocuda(x):
    return x.cuda()

def get_log_odds(raw_marginals):
    marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(marginals / (1 - marginals))

#just a way to get nice logging messages from the sdo package
logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S")

# dataset parameters
subsample = 1
original_ratio = 512
img_shape = int(original_ratio/subsample)
instr = ['AIA', 'AIA', 'AIA']
channels = ['0171', '0193', '0094']

# NN parameters
batch_size = 100
latent_size = 256
num_epochs = 100
lr = 1e-4

#some cuda initialization
torch.backends.cudnn.enabled = True
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available! Unable to continue")
device = torch.device("cuda")
print("Using device {} for training, current device: {}, total devices: {}".format(
device, torch.cuda.current_device(), torch.cuda.device_count()))

# loading data
train_data = SDO_Dataset(device=device, instr=instr, channels=channels, yr_range=[2015, 2018], 
                         subsample=subsample, normalization=1, bytescaling=True)
train_loader = DataLoader(train_data, batch_size=10, shuffle=False)

# defining the model
netE = tocuda(Encoder(latent_size, noise=True))
netG = tocuda(Generator(latent_size))
netD = tocuda(Discriminator(latent_size, dropout=0.2, output_size=1))

netE.apply(weights_init)
netG.apply(weights_init)
netD.apply(weights_init)

optimizerG = optim.Adam([{'params' : netE.parameters()},
                         {'params' : netG.parameters()}], lr=lr, betas=(0.5,0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()

# training loop
for epoch in range(num_epochs):
    i = 0
    for batch_index, data in enumerate(train_loader):
        print("Processing batch n %d" % batch_index)
        real_label = Variable(tocuda(torch.ones(batch_size)))
        fake_label = Variable(tocuda(torch.zeros(batch_size)))

        noise1 = Variable(tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))
        noise2 = Variable(tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))

        if epoch == 0 and i == 0:
            netG.output_bias.data = get_log_odds(tocuda(data))

        if data.size()[0] != batch_size:
            continue

        d_real = Variable(tocuda(data))

        z_fake = Variable(tocuda(torch.randn(batch_size, latent_size, 1, 1)))
        d_fake = netG(z_fake)

        z_real, _, _, _ = netE(d_real)
        z_real = z_real.view(batch_size, -1)

        mu, log_sigma = z_real[:, :latent_size], z_real[:, latent_size:]
        sigma = torch.exp(log_sigma)
        epsilon = Variable(tocuda(torch.randn(batch_size, latent_size)))

        output_z = mu + epsilon * sigma

        output_real, _ = netD(d_real + noise1, output_z.view(batch_size, latent_size, 1, 1))
        output_fake, _ = netD(d_fake + noise2, z_fake)

        loss_d = criterion(output_real, real_label) + criterion(output_fake, fake_label)
        loss_g = criterion(output_fake, real_label) + criterion(output_real, fake_label)

        if loss_g.data[0] < 3.5:
            optimizerD.zero_grad()
            loss_d.backward(retain_graph=True)
            optimizerD.step()

        optimizerG.zero_grad()
        loss_g.backward()
        optimizerG.step()

        if i % 1 == 0:
            print("Epoch :", epoch, "Iter :", i, "D Loss :", loss_d.data[0], "G loss :", loss_g.data[0],
                  "D(x) :", output_real.mean().data[0], "D(G(x)) :", output_fake.mean().data[0])

        if i % 50 == 0:
            vutils.save_image(d_fake.cpu().data[:16, ], '%s/fake.png' % (save_image_dir))
            vutils.save_image(d_real.cpu().data[:16, ], '%s/real.png'% (save_image_dir))

        i += 1

    if epoch % 10 == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (save_model_dir, epoch))
        torch.save(netE.state_dict(), '%s/netE_epoch_%d.pth' % (save_model_dir, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (save_model_dir, epoch))

        vutils.save_image(d_fake.cpu().data[:16, ], '%s/fake_%d.png' % (save_image_dir, epoch))
        