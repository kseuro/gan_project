import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

"""
Inputs:
- datarun: path to root dataset folder
- workers: number of worker threads for loadings data with Dataloader
- nc: number of color channels (rgb = 3) (bw = 1)
- nz: length of the latent vector
- ngf: depth of feature maps carried through the generator
- ndf: sets depth of feature maps propagated through discriminator
- lr: learning rate, should be 0.0002
- beta1: hyperparameter for Adam optimizer. Should be 0.5
"""


class DNet(torch.nn.Module):
    """
    Designed to be a binary classification network that takes an image
        as input and outputs a scalar probability that the input image is
        real (as opposed to generated).
    In this implementation, D takes a 1x28x28 input image and processes it
        through a series of Conv2d, BatchNorm2d, and LeakyReLU layers. The
        final probability is output through a sigmoid activation function.
    Per the DCGAN paper, it is good practice to use strided convolution
        rather than pooling to downsample because it lets the network learn
        its own pooling function. BatchNorm and LeakyReLU function promote
        good gradient flow - critical to the learning of both G and D.
    Inputs:
        - nc: number of color channels (rgb = 3) (bw = 1)
        - ndf: sets depth of feature maps propagated through discriminator
    """
    def __init__(self, nc, ndf, ngpu):
        super(DNet, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            # State size = (ndf) x 14 x 14 = 5488
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),
            # State size = (ndf * 2) x 7 x 7 = 2744
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            # State size = (ndf * 4) x 4 x 4 = 1792
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class GNet(torch.nn.Module):
    """
    Designed to map a latent space vector (z) to data-space. Since the data
        are images, the conversion of z to data-space means creating an image
        with the same size as the training images (1x28x28).
    In practice, this is done with a series of strided 2D conv-transpose
        layers, paired with a 2D batch-norm layer and ReLU activation.
        The output is passed through a Tanh function to it to the input
        data range of [-1, 1].
    Inputs:
        - nc: number of color channels (rgb = 3) (bw = 1)
        - nz: length of the latent vector (100)
        - ngf: depth of feature maps carried through generator (28 for MNSIT)
        - Transposed convolution is also known as fractionally-strided conv.
            - One-to-many operation
    """
    def __init__(self, nc, nz, ngf, ngpu):
        super(GNet, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias = False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 4 x 4 = 3584
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 8 x 8 = 7168
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 16 x 16 = 28627
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 32 x 32 = 57334
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias = False),
            nn.Tanh()
            # State size: (nc) * 28 * 28 = 784
        )

    def forward(self, input):
        return self.main(input)
