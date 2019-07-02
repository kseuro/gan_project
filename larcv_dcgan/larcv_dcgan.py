# DCGAN implementation using LArCV1 Dataset
# Kai Kharpertian
# Tufts University
# Department of Physics

##############################
# Imports
##############################
import torch
import torch.nn               as nn
import torch.nn.parallel
import torch.backends.cudnn   as cudnn
import torch.optim            as optim
import torch.utils.data
import torchvision.datasets   as dset
import torchvision.transforms as transforms
import torchvision.utils      as vutils
from ganfuncs import Print
##############################
# Networks
##############################
class GNet(torch.nn.Module):
    """
    Designed to map a latent space vector (z) to data-space. Since the data
        are images, the conversion of z to data-space means creating an image
        with the same size as the training images (1x28x28).
    In practice, this is done with a series of strided 2D conv-transpose
        layers, paired with a 2D batch-norm layer and ReLU activation.
        The output is passed through a Tanh function to map it to the input
        data range of [-1, 1].
    Inputs:
        - nc:  number of color channels    (rgb = 3) (bw = 1)
        - nz:  length of the latent vector (100)
        - ngf: depth of feature maps carried through generator
        - Transposed convolution is also known as fractionally-strided conv.
            - One-to-many operation
    ConvTranspose2d output volume:
        Input:  [N, C, Hin,  Win]
        Output: [N, C, Hout, Wout] where:
            Hout = (Hin - 1) * stride - 2 * pad + K + out_pad (default = 0)
            Wout = (Win - 1) * stride - 2 * pad + K + out_pad (default = 0)
            K = 4, S = 2, P = 1: doubles img. dim each layer
    """
    def __init__(self, nc, nz, ngf):
        super(GNet, self).__init__()
        self.main = nn.Sequential(
            # Print(),
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias = False),
            # Print(),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias = False),
            # Print(),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias = False),
            # Print(),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias = False),
            # Print(),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias = False),
            # Print(),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class DNet(torch.nn.Module):
    """
    Designed to acts as a binary classifier that takes an image
        as input and outputs a scalar probability that the input image is
        real (as opposed to generated).
    In this implementation, D takes a 1x64x64 input image and processes it
        through a series of Conv2d, BatchNorm2d, and LeakyReLU layers. The
        final probability is output through a sigmoid activation function.
    Per the DCGAN paper, it is good practice to use strided convolution
        rather than pooling to downsample because it lets the network learn
        its own pooling function. BatchNorm and LeakyReLU functions promote
        good gradient flow - critical to the learning of both G and D.
    Inputs:
        - nc:  number of color channels (rgb = 3) (bw = 1)
        - ndf: sets depth of feature maps propagated through discriminator
    Convolutional output volume:
        O = [i + 2*p - K - (K-1)*(d-1)] / S + 1
        O = Output dim
        i = Input dim
        d = Dilation rate
        K = Kernel size
        P = Padding
        S = Stride
    """
    def __init__(self, nc, ndf):
        super(DNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias = False),
            Print(),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias = False),
            Print(),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias = False),
            Print(),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias = False),
            Print(),
            nn.BatchNorm2d(ndf * 8),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
