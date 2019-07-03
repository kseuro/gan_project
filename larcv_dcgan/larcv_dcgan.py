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
        with the same size as the training images (1x512x512) = 262,144
    In practice, this is done with a series of strided 2D conv-transpose
        layers, paired with a 2D batch-norm layer and ReLU activation.
        The output is passed through a Tanh function to map it to the input
        data range of [-1, 1].
    Inputs:
        - nc:  number of color channels (bw = 1)
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
            # Print(), # [1, 100, 1, 1]
            nn.ConvTranspose2d(in_channels = nz, out_channels = ngf * 8,
                               kernel_size = 4, stride = 1, padding = 0,
                               output_padding = 0, groups = 1, bias = False,
                               dilation = 1),
            # Print(), # [1, 1024, 4, 4]
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels = ngf * 8, out_channels = ngf * 7,
                               kernel_size = 4, stride = 2, padding = 1,
                               output_padding = 0, groups = 1, bias = False,
                               dilation = 1),
            # Print(), # [1, 896, 8, 8]
            nn.BatchNorm2d(ngf * 7),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels = ngf * 7, out_channels = ngf * 6,
                               kernel_size = 4, stride = 2, padding = 1,
                               output_padding = 0, groups = 1, bias = False,
                               dilation = 1),
            # Print(), # [1, 768, 16, 16]
            nn.BatchNorm2d(ngf * 6),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels = ngf * 6, out_channels = ngf * 5,
                               kernel_size = 4, stride = 2, padding = 1,
                               output_padding = 0, groups = 1, bias = False,
                               dilation = 1),
            # Print(), # [1, 640, 32, 32]
            nn.BatchNorm2d(ngf * 5),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels = ngf * 5, out_channels = ngf * 4,
                               kernel_size = 4, stride = 2, padding = 1,
                               output_padding = 0, groups = 1, bias = False,
                               dilation = 1),
            # Print(), # [1, 512, 64, 64]
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels = ngf * 4, out_channels = ngf * 3,
                               kernel_size = 4, stride = 2, padding = 1,
                               output_padding = 0, groups = 1, bias = False,
                               dilation = 1),
            # Print(), # [1, 384, 128, 128]
            nn.BatchNorm2d(ngf * 3),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels = ngf * 3, out_channels = ngf * 2,
                               kernel_size = 4, stride = 2, padding = 1,
                               output_padding = 0, groups = 1, bias = False,
                               dilation = 1),
            # Print(), # [1, 256, 128, 128]
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels = ngf * 2, out_channels = nc,
                               kernel_size = 4, stride = 2, padding = 1,
                               output_padding = 0, groups = 1, bias = False,
                               dilation = 1),
            # Print(), # [1, 1, 512, 512]
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
        - nc:  number of color channels (bw = 1)
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
            nn.Conv2d(in_channels = nc, out_channels = ndf, kernel_size = 4,
                      stride = 2, padding = 1, dilation = 1, groups = 1,
                      bias = False),
            # Print(), # torch.Size([10, 128, 256, 256])
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(in_channels = ndf, out_channels = ndf * 2, kernel_size = 4,
                      stride = 2, padding = 1, dilation = 1, groups = 1,
                      bias = False),
            # Print(), # torch.Size([10, 256, 128, 128])
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(in_channels = ndf * 2, out_channels = ndf * 3, kernel_size = 4,
                      stride = 2, padding = 1, dilation = 1, groups = 1,
                      bias = False),
            # Print(), # torch.Size([10, 384, 64, 64])
            nn.BatchNorm2d(ndf * 3),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(in_channels = ndf * 3, out_channels = ndf * 4, kernel_size = 4,
                      stride = 2, padding = 1, dilation = 1, groups = 1,
                      bias = False),
            # Print(), # torch.Size([10, 512, 32, 32])
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(in_channels = ndf * 4, out_channels = ndf * 5, kernel_size = 4,
                      stride = 2, padding = 1, dilation = 1, groups = 1,
                      bias = False),
            # Print(), # torch.Size([10, 640, 16, 16])
            nn.BatchNorm2d(ndf * 5),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(in_channels = ndf * 5, out_channels = ndf * 6, kernel_size = 4,
                      stride = 2, padding = 1, dilation = 1, groups = 1,
                      bias = False),
            # Print(), # torch.Size([10, 768, 8, 8])
            nn.BatchNorm2d(ndf * 6),
            nn.Conv2d(in_channels = ndf * 6, out_channels = 1, kernel_size = 8,
                      stride = 1, padding = 0, dilation = 1, groups = 1,
                      bias = False),
            # Print(), # torch.Size([10, 1, 1, 1])
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
