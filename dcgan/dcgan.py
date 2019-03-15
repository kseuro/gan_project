# Simple DCGAN implementation using MNIST dataset
# Kai Kharpertian
# Tufts University
# Department of Physics

###############
# Dependencies
###############
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

###############
# Inputs
###############
# Hardware & Data
###############
# dataroot: path to root dataset folder
dataroot = '/media/hdd1/kai/datasets/mnist'
# workers: number of worker threads for loadings data with Dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
# size using a transformer.
image_size = 28

# nc: number of color channels [rgb = 3] [bw = 1]
nc = 1
# ngpu: number of gpu's available. Use 0 for CPU mode [much slower]
ngpu = 0

###############
# Model Params
###############
# nz: length of the latent vector
nz = 100
# ngf: depth of feature maps carried through the generator [MNIST = 28]
ngf = 28
# ndf: sets depth of feature maps propagated through discriminator
ndf = 28
# Number of training epochs
num_epochs = 5
# lr: learning rate, should be 0.0002 per DCGAN paper
lr = 0.0002
# beta1: hyperparameter for Adam optimizer. Should be 0.5 per DCGAN paper
beta1 = 0.5

###############
# Data
###############
def mnist_data(root):
    compose = transforms.Compose( [transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize( (.5, .5, .5),
                                                         (.5, .5, .5)),
                                  ])
    return dset.MNIST(root = root, train = True, transform = compose,
                      download = False)

dataset = mnist_data(dataroot)
dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size = batch_size,
                                        shuffle = True,
                                        num_workers = workers,
                                        drop_last = True)
device = torch.device("cuda:2" if (torch.cuda.is_available() and ngpu > 0)
                        else "cpu")

###############
# Custom weights
###############
def weights_init(m):
    """
    Custon weight init based on DCGAN paper recommendations of
    mean = 0.0, stdev = 0.2.
    Input:   initialized model
    Returns: reinitialized conv, conv-transpose, and batch-norm
             layers that meet above criteria.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

###############
# Networks
###############
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
    def __init__(self, ngpu):
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
    Convolutional output volume: [W - K + 2P] / S + 1
        W = Input dim
        K = Kernel size
        P = Padding
        S = Stride
    """
    def __init__(self, ngpu):
        super(DNet, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 2, 2, 0, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            # State size = (ndf) x 14 x 14 = 5488
            # outvol: (28 - 2 + 2*0) / 2 + 1 = 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),
            # State size = (ndf * 2) x 7 x 7 = 2744
            # outvol: (28 - 2 + 2*0) / 2 + 1 = 14
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            # State size = (ndf * 4) x 4 x 4 = 1792
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

###############
# Instantiation
###############
# Networks
netG = GNet(ngpu).to(device)
netD = DNet(ngpu).to(device)

# MultiGPU option
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Custom weight init
netG.apply(weights_init)
netD.apply(weights_init)

###############
# Loss Fn and Optim
###############
# BCE loss function
criterion = nn.BCELoss()

# Batch of latent vectors — used to viz. G's progress
fixed_noise = torch.randn(64, nz, 1, 1, device = device)

# Est. convention for real and fake labels
real_label = 1
fake_label = 0

# Optimizers
optimG = optim.Adam(netG.parameters(), lr = lr, betas = (beta1, 0.999))
optimD = optim.Adam(netD.parameters(), lr = lr, betas = (beta1, 0.999))

###############
# Training
###############
"""
    - Discriminator training
    - Goal of D is to maximize prob of correctly classifying a given input as
      either real or fake.
    - Want to maximize log(D(x)) + log(1 - D(G(z)))
    - Training implemented in two steps:
        - 1. Construct batch of real samples from the training set
            - Forward pass through D, calc loss log(D(x))
            - Calc gradients with backward pass
        - 2. Construct batch of fake samples with the current G
            - Forward pass fake batch through D, calc loss log(1-D(G(z)))
            - Accumulate gradients with a backward pass
        - With gradients from both real and fake batches, call a step
            on D's optim.

    - Generator training
    - Want to train G by maximizing log(D(G(z)))
        - Minimizing log(1 - D(G(z))), as in original paper, does not give
            good gradients
    - Accomplish maximization by:
        - Classifying G's output using D
        - Computing G's loss using real labels as Ground Truth (GT)
            - Using real labels as GT for loss allows for use of log(x)
                part of BCE, rather than log(1 - x)
        - Computing G's gradients in backward pass
        - Update G's params by optim update step

    - Static reporting
    - Updated report at the end of each epoch to track progress
    - Will push fixed noise batch through G
    - Loss_D: D loss calc as sum of losses for all real and fake batches
        - log(D(x)) + log(D(G(z)))
    - Loss_G: G loss calc as log(D(G(z)))
    - D(x): Average output (across single batch) of D for real batch
        - Should start close to 1 and begin to converge to 0.5
    - D(G(z)): Average D output for all fake batch.
        - First number: before update
        - Second number: after update
"""
# Training Loop
# Lists to track progress
img_list = []
G_losses = []
D_losses = []
iters    = 0

# print("Parameters:             ")
# print("Workers:                ",workers)
# print("Channels:               ",nc)
# print("Number of GPU's:        ",ngpu)
# print("Latent Vector Length:   ",nz)
# print("G Feature Depth:        ",ngf)
# print("D Feature Depth:        ",ndf)
# print("Learning Rate:          ",lr)
# print("Beta:                   ",beta1)
# print("Press return to start training loop...")
# input()

# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ###############
        # Update D: maximize log(D(x)) + log(1 - D(G(z)))
        ###############
        # Train on real batch
        netD.zero_grad()

        print('     ')
        print('     ')
        print('     ')
        # Format batch
        real_cpu = data[0].to(device) # type: Torch Tensor: [1, 1, 28, 28]
        print('real_cpu.size():', real_cpu.size())

        b_size   = real_cpu.size(0)   # type: int = 1
        print('b_size:', b_size)

        label    = torch.full((b_size,), real_label, device = device)
        # Forward pass real batch through D
        output   = netD(real_cpu).view(0)

        # Calculate loss on real batch
        print('output.size():', output.size())
        print('label.size():', label.size())
        print('     ')
        print('     ')
        print('     ')
        errD_real = criterion(output, label) # ('Input', 'Target')

        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Train on a fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device = device)
        # Generate fake img batch using G
        fake_data = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batches with D
        output = netD(fake_data.detach()).view(-1)
        # Calc D's loss on fake batche
        errD_fake = criterion(output, label)
        # Calc gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add accumulated gradients from real and fake batches
        errD = errD_real + errD_fake
        # Update D
        optimD.step()

        ###############
        # Update G: maximize log(D(G(z)))
        ###############
        netG.zero_grad()
        label.fill_(real_batch) # fake labels -> real labels for G loss
        # Perform second forward pass of fake batch through updated D
        output = netD(fake).view(-1)
        # Calc G's loss based on this output
        errG = criterion(output, label)
        # Calc G's gradient
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizer.step()

        ###############
        # Output training stats
        ###############
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                       errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save losses for plotting
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check G's progress
            if (iters % 500 == 0) or ( (epoch == num_epochs-1) and
                                     ( i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake,
                                                 padding = 2,
                                                 normalize = True))
            iters += 1
