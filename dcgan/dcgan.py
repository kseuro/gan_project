# DCGAN implementation using MNIST dataset
# Kai Kharpertian
# Tufts University
# Department of Physics

##############################
# Dependencies
##############################
import os
import errno
import torch
import torch.nn               as nn
import torch.nn.parallel
import torch.backends.cudnn   as cudnn
import torch.optim            as optim
import torch.utils.data
import torchvision.datasets   as dset
import torchvision.transforms as transforms
import torchvision.utils      as vutils
import numpy                  as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from   utils    import Logger
from   datetime import datetime

##############################
# Inputs
##############################
# workers: number of worker threads for loadings data with Dataloader
workers = 2
# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this
# size using a transformer.
image_size = 28
# nc: number of color channels [rgb = 3] [bw = 1]
nc = 1
# ngpu: number of gpu's available. Use 0 for CPU mode.
ngpu = 1
# nz: length of the latent vector
nz = 100
# ngf: depth of feature maps carried through the generator [MNIST = 28]
ngf = 28
# ndf: sets depth of feature maps propagated through discriminator
ndf = 28
# Number of training epochs
num_epochs = 100
# lr: learning rate, should be 0.0002 per DCGAN paper
lr = 0.0002
# beta1: hyperparameter for Adam optimizer. Should be 0.5 per DCGAN paper
beta1 = 0.5

##############################
# Data
##############################
def mnist_data(root):
    compose = transforms.Compose( [transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize( (.5, .5, .5),
                                                         (.5, .5, .5)),
                                  ])
    return dset.MNIST(root = root, train = True, transform = compose,
                      download = False)

now       = datetime.now()
date_time = now.strftime("%m-%d-%Y, %H:%M:%S")
dataroot  = '/media/hdd1/kai/datasets/mnist'
out_dir   = '/media/hdd1/kai/projects/gan_project/dcgan/data/images' + date_time

dataset  = mnist_data(dataroot)
dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size = batch_size,
                                        shuffle = True,
                                        num_workers = workers,
                                        drop_last = False)
device = torch.device("cuda:1" if (torch.cuda.is_available() and ngpu > 0)
                        else "cpu")

# TensorboardX
logger = Logger(model_name = '_dcgan_test_'+ date_time , data_name = 'MNIST')

##############################
# Helper classes and functions
##############################
class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

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

def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def save_results(dataloader, out_dir, date_time, img_list, img_names,
                 G_losses, D_losses):
    """
        Save and visualize results after the completion of training
    """

    make_dir(out_dir)

    # Save G's sample images                        # [64, 1, 28, 28]
    for index in range(len(img_list)):              # 2
        for img in range(img_list[index].shape[0]): # 64
            # (filename, array)
            plt.imsave(img_names[index] + '_' + str(img) + '.png',
                       img_list[index][img].reshape(28, 28), cmap = 'gray')

    # Plot G and D losses
    plt.figure()
    plt.title("Generator and Discriminator Loss Curves")
    plt.plot(G_losses, label = 'G')
    plt.plot(D_losses, label = 'D')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('{}/{}.png'.format(out_dir, date_time), dpi = 300)

    # # Side-by-Side of real vs fake images
    # # Get batch of real images
    # real_batch = next(iter(dataloader))
    #
    # # Plot real images
    # name1 = 'real_batch'
    # plt.figure()
    # plt.plot(np.transpose(vutils.make_grid(real_batch[0].to(device)[:28],
    #                                           padding=5, normalize=True).cpu(),
    #                                           (1,2,0)))
    # plt.title("Real Images")
    # plt.savefig(name1)
    #
    # # Plot G's images from last epoch
    # name2 = 'G_last_epoch'
    # plt.figure()
    # plt.plot(np.transpose(img_list[-1],(1,2,0)))
    # plt.title("Fake Images")
    # plt.savefig(name2)

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
        The output is passed through a Tanh function to it to the input
        data range of [-1, 1].
    Inputs:
        - nc: number of color channels (rgb = 3) (bw = 1)
        - nz: length of the latent vector (100)
        - ngf: depth of feature maps carried through generator (28 for MNSIT)
        - Transposed convolution is also known as fractionally-strided conv.
            - One-to-many operation
    ConvTranspose2d output volume:
        Input:  [N, C, Hin,  Win]
        Output: [N, C, Hout, Wout] where:
            Hout = (Hin - 1) * stride - 2 * pad + K + out_pad (default = 0)
            Wout = (Win - 1) * stride - 2 * pad + K + out_pad (default = 0)
            K = 4, S = 2, P = 1: doubles img. dim each layer
    """
    def __init__(self, ngpu):
        super(GNet, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Print(), # [128, 100, 1, 1]
            nn.ConvTranspose2d(nz, ngf * 8, 2, 1, 0, bias = False),
            # Print(), # [128, 224, 2, 2]
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # outvol: 31
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias = False),
            # Print(), # [128, 112, 4, 4]
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # outvol: 62
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias = False),
            # Print(), # [128, 56, 8, 8]
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # outvol: 124
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias = False),
            # Print(), # [128, 28, 16, 16]
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # outvol: 248
            nn.ConvTranspose2d(ngf, nc, 4, 2, 3, bias = False),
            # Print(), # [128, 1, 28, 28]
            # outvol: 496
            nn.Tanh()
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
    Convolutional output volume:
        O = [i + 2*p - K - (K-1)*(d-1)] / S + 1
        O = Output dim
        i = Input dim
        d = Dilation rate
        K = Kernel size
        P = Padding
        S = Stride
    """
    def __init__(self, ngpu):
        super(DNet, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 2, 2, 0, bias = False),
            # Print(), # [128, 28, 14, 14]
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(ndf, ndf * 2, 2, 2, 1, bias = False),
            # Print(), # [128, 56, 8, 8]
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(ndf * 2, ndf * 4, 2, 2, 2, bias = False),
            # Print(), # [128, 112, 6, 6]
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(ndf * 4, 1, 6, 1, 0, bias = False),
            # Print(), # [128, 1, 1, 1]
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

##############################
# Instantiation
##############################
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

##############################
# Loss Fn and Optim
##############################
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

##############################
# Training
##############################
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
img_list    = []
img_names   = []
G_losses    = []
D_losses    = []
iters       = 0
num_batches = len(dataloader)

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
    for n_batch, data in enumerate(dataloader, 0):
        ##############################
        # Update D: maximize log(D(x)) + log(1 - D(G(z)))
        ##############################
        ## Train on real batch
        netD.zero_grad()

        # Format batch
        real_cpu = data[0].to(device) # type: Torch Tensor: [128, 1, 28, 28]

        # All real data -> create tensor full of 1's
        b_size = real_cpu.size(0)        # type: int = 128
        label  = torch.full((b_size,), real_label, device = device)
        label  = label.view(-1, 1, 1, 1) # reshape to match output dim

        # Forward pass real batch through D
        output = netD(real_cpu)

        # Calculate loss on real batch
        errD_real = criterion(output, label) # ('Input', 'Target')

        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()           # float.32

        ## Train on a fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device = device)

        # Generate fake img batch using G
        fake_data = netG(noise)
        label.fill_(fake_label)

        # Classify all fake batches with D
        output = netD(fake_data.detach()).view(-1, 1, 1, 1)
        # print(output.size()) # [3200, 1, 1, 1]

        # Calc D's loss on fake batch
        errD_fake = criterion(output, label)

        # Calc gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Add accumulated gradients from real and fake batches
        errD = errD_real + errD_fake

        # Update D
        optimD.step()

        ##############################
        # Update G: maximize log(D(G(z)))
        ##############################
        netG.zero_grad()
        label.fill_(real_label) # fake labels -> real labels for G loss

        # Perform second forward pass of fake batch through updated D
        output = netD(fake_data).view(-1, 1, 1, 1)

        # Calc G's loss based on this output
        errG = criterion(output, label)

        # Calc G's gradient
        errG.backward()
        D_G_z2 = output.mean().item()

        # Update G
        optimG.step()

        # Log batch error to TBx
        logger.log(errD_fake, errD_real, errG, epoch, n_batch, num_batches)

        ##############################
        # Output training stats
        ##############################
        if n_batch % 50 == 0:
            # disp status logs
            # logger.display_status(epoch, num_epochs, n_batch, num_batches,
            #                       d_error, g_error, d_pred_real, d_pred_fake)
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, n_batch, len(dataloader),
                       errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save losses for plotting
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        # Check G's progress
        if (iters % num_batches == 0) or ( (epoch  == num_epochs - 1 )  and
                                         ( n_batch == len(dataloader) - 1 ) ):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu() # [64, 1, 28, 28]
            img_list.append(fake)
            img_names.append('{}/{}_epoch_{}_batch'.format(out_dir,
                                                               epoch,
                                                               n_batch))
        iters += 1

save_results(dataloader, out_dir, date_time, img_list, img_names,
             G_losses, D_losses)
