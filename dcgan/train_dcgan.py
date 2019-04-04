# Training routine for DCGAN model
# Kai Kharpertian
# Tufts University | Department of Physics
# Tufts Neutrino Group

##############################
# Import scripts
##############################
import os, time
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
import matplotlib.pyplot      as plt
import itertools
import ganfuncs
import imageio
import random
from   datetime import datetime
from   dcgan    import GNet, DNet
from tensorboardX import SummaryWriter

##############################
# Inputs
##############################
workers    = 2      # dataloader worker threads
batch_size = 25     # Batch size during training
image_size = 64     # Spatial size of training images.
nc         = 1      # number of color channels [rgb = 3] [bw = 1]
ngpu       = 1      # 0: CPU mode; > 1: MultiGPU mode
nz         = 100    # length of the latent vector
ngf        = 128    # depth of generator feature maps
ndf        = 128    # depth of discriminator feature maps
num_epochs = 20     # number of epochs
lr         = 0.0002 # Should be 0.0002 per DCGAN paper
beta1      = 0.5    # Should be 0.5 per DCGAN paper
z_dim      = 5      # Defines batch size of fixed or random noise vectors
num_iters  = 0      # Start at zero

##############################
# Data and logging
##############################
dataroot    = '/media/hdd1/kai/datasets/mnist'
dataloader  = ganfuncs.mnist_data(image_size, batch_size, workers, dataroot)
num_batches = len(dataloader)

# TensorboardX
writer = SummaryWriter('/media/hdd1/kai/tensorBoard/runs/gan_project')

# Select GPU or CPU
device = torch.device("cuda:1" if (torch.cuda.is_available() and ngpu > 0)
                      else "cpu")

# Fixed noise vector for testing G's performance
fixed_noise = torch.randn( (z_dim * z_dim, nz) ).view(-1, nz, 1, 1).to(device)

# TODO: implement periodic saving of checkpoints and best checkpoint

##############################
# Instantiation
##############################
# Networks
netG = GNet(nc, nz, ngf)
netD = DNet(nc, ndf)

# MultiGPU option
# if (device.type == 'cuda') and (ngpu > 1):
#     netG = nn.DataParallel(netG, list(range(ngpu)))
#     netD = nn.DataParallel(netD, list(range(ngpu)))

# Custom weight init
netG.apply(ganfuncs.weights_init) # weights on cpu
netD.apply(ganfuncs.weights_init) # weights on cpu
netG.to(device)
netD.to(device)

##############################
# Loss Fn and Optim
##############################
# Binary Cross Entropy Loss
BCE_loss = nn.BCELoss()

# Adam optimizer
optimG = optim.Adam(netG.parameters(), lr = lr, betas = (beta1, 0.999))
optimD = optim.Adam(netD.parameters(), lr = lr, betas = (beta1, 0.999))

# Dictionary of accuracies
train_hist = ganfuncs.train_hist()

# Times and save directory
start_time, out_dir, now, date_time = ganfuncs.train_start(time.time())

##############################
# Training Loop
##############################
for epoch in range(num_epochs):

    D_losses = []
    G_losses = []

    epoch_start_time = time.time()

    for x_, _ in dataloader:
        '''
        - Discriminator training
            - Goal of D is to maximize prob of correctly classifying a given
              input as either real or fake.
            - Want to maximize log(D(x)) + log(1 - D(G(z)))
        '''
        netD.zero_grad()

        mini_batch = x_.size()[0]

        # Establish soft data labels
        # Flip target on BCE loss: real: 1->0, fake: 0->1
        real_label = random.uniform(0, 0.1)
        fake_label = random.uniform(0.9, 1.0)

        # real target
        y_real_ = torch.full( (mini_batch,), real_label, device = device)
        y_real_ = y_real_.view(-1, 1, 1, 1)

        # fake target
        y_fake_ = torch.full( (mini_batch,), fake_label, device = device)
        y_fake_ = y_fake_.view(-1, 1, 1, 1)

        # Train D on a real batch
        x_ = x_.to(device)
        D_result = netD(x_)                       # Forward pass through D
        D_real_loss = BCE_loss(D_result, y_real_) # Calc loss log(D(x))

        # Generate fake batch
        z_ = ganfuncs.random_noise(z_dim, nz)     # ([25, 100, 1, 1])
        G_result = netG(z_)                       # ([25, 1, 64, 64])

        # Train D on fake batch
        D_result = netD(G_result)                 # Forward pass through D
        D_fake_loss = BCE_loss(D_result, y_fake_) # Calc loss log(D(G(z))
        D_fake_score = D_result.data.mean()

        # Accumulate losses and backprop
        D_train_loss = D_real_loss + D_fake_loss
        D_train_loss.backward()
        optimD.step()

        # Append D losses to list
        D_losses.append(D_train_loss.data)

        '''
        - Generator training
        - Want to train G by maximizing log(D(G(z)))
            - Minimizing log(1 - D(G(z))), as in Goodfellow paper, does not
                give good gradients
        - Accomplish maximization by:
            - Classifying G's output using D
            - Computing G's loss using real labels as Ground Truth (GT)
                - Using real labels as GT for loss allows for use of log(x)
                    part of BCE, rather than log(1 - x)
            - Computing G's gradients in backward pass
            - Update G's params by optim update step
        '''
        netG.zero_grad()

        # Generate random noise
        z_ = ganfuncs.random_noise(z_dim, nz) # [25, 100, 1, 1]

        # Push noise through G
        G_result = netG(z_)                   # [25, 1, 64, 64]

        # Push G's output through D
        D_result = netD(G_result)             # [25, 1, 1, 1]

        # Calculate loss and backprop
        G_train_loss = BCE_loss(D_result, y_real_)
        G_train_loss.backward()
        optimG.step()

        # Append G losses to list
        G_losses.append(G_train_loss.data)

        # Calculate average loss for D and G
        loss_d = torch.mean(torch.FloatTensor(D_losses))
        loss_g = torch.mean(torch.FloatTensor(G_losses))

        # Write losses to training history
        train_hist['D_losses'].append(loss_d)
        train_hist['G_losses'].append(loss_g)

        # Output status to terminal
        if num_iters % 100 == 0:
            print('[epoch.{} / num_epochs.{}]'.format(epoch, num_epochs))
            print('loss_d:{:.3f}, loss_g:{:.3f}'.format(loss_d, loss_g))

        # Output losses to TbX
        writer.add_scalar('D_Loss', loss_d.item(), epoch)
        writer.add_scalar('G_Loss', loss_g.item(), epoch)

        num_iters += 1

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time

    loss_d = torch.mean(torch.FloatTensor(D_losses))
    loss_g = torch.mean(torch.FloatTensor(G_losses))

    # Write end-of-epoch losses to training history
    train_hist['D_losses'].append(loss_d)
    train_hist['G_losses'].append(loss_g)
    train_hist['per_epoch_time'].append(epoch_time)

    # Output epoch summary to terminal
    print_string = '[{}/{}] - epoch_time:{}, loss_d:{:.3f}, loss_g:{:.3f}'
    print(print_string.format((epoch + 1), num_epochs, epoch_time, loss_d, loss_g))

    # Output end-of-epoch losses to TbX
    writer.add_scalar('D_Loss', loss_d.item(), epoch)
    writer.add_scalar('G_Loss', loss_g.item(), epoch)

    # Save current-state GAN outputs
    ganfuncs.save_outputs(netG, out_dir, epoch, num_epochs, datetime,
                          fixed_noise)

##############################
# Training Complete
# Save: model outputs
#       weights/params
#       loss plots
##############################
print('#################')
print('End of training')
print('#################')

writer.close()
end_time = time.time()
total_time = end_time - start_time
train_hist['total_time'].append(total_time)
print('Total time:', total_time)
print("Avg time per epoch: %.2f, total %d epochs time: %.2f" %
     (torch.mean(torch.FloatTensor(train_hist['per_epoch_time'])),
     num_epochs, total_time))

# Save model parameters
ext = '.tar'
ganfuncs.save_model(netG, out_dir, ext, G = True)  # G params
ganfuncs.save_model(netD, out_dir, ext, G = False) # D params

# Plot losses
ganfuncs.plot_losses(out_dir, datetime,
                     train_hist['G_losses'],
                     train_hist['D_losses'])

# Save evaluation images as animation
ganfuncs.save_ani(out_dir, num_epochs, fixed = True, random = True)
