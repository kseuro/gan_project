# Implementation of a convolutional GAN trained on MNSIT dataset
# Kai Kharpertian
# Feb. 2019

##############################
# Dependencies
##############################
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

# Visualization
from utils import Logger

# Functionality
from conv_nets import GNet, DNet
import conv_gan_funcs as funcs
##############################
# Hardware & Inputs
##############################
datarun = '/media/hdd1/kai/datasets/mnist'
ngpu    = 1
workers = 2
device  = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0)
                       else "cpu")


##############################
# Driver Function
##############################
def main():

    # Create models
    dNet = DNet(1, 28, ngpu).to(device)
    gNet = GNet(1, 100, 28, ngpu).to(device)

    # DCGAN initialization
    dNet.apply(funcs.weights_init)
    gNet.apply(funcs.weights_init)

    # Uncomment to print model(s)
    # print(dNet)
    # print(gNet)

    # Training parameter
    batch_size = 100
    num_epochs = 200

    # Load Data
    data = funcs.mnist_data(datarun)

    # Create loader with data - iterable object
    dataloader = torch.utils.data.DataLoader(data, batch_size = batch_size,
                                              shuffle = True,
                                              num_workers = workers)

    # Create logger instance
    logger = Logger(model_name = "Test_GAN_GPU", data_name = "MNIST")

    # Optimization & training params
    num_batches = len(dataloader)
    d_optim = optim.Adam(dNet.parameters(), lr = 2e-4)
    g_optim = optim.Adam(gNet.parameters(), lr = 2e-4)
    loss_fn = nn.BCELoss().to(device)

    funcs.train(dataloader, data, logger, num_batches, num_epochs, d_optim,
                g_optim, loss_fn, gNet, dNet)

if __name__ == '__main__':
    main()
