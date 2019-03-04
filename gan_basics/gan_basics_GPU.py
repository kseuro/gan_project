# Basic implementation of a GAN using MNSIT dataset
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
from seq_nets import GNet, DNet
import seq_gan_funcs as funcs
##############################
# Hardware Env.
##############################
GPUID = 0
torch.cuda.device(GPUID)

##############################
# Driver Function
##############################
def main():

    dNet = DNet()
    dNet = dNet.cuda(GPUID)
    gNet = GNet()
    gNet = gNet.cuda(GPUID)

    # Training parameter
    batch_size = 100
    num_epochs = 200

    # Load Data
    data = funcs.mnist_data()

    # Create loader with data - iterable object
    data_loader = torch.utils.data.DataLoader(data, batch_size = batch_size,
                                              shuffle = True)

    # Create logger instance
    logger = Logger(model_name = "Test_GAN_GPU", data_name = "MNIST")

    # Optimization & training params
    num_batches = len(data_loader)
    d_optim = optim.Adam(dNet.parameters(), lr = 2e-4)
    g_optim = optim.Adam(gNet.parameters(), lr = 2e-4)
    loss_fn = nn.BCELoss().cuda(GPUID)

    funcs.train(data_loader, data, logger, num_batches, num_epochs, d_optim,
                g_optim, loss_fn, gNet, dNet)

if __name__ == '__main__':
    main()
