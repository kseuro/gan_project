# Basic implementation of a GAN using MNSIT dataset
# Kai Kharpertian
# Feb. 2019
# Original implementation by: Diego Gomez Mosquera (Feb. 1, 2018)

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
import seq_nets
import seq_gan_funcs
##############################
# Hardware Env.
##############################
GPUID = 0
torch.cuda.device(GPUID)

##############################
# Driver Function
##############################
def main():

    DNet = DNet()
    DNet.cuda(GPUID)
    GNet = GNet()
    GNet.cuda(GPUID)

    # Load Data
    data = mnist_data()

    # Create loader with data - iterable object
    data_loader = torch.utils.data.DataLoader(data, batch_size = 100,
                                              shuffle = True)

    # Create logger instance
    logger = Logger(model_name = "Test_GAN_GPU", data_name = "MNIST")

    # Test generator every few steps
    num_test_samples = 16
    test_noise = noise(num_test_samples)

    # Optimization & training params
    num_epochs  = 200
    num_batches = len(data_loader)
    d_optim = optim.Adam(DNet.parameters(), lr = 2e-4)
    g_optim = optim.Adam(GNet.parameters(), lr = 2e-4)
    loss_fn = nn.BCELoss().cuda(GPUID)

    train(data_loader, data, logger, num_batches, num_epochs, d_optim, g_optim,
          loss_fn, GNet, DNet)

if __name__ == '__main__':
    main()
