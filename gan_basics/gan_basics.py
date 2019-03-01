# Basic implementation of a GAN using MNSIT dataset
# Computation carried out on CPU
# Kai Kharpertian
# Feb. 2019
# Original implementation by: Diego Gomez Mosquera (Medium | Feb. 1, 2018)

###############
# Dependencies
###############
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

# Visualization
from utils import Logger

###############
# Dataset
###############
def mnist_data():
    compose = transforms.Compose( [transforms.ToTensor(),
                                   transforms.Normalize( (.5, .5, .5),
                                                         (.5, .5, .5))
                                  ])
    root = '/media/hdd1/kai/mnist_dataset'
    return datasets.MNIST(root = root, train = True, transform = compose,
                          download = False)

# Load Data
data = mnist_data()

# Create loader with data - iterable object
data_loader = torch.utils.data.DataLoader(data, batch_size = 100,
                                          shuffle = True)

# Number of batches
num_batches = len(data_loader)

# Functionality
def img_to_vec(img):
    return img.view(img.size(0), 784)

def vec_to_img(vec):
    return  vec.view(vec.size(0), 1, 28, 28)

def noise(size):
    """
    Creates 1D vector of gaussian noise
    """
    n = Variable(torch.randn(size, 100))
    return n

def ones_target(size):
    """
    Tensor containing ones. Shape = size
    """
    ones = Variable(torch.ones(size, 1))
    return ones

def zeros_target(size):
    """
    Tensor containing zeroes. Shape = size
    """
    zeros = Variable(torch.zeros(size,1))
    return zeros

###############
# Discriminator
# Input: Flattened images of dataset
# Returns: Prob. of input belonging to dataset

# Architecture: 3 hidden layers with LReLU & Dropout
#               Sigmoid applied to output
###############
class DNet(torch.nn.Module):
    def __init__(self):
        super(DNet, self).__init__()
        n_features = 784 # 28x28 input image = 784 flat vector
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 1024), nn.LeakyReLU(0.2), nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            nn.Linear(256, n_out), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

DNet = DNet()

###############
# Generator
###############
class GNet(torch.nn.Module):
    def __init__(self):
        super(GNet, self).__init__()
        n_features = 100
        n_out = 784

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256), nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512), nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024), nn.LeakyReLU(0.2)
        )
        self.out = nn.Sequential(
            nn.Linear(1024, n_out), nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

GNet = GNet()

###############
# Optimization
###############
d_optim = optim.Adam(DNet.parameters(), lr = 2e-4)
g_optim = optim.Adam(GNet.parameters(), lr = 2e-4)

# Loss Function
loss_fn = nn.BCELoss()

def train_DNet(optimizer, real_data, G_data):
    N = real_data.size(0)
    optimizer.zero_grad() # Reset gradients

    # 1.1 Train on Real Data
    pred_real = DNet(real_data)
    error_real = loss_fn(pred_real, ones_target(N)) # Calc error
    error_real.backward() # Backprop

    # 1.2 Train on Fake Data
    pred_fake = DNet(fake_data)
    error_fake = loss_fn(pred_fake, zeros_target(N)) # calc error
    error_fake.backward() # backprop

    # Update weights with gradients
    optimizer.step()

    # Return error and preds for real and fake inputs
    return error_real + error_fake, pred_real, pred_fake

def train_GNet(optimizer, fake_data):
    N = fake_data.size(0)
    optimizer.zero_grad() # Reset gradients

    # Sample noise and gen fake data
    pred = DNet(fake_data)

    # Calc error and backprop
    error = loss_fn(pred, ones_target(N))
    error.backward()

    # Update weights with gradients
    optimizer.step()

    # Return error
    return error

# Test generator every few steps
num_test_samples = 16
test_noise = noise(num_test_samples)

###############
# Training
###############
# Create logger instance
logger = Logger(model_name = "Test_GAN", data_name = "MNIST")

# Total number of training epochs
num_epochs = 200

for epoch in range(num_epochs):
    for n_batch, (real_batch, _ ) in enumerate(data_loader):
        N = real_batch.size(0)

        # 1. Train DNet
        real_data = Variable(img_to_vec(real_batch))

        # Gen. fake data and detach
        # detach so that grad not calc'd for GNet
        fake_data = GNet(noise(N)).detach()

        # Train D
        D_error, D_pred_real, D_pred_fake = \
            train_DNet(d_optim, real_data, fake_data)

        # Train GNet
        # Gen fake data
        fake_data = GNet(noise(N))

        # Train G
        G_error = train_GNet(g_optim, fake_data)

        # Log batch error
        logger.log(D_error, G_error, epoch, n_batch, num_batches)

        # Disp. progress every few batches
        if n_batch % 100 == 0:
            test_images = vec_to_img(GNet(test_noise))
            test_images = test_images.data

            logger.log_images(
                test_images, num_test_samples, epoch, n_batch, num_batches
            )
            # Disp status logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches, D_error, G_error,
                D_pred_real, D_pred_fake
            )
