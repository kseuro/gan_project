import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.autograd.variable import Variable
from seq_nets import GNet, DNet

def mnist_data():
    compose = transforms.Compose( [transforms.ToTensor(),
                                   transforms.Normalize( (.5, .5, .5),
                                                         (.5, .5, .5))
                                  ])
    root = '/media/hdd1/kai/mnist_dataset'
    return datasets.MNIST(root = root, train = True, transform = compose,
                          download = False)

def img_to_vec(img):
    return img.view(img.size(0), 784)

def vec_to_img(vec):
    return  vec.view(vec.size(0), 1, 28, 28)

def noise(size):
    """
    Creates 1d vector of gaussian noise
    """
    n = Variable(torch.randn(size, 100)).cuda()
    return n

def ones_target(size):
    """
    Tensor containing ones. Shape = size
    """
    ones = Variable(torch.ones(size, 1)).cuda()
    return ones

def zeros_target(size):
    """
    Tensor containing zeroes. Shape = size
    """
    zeros = Variable(torch.zeros(size,1)).cuda()
    return zeros

def train_dNet(optimizer, real_data, fake_data, loss_fn, dNet):
    N = real_data.size(0)
    optimizer.zero_grad() # Reset gradients

    # 1.1 Train on Real data
    pred_real = dNet(real_data)
    error_real = loss_fn(pred_real, ones_target(N)) # Calc error
    error_real.backward() # Backprop

    # 1.2 Train on Fake data
    pred_fake = dNet(fake_data)
    error_fake = loss_fn(pred_fake, zeros_target(N)) # calc error
    error_fake.backward() # backprop

    # Update weights with gradients
    optimizer.step()

    # Return error and preds for real and fake inputs
    return error_real + error_fake, pred_real, pred_fake

def train_gNet(optimizer, fake_data, loss_fn, gNet, dNet):
    N = fake_data.size(0)
    optimizer.zero_grad() # Reset gradients

    # Sample noise and gen fake data
    pred = dNet(fake_data)

    # Calc error and backprop
    error = loss_fn(pred, ones_target(N))
    error.backward()

    # Update weights with gradients
    optimizer.step()

    # Return error
    return error

def train(data_loader, data, logger, num_batches, num_epochs, d_optim, g_optim,
           loss_fn, gNet, dNet):
    for epoch in range(num_epochs):
        # n_batch = x, real_batch = target
        for n_batch, (real_batch, _ ) in enumerate(data_loader):
            N = real_batch.size(0)
            real_batch = real_batch.cuda()

            # 1. Train dNet
            real_data = Variable(img_to_vec(real_batch))

            # gen. fake data and detach
            # detach so that grad not calc'd for gNet
            fake_data = gNet(noise(N)).detach()

            # Train d
            d_error, d_pred_real, d_pred_fake = \
                train_dNet(d_optim, real_data, fake_data, loss_fn, dNet)

            # Train gNet
            # gen fake data
            fake_data = gNet(noise(N))

            # Train g
            g_error = train_gNet(g_optim, fake_data, loss_fn, gNet, dNet)

            # Log batch error
            logger.log(d_error, g_error, epoch, n_batch, num_batches)

            # Test generator every few steps
            num_test_samples = 16
            test_noise = noise(num_test_samples)

            # disp. progress every few batches
            if n_batch % 100 == 0:
                test_images = vec_to_img(gNet(test_noise))
                test_images = test_images.data

                logger.log_images(
                test_images, num_test_samples, epoch, n_batch, num_batches
                )
                # disp status logs
                logger.display_status(
                epoch, num_epochs, n_batch, num_batches, d_error, g_error,
                d_pred_real, d_pred_fake
                )
