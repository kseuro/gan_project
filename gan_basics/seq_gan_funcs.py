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

def train(data_loader, data, logger, num_batches, num_epochs, d_optim, g_optim,
           loss_fn, GNet, DNet):
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
