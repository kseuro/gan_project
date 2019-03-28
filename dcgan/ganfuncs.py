# ganfuncs.py
# Helper classes and functions for training DCGAN

##############################
# Import scripts
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
import itertools
import imageio
from   datetime import datetime

##############################
# Helper classes and functions
##############################
def mnist_data(image_size, batch_size, workers, dataroot):
    compose = transforms.Compose( [transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize( (.5, .5, .5),
                                                         (.5, .5, .5)),
                                  ])
    dataset = dset.MNIST(root = root, train = True, transform = compose,
                         download = False)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size = batch_size,
                                             shuffle = True,
                                             num_workers = workers)
    return dataloader

class Print(nn.Module):
    '''
        Outputs the shape of convolutional layers in model.
        Call Print() inbetween layers to get shape output to
            the terminal.
    '''
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

def weights_init(m):
    """
    Custon weight init based on DCGAN paper recommendations of
    mean = 0.0, stdev = 0.2
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

def random_noise(dim, nz, device):
    return torch.randn( (dim * dim, nz) ).view(-1, nz, 1, 1).to(device)

##############################
# Saving and plotting
##############################
def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def train_hist():
    train_hist                   = {}
    train_hist['D_losses']       = []
    train_hist['G_losses']       = []
    train_hist['per_epoch_time'] = []
    train_hist['total_time']     = []

    return train_hist

def train_start():
    print('#################')
    print('Start of training')
    print('#################')
    start_time = time.time()
    now       = datetime.now()
    date_time = now.strftime("%m-%d-%Y, %H:%M:%S")
    out_dir   = '/media/hdd1/kai/projects/gan_project/dcgan/MNSIT' + date_time

    return start_time, out_dir, now, date_time

def save_model(model, savepath, ext, G):
    if G == True:
        savepath = savepath + '/generator_params/'
    elif G == True:
        savepath = savepath + '/discriminator_params/'

    make_dir(savepath)
    torch.save(model.state_dict(), savepath + ext)

def save_outputs(model, out_dir, epoch, num_epochs, datetime, fixed_noise):
    '''
        Creates a set of directories for saving G's output at the end of
        each epoch. Both random noise and fixed noise results are saved.
    '''
    out_dir_random = out_dir + 'random_result_' + str(epoch + 1)
    out_dir_fixed_ = out_dir + 'fixed_result_'  + str(epoch + 1)

    make_dir(out_dir)
    make_dir(out_dir_random)
    make_dir(out_dir_fixed_)

    # results using random noise
    savepath = out_dir_random + '/g_random_result'
    show_results(model, (num_epochs + 1), fixed_noise, savepath,
                          show = False, isFixed = False)
    # results using fixed noise
    savepath = out_dir_fixed_ + '/g_fixed_result'
    show_results(model, (num_epochs + 1), fixed_noise, savepath,
                          show = False, isFixed = True)

def show_results(model, num_epochs, fixed_noise, savepath, show=False, isFixed=False):

    z_ = random_noise(5, 100)

    if isFixed:
        test_imgs = model(fixed_noise)
    else:
        test_imgs = model(z_)

    size_fig_grid = 5
    fig, axes = plt.subplots(size_fig_grid, size_fig_grid, figsize = (5, 5))
    for i, j in itertools.product(range(size_fig_grid), range(size_fig_grid)):
        axes[i, j].get_xaxis().set_visible(False)
        axes[i, j].get_yaxis().set_visible(False)

    for k in range(5 * 5):
        i = k // 5
        j = k % 5
        axes[i, j].cla()
        axes[i, j].imshow(test_imgs[k, 0].cpu().data.numpy(), cmap = 'gray')

    label = 'Epoch_{}'.format(num_epochs)
    fig.text(0.5, 0.04, label, ha = 'center')
    plt.savefig(savepath)

    if show:
        plt.show()
    else:
        plt.close()

def plot_losses(out_dir, date_time, G_losses, D_losses):
    """
        Save and visualize results after the completion of training
    """

    make_dir(out_dir)

    # # Save G's sample images               # [64, 1, 64, 64]
    # for index, arr in enumerate(img_list): # index, object
    #     for img in range(arr[0].shape[0]): # 64
    #         # (filename, array)
    #         plt.imsave(img_names[index] + '_' + str(img) + '.png',
    #                    arr[img, :, :, :].reshape((64, 64)), cmap = 'gray')

    # Plot G and D losses
    plt.figure()
    plt.title("Generator and Discriminator Loss Curves")
    plt.plot(G_losses, label = 'G')
    plt.plot(D_losses, label = 'D')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('{}/{}.png'.format(out_dir, date_time), dpi = 300)

def gen_ani(str1, str2, str3, num_epochs, out_dir):
    '''
        Generates an animation of G's outputs using .png files.
        str1: chooses fixed or random images
        str2: name of .png image
        str3: output name of animation (must have .gif extension)
    '''
    images = []
    for e in range(num_epochs):
        img_name = out_dir + str1  + str(e + 1) + str2
        images.append(imageio.imread(img_name))
    imageio.mimsave(out_dir + str3, images, fps=30)

def save_ani(out_dir, num_epochs, fixed = True, random = True):
    if fixed == True:
        str1 = 'fixed_result_'
        str2 = '/g_fixed_result.png'
        str3 = 'generator_animation_fixed.gif'
        gen_ani(str1, str2, str3, num_epochs, out_dir)

    if random == True:
        str1 = 'random_result_'
        str2 = '/g_random_result.png'
        str3 = 'generator_animation_random.gif'
        gen_ani(str1, str2, str3, num_epochs, out_dir)
