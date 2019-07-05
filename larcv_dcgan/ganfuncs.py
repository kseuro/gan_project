# ganfuncs.py
# purpose: collection of helper functions and classes for
#          training a DCGAN on the LArCV1 dataset
# Kai Kharpertian
# Tufts University
# Department of Physics

##############################
# Import scripts
##############################
import os, time
import errno
import torch
import torch.nn                 as nn
import torch.nn.parallel
import torch.backends.cudnn     as cudnn
import torch.optim              as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets     as dset
import torchvision.transforms   as transforms
import torchvision.utils        as vutils
import numpy                    as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import imageio
import ROOT
from   larcv    import larcv
from   datetime import datetime

##############################
# Helper classes
##############################
class CenterCropLongEdge(object):
  """Crops the given PIL Image on the long edge.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    return transforms.functional.center_crop(img, min(img.size))

  def __repr__(self):
    return self.__class__.__name__

class Print(nn.Module):
    '''
        Outputs the shape of convolutional layers in model.
        Call Print() in between layers to get shape output to
            the terminal.
    '''
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

##############################
# Helper functions
##############################
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
  images = []
  dir = os.path.expanduser(dir)
  for target in sorted(os.listdir(dir)):
    d = os.path.join(dir, target)
    if not os.path.isdir(d):
      continue

    for root, _, fnames in sorted(os.walk(d)):
      for fname in sorted(fnames):
        if is_image_file(fname):
          path = os.path.join(root, fname)
          item = (path, class_to_idx[target])
          images.append(item)
    length = len(images)
    for i in range(length - length/2, length):
      print(images[i])
    input('Waiting for input')

  return images

def weights_init(m):
    """
    Custon weight init based on DCGAN paper recommendations of
    mean = 0.0, stdev = 0.2
    Input:   - initialized model
    Returns: - reinitialized conv, conv-transpose, and batch-norm
               layers that meet above criteria.
             - NOTE: this function moves the weights on the cpu!
                     Move network onto GPU after calling!
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def random_noise(dim, nz, device):
    return torch.randn( (dim, nz) ).view(-1, nz, 1, 1).to(device)

def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def train_hist():
    """
        Returns dictionary for storing training results
    """
    train_hist                   = {}
    train_hist['D_losses']       = []
    train_hist['G_losses']       = []
    train_hist['per_epoch_time'] = []
    train_hist['total_time_s']   = []
    train_hist['total_time_h']   = []
    train_hist['total_iter']     = []

    return train_hist

def train_start(out_dir, time):
    """
        Returns: - custom out_dir for saving model outputs
                 - start_time for computing training time
    """
    start_time = time
    now        = datetime.now()
    date       = now.strftime("%m-%d-%Y")
    time       = now.strftime("%H-%M-%S")
    date_time  = date + '_' + time
    out_dir    = out_dir + date_time
    make_dir(out_dir)

    return start_time, date_time, out_dir, now

def save_model(model, savepath, ext, G):
    if G == True:
        out_params = savepath + '/generator_params'
    elif G == False:
        out_params = savepath + '/discriminator_params'
    torch.save(model.state_dict(), out_params + ext)

def save_outputs(model, out_dir, epoch, num_epochs, datetime, fixed_noise, device):
    '''
        Creates a set of directories for saving G's output at the end of
         each epoch. Both random noise and fixed noise results are saved.
        Out_dir string is created in train_start function.
    '''
    out_dir_random = out_dir + '/random_result_' + str(epoch + 1)
    out_dir_fixed_ = out_dir + '/fixed_result_'  + str(epoch + 1)

    make_dir(out_dir)
    make_dir(out_dir_random)
    make_dir(out_dir_fixed_)

    # results using random noise
    savepath = out_dir_random + '/g_random_result'
    show_results(model, epoch, fixed_noise, savepath, device, isFixed = False)

    # results using fixed noise
    savepath = out_dir_fixed_ + '/g_fixed_result'
    show_results(model, epoch, fixed_noise, savepath, device, isFixed = True)

def show_results(model, epoch, fixed_noise, savepath, device, isFixed=False):
    """
        Function for visualizing intermediate results during and after training.
    """
    z_ = random_noise(1, 100, device)

    if isFixed:
        test_imgs = model(fixed_noise)
    else:
        test_imgs = model(z_)

    # sfg = 1 # size of grid in figure
    # fig, axes = plt.subplots(sfg, sfg, figsize = (sfg, sfg))
    # for i, j in itertools.product(range(sfg), range(sfg)):
    #     axes[i, j].get_xaxis().set_visible(False)
    #     axes[i, j].get_yaxis().set_visible(False)
    #
    # for k in range(1):
    #     axes[i, j].cla()
    #     axes[i, j].imshow(test_imgs[k, 0].cpu().data.numpy(), cmap = 'gray')

    make_dir(savepath)
    label = savepath + '/Epoch_{}'.format(epoch) + '.png'
    test_img = test_imgs[0, 0].cpu().data.numpy()
    plt.imsave(label, test_img, cmap='gray')

def plot_losses(out_dir, date_time, G_losses, D_losses):
    """
        Save and visualize results after the completion of training.
    """
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
        Generates an animation of G's outputs using saved .png files.
        str1: selects whether the image is fixed or random
        str2: name of .png image
        str3: output name of animation (must have .gif extension)
    '''
    images = []
    for e in range(num_epochs):
        img_name = out_dir + str1  + str(e + 1) + str2
        images.append(imageio.imread(img_name))
    imageio.mimsave(out_dir + str3, images, fps = 5)

def save_ani(out_dir, num_epochs, fixed = True, random = True):
    if fixed == True:
        str1 = '/fixed_result_'
        str2 = '/g_fixed_result.png'
        str3 = '/generator_animation_fixed.gif'
        gen_ani(str1, str2, str3, num_epochs, out_dir)

    if random == True:
        str1 = '/random_result_'
        str2 = '/g_random_result.png'
        str3 = '/generator_animation_random.gif'
        gen_ani(str1, str2, str3, num_epochs, out_dir)
