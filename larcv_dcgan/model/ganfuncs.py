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
import ROOT
from   larcv    import larcv
from   datetime import datetime

##############################
# Helper classes
##############################
class SegData:
    def __init__(self):
        self.dim     = None
        self.images  = None  # adc image
        self.labels  = None  # labels
        self.weights = None  # weights
        return

    def shape(self):
        if self.dim is None:
            raise ValueError("SegData instance hasn't been filled yet")
        return self.dim

class LArCV1Dataset:
    """
        Class for creating numpy image data objects from ROOT files.
        Object needs cfgfile string in order to locate and load ROOT files.
        methods: init: creates an instan of data file interface.
                 getbatch: returns LArCV image as torch tensor.
    """
    def __init__(self, name, cfgfile ):
        # inputs
        # cfgfile: path to configuration.
        self.name = name
        self.cfgfile = cfgfile
        return

    def init(self):
        self.io = larcv.ThreadDatumFiller(self.name)
        self.io.configure(self.cfgfile)
        self.nentries = self.io.get_n_entries()
        self.io.set_next_index(0)
        print("[LArCV1Data] able to create ThreadDatumFiller")
        return

    def getbatch(self, batchsize):
        self.io.batch_process(batchsize)
        time.sleep(0.1)
        itry = 0
        while self.io.thread_running() and itry<100:
            time.sleep(0.01)
            itry += 1
        if itry>=100:
            raise RuntimeError("Batch Loader timed out")
        # fill SegData object
        data = SegData()

        dimv = self.io.dim() # c++ std vector through ROOT bindings
        self.dim  = (dimv[0], dimv[1], dimv[2], dimv[3] )
        self.dim3 = (dimv[0], dimv[2], dimv[3] )

        # numpy arrays
        data.np_images    = np.zeros( self.dim,  dtype=np.float32 )
        print("BEFORE DATA.NP_IMAGES")
        input()
        # Dataloader Seg Faults at this line
        data.np_images[:] = larcv.as_ndarray(self.io.data()).reshape(self.dim)[:]
        # Dataloader Seg Faults at this line

        # pytorch tensors
        data.images = torch.from_numpy(data.np_images)

        return data

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

def random_noise(dim, nz):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    return torch.randn( (dim * dim, nz) ).view(-1, nz, 1, 1).to(device)

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
    make_dir(out_dir)
    start_time = time
    now        = datetime.now()
    date       = now.strftime("%m-%d-%Y")
    time       = now.strftime("%H-%M-%S")
    date_time  = date + '_' + time
    out_dir    = out_dir + date_time

    return start_time, out_dir, now

def save_model(model, savepath, ext, G):
    if G == True:
        out_params = savepath + '/generator_params'
    elif G == False:
        out_params = savepath + '/discriminator_params'

    make_dir(savepath)
    torch.save(model.state_dict(), out_params + ext)

def save_outputs(model, out_dir, epoch, num_epochs, datetime, fixed_noise):
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
    show_results(model, epoch, fixed_noise, savepath, isFixed = False)

    # results using fixed noise
    savepath = out_dir_fixed_ + '/g_fixed_result'
    show_results(model, epoch, fixed_noise, savepath, isFixed = True)

def show_results(model, epoch, fixed_noise, savepath, isFixed=False):
    """
        Function for visualizing intermediate results during and after training.
    """
    z_ = random_noise(1, 100)

    if isFixed:
        test_imgs = model(fixed_noise)
    else:
        test_imgs = model(z_)

    sfg = 1 # size of grid in figure
    fig, axes = plt.subplots(sfg, sfg, figsize = (sfg, sfg))
    for i, j in itertools.product(range(sfg), range(sfg)):
        axes[i, j].get_xaxis().set_visible(False)
        axes[i, j].get_yaxis().set_visible(False)

    for k in range(1):
        axes[i, j].cla()
        axes[i, j].imshow(test_imgs[k, 0].cpu().data.numpy(), cmap = 'gray')

    label = 'Epoch_{}'.format(epoch)
    fig.text(0.5, 0.04, label, ha = 'center')
    plt.savefig(savepath)
    plt.close()

def plot_losses(out_dir, date_time, G_losses, D_losses):
    """
        Save and visualize results after the completion of training.
    """
    make_dir(out_dir)

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
