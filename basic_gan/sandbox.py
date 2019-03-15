import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

image_size = 28
batch_size = 1
workers    = 2
ngpu       = 1
real_label = 1
device = torch.device("cuda:2" if (torch.cuda.is_available() and ngpu > 0)
                        else "cpu")
dataroot = '/media/hdd1/kai/datasets/mnist'
def mnist_data(root):
    compose = transforms.Compose( [transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize( (.5, .5, .5),
                                                         (.5, .5, .5)),
                                  ])
    return dset.MNIST(root = root, train = True, transform = compose,
                      download = False)

dataset = mnist_data(dataroot)
dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size = batch_size,
                                        shuffle = True,
                                        num_workers = workers)

for i, data in enumerate(dataloader, 0):
    print(i)
    print(type(data))
    print(len(data))
    real_cpu = data[0].to(device)
    b_size   = real_cpu.size(0)
    label    = torch.full((b_size,), real_label, device = device)

print(b_size)
print(type(b_size))
print(type(real_cpu))
print(real_cpu.size())
print(label)
