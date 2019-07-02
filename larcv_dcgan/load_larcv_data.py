import os
import ROOT
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from larcv import larcv

# Define function to create a dir
def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# Define an IO manager

_files0 = ["/media/hdd1/larbys/ssnet_dllee_trainingdata/train00.root"]
# _files1 = ["/media/hdd1/larbys/ssnet_dllee_trainingdata/train01.root"]
# _files2 = ["/media/hdd1/larbys/ssnet_dllee_trainingdata/train02.root"]

iomanager0 = larcv.IOManager(larcv.IOManager.kREAD,"", larcv.IOManager.kTickBackward)
iomanager0.add_in_file(_files0[0])
# iomanager0.add_in_file(_files1[0])
# iomanager0.add_in_file(_files2[0])
iomanager0.initialize()

root_dir = '/home/kseuro/larcv_png_data/larcv_png_3_ch/larcv_png_3_ch/'
nEntries  = iomanager0.get_n_entries()
batchsize = 1
index = random.randint(0, nEntries)
stop  = index + batchsize

producer = "wire"
data_img = np.zeros((batchsize, 1, 512, 512), dtype=np.float32)

for i in range(index, stop):
    iomanager0.read_entry(i)
    ev_img      = iomanager0.get_data(larcv.kProductImage2D, producer)
    data_vec    = ev_img.Image2DArray()         # len(data_vec) = 3
    ADC_Y_plane = larcv.as_ndarray(data_vec[2]) # shape = (512, 512)
    ADC_Y_plane[ADC_Y_plane < 10] = 0
    data_img[(i - index),0,:,:] = ADC_Y_plane

    # Output images in batch
    im_name = 'larcv' + str(i - index) + '.png'
    out_dir = root_dir + im_name
    make_dir(out_dir)
    cv.imwrite(outdir, ADC_Y_plane*10)

# data_img = torch.from_numpy(data_img)
