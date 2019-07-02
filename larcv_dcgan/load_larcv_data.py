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
_files1 = ["/media/hdd1/larbys/ssnet_dllee_trainingdata/train01.root"]
_files2 = ["/media/hdd1/larbys/ssnet_dllee_trainingdata/train02.root"]

iomanager0 = larcv.IOManager(larcv.IOManager.kREAD,"", larcv.IOManager.kTickBackward)
iomanager0.add_in_file(_files0[0])
iomanager0.add_in_file(_files1[0])
iomanager0.add_in_file(_files2[0])
iomanager0.initialize()

# make sure num imgs is divisible by 10
nEntries  = iomanager0.get_n_entries()
if ((nEntries % 10) != 0) and (nEntries > 0):
    while (nEntries % 10) != 0:
        nEntries -= 1

print('nEntries:', nEntries)
index = 0
batchsize = 10
stop = index + batchsize

producer = "wire"
data_img = np.zeros((batchsize, 1, 512, 512), dtype=np.float32)
data_root = '/home/kseuro/larcv_png_data/larcv_png_1_ch/larcv_png_1_ch/'

while (stop != nEntries):
    for i in range(index, stop):
        iomanager0.read_entry(i)
        ev_img      = iomanager0.get_data(larcv.kProductImage2D, producer)
        data_vec    = ev_img.Image2DArray()         # len(data_vec) = 3
        ADC_Y_plane = larcv.as_ndarray(data_vec[2]) # shape = (512, 512)
        ADC_Y_plane[ADC_Y_plane < 10] = 0
        data_img[i,0,:,:] = ADC_Y_plane

        # Output images in batch to disk
        prefix = 'larcv' + str(i - index)              # larcv1234
        im_name = prefix + '.png'                      # larcv1234.png
        out_dir = data_root + 'larcv' + str(i - index) # data_root/larcv1234
        print('out_dir:', out_dir)
        print('im_name:', im_name)
        make_dir(out_dir)
        im_out = out_dir + '/' + im_name # data_root/larcv1234/larcv1234.png
        cv.imwrite(im_out, ADC_Y_plane*10)

        index = stop
        stop += batchsize
