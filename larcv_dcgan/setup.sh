#!/bin/bash

home=$PWD
source ~/setup_root6.sh
source ~/setup_cuda.sh

cd larcv
source configure.sh

cd $home

cd larcvdataset
source setenv.sh

cd $home

export CUDA_VISIBLE_DEVICES=0,1
