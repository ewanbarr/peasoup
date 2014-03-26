# This is where common definitions go

#cuda setup
CUDA_DIR   = /usr/local/cuda-5.0/

#Thrust 1.7.1 or higher
THRUST_DIR = /mnt/home/ebarr/Soft/thrust/thrust 

#dedisp setup
DEDISP_DIR = /lustre/home/ebarr/Soft/dedisp

GCC       = gcc
GXX       = g++
AR        = ar
NVCC      = $(CUDA_DIR)/bin/nvcc
SHELL     = /bin/bash
UCFLAGS   = -DUSE_NVTX
