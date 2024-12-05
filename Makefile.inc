# This is where common definitions go

INSTALL_DIR = /usr/local

#cuda setup
CUDA_DIR   = /usr/local/cuda/

#Thrust 1.7.1 or higher
THRUST_DIR = /usr/local/cuda/

# dedisp setup
DEDISP_DIR = /usr/local/

GCC       = gcc
GXX       = g++
AR        = ar
NVCC      = $(CUDA_DIR)/bin/nvcc
SHELL     = /bin/bash
UCFLAGS   = -DUSE_NVTX
