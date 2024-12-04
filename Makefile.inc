# This is where common definitions go

INSTALL_DIR = /hercules/scratch/vishnu/VISHNU_PEASOUP/binaries

#cuda setup
CUDA_DIR   = /mpcdf/soft/CentOS_7/packages/x86_64/cuda/10.2.89

#Thrust 1.7.1 or higher
THRUST_DIR = /mpcdf/soft/CentOS_7/packages/x86_64/cuda/10.2.89

# dedisp setup
DEDISP_DIR = /hercules/scratch/vishnu/VISHNU_PEASOUP/binaries

GCC       = gcc
GXX       = g++
AR        = ar
NVCC      = $(CUDA_DIR)/bin/nvcc
SHELL     = /bin/bash
UCFLAGS   = -DUSE_NVTX
