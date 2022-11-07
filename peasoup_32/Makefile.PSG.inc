# This is where common definitions go

#cuda setup
CUDA_DIR   = /shared/apps/cuda/CUDA-v6.0.26

#Thrust 1.7.1 or higher
THRUST_DIR = $(CUDA_DIR)/include

#dedisp setup
DEDISP_DIR = /home-2/jkraus/workspace/PulsarSearch/dedisp

GCC       = gcc
GXX       = g++
AR        = ar
NVCC      = $(CUDA_DIR)/bin/nvcc -lineinfo
SHELL     = /bin/bash
UCFLAGS   = -DUSE_NVTX
