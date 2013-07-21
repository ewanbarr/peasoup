# This is where common definitions go

#cuda setup
CUDA_DIR   = /usr/local/cuda-5.0/
THRUST_DIR = /usr/local/cuda-5.0/include/

#dedisp setup
DEDISP_DIR = /lustre/home/ebarr/Soft/dedisp

# Fermi
GPU_ARCH = sm_20

GCC       = gcc
GXX       = g++
AR        = ar
NVCC      = /usr/local/cuda-5.0/bin/nvcc
SHELL     = /bin/csh