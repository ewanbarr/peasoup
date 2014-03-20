# This is where common definitions go

#cuda setup
CUDA_DIR   = $(CUDAROOT)
THRUST_DIR = $(CUDAROOT)/include/

#dedisp setup
DEDISP_DIR = /homeb/zam/jkraus/workspace/PulsarSearch/dedisp

GCC       = gcc
GXX       = g++
AR        = ar
NVCC      = $(CUDAROOT)/bin/nvcc
SHELL     = /bin/bash
UCFLAGS   = -DUSE_NVTX