# This is where common definitions go

INSTALL_DIR = /usr/local

# CUDA setup
CUDA_DIR   = /usr/local/cuda/

# Thrust 1.7.1 or higher
THRUST_DIR = /usr/local/cuda/

# Dedisp setup
DEDISP_DIR = /usr/local/

GCC       = gcc
GXX       = g++
AR        = ar
NVCC      = $(CUDA_DIR)/bin/nvcc
SHELL     = /bin/bash
UCFLAGS   = -DUSE_NVTX

# Detect CUDA version
CUDA_VERSION ?= $(shell $(NVCC) --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)

# Set GPU architecture flags dynamically based on CUDA version
GPU_ARCH_FLAG =

ifeq ($(CUDA_VERSION),)
    $(warning "CUDA version not detected, using default compute capabilities")
    GPU_ARCH_FLAG = -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80
else
    CUDA_MAJOR := $(shell echo $(CUDA_VERSION) | cut -d'.' -f1)
    CUDA_MINOR := $(shell echo $(CUDA_VERSION) | cut -d'.' -f2)

    ifneq ($(shell expr $(CUDA_MAJOR) \>= 10), 0)
        GPU_ARCH_FLAG += -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70
    endif
    ifneq ($(shell expr $(CUDA_MAJOR) \>= 11), 0)
        GPU_ARCH_FLAG += -gencode arch=compute_75,code=sm_75
    endif
    ifneq ($(shell expr $(CUDA_MAJOR) \>= 11 \& $(CUDA_MINOR) \>= 2), 0)
        GPU_ARCH_FLAG += -gencode arch=compute_80,code=sm_80
    endif
    ifneq ($(shell expr $(CUDA_MAJOR) \>= 11 \& $(CUDA_MINOR) \>= 8), 0)
        GPU_ARCH_FLAG += -gencode arch=compute_86,code=sm_86
    endif
    ifneq ($(shell expr $(CUDA_MAJOR) \>= 12), 0)
        GPU_ARCH_FLAG += -gencode arch=compute_89,code=sm_89
    endif
    ifneq ($(shell expr $(CUDA_MAJOR) \>= 12 \& $(CUDA_MINOR) \>= 2), 0)
        GPU_ARCH_FLAG += -gencode arch=compute_90,code=sm_90
    endif
endif