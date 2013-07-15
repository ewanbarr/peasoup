include Makefile.inc

# Output directories                                                                                                                        
BIN_DIR     = ./bin
OBJ_DIR     = ./obj

LIB_DIR = ./lib
INCLUDE_DIR := ./include

#Major dependencies
DEDISP_DIR = /lustre/home/ebarr/Soft/dedisp
DEDISP = -I${DEDISP_DIR}/include/ -L${DEDISP_DIR}/lib -ldedisp

TCLAP_DIR = 

SRC_DIR  := ./src
DATA_TYPES_DIR := ${SRC_DIR}/data_types
TRANSFORMS_DIR := ${SRC_DIR}/transforms
OPTIMISE := -O3
DEBUG    := -g

INCLUDE   := -I$(INCLUDE_DIR) -I$(THRUST_DIR) -I$(TCLAP_DIR)/tclap/
CUDA_LIBS := -L$(CUDA_DIR)/lib64 -lcuda -lcudart


CUFLAGS  := --compiler-options -Wall --machine 64 -arch=$(GPU_ARCH) -Xcompiler ${DEBUG}
FLAGS   := -fPIC

#EXE_FILES := newGPUseek.cu
#EXE_NAMES := newGPUseek 

CPPFILES = ${OBJ_DIR}/filterbank.o ${OBJ_DIR}/dedisperser.o

all: ${CPPFILES}


${OBJ_DIR}/filterbank.o: ${DATA_TYPES_DIR}/filterbank.cpp	
	${GXX} -c ${OPTIMISE} ${FLAGS} ${INCLUDE} $< -o $@

${OBJ_DIR}/dedisperser.o: ${TRANSFORMS_DIR}/dedisperser.cpp
	${GXX} -c ${OPTIMISE} ${FLAGS} ${INCLUDE} ${DEDISP} $< -o $@

clean:
	rm -f ${OBJ_DIR}/*.o