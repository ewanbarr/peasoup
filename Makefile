include Makefile.inc
# Output directories                                                                                                                        
BIN_DIR     = ./bin
LIB_DIR     = ./lib

# Dependencies
DEDISP_DIR = /lustre/home/ebarr/Soft/dedisp
TCLAP_DIR  = ./

# Paths
SRC_DIR  = ./src
INCLUDE_DIR = ./include
DATA_TYPES_DIR = ${SRC_DIR}/data_types
TRANSFORMS_DIR = ${SRC_DIR}/transforms

# Compiler flags
OPTIMISE = -O3
DEBUG    = -g 

INCLUDE   = -I$(INCLUDE_DIR) -I$(THRUST_DIR) -I$(TCLAP_DIR)/tclap/ -I${DEDISP_DIR}/include
LIBS = -L$(CUDA_DIR)/lib64 -lcuda -lcudart -L${DEDISP_DIR}/lib -ldedisp

CUFLAGS  = --compiler-options -Wall --machine 64 -arch=$(GPU_ARCH) -Xcompiler ${DEBUG}
FLAGS    = -fPIC -Wall ${OPTIMISE} ${DEBUG}



CPPOBJS = ${OBJ_DIR}/filterbank.o ${OBJ_DIR}/timeseries.o ${OBJ_DIR}/dedisperser.o

all: directories ${CPPOBJS} ${LIB_DIR}/libpeasoup.so

directories:
	@mkdir -p ${BIN_DIR}

clean:
	rm -f ${LIB_DIR}/*.so ${OBJ_DIR}/*.o