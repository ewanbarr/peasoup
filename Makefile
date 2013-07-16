include Makefile.inc
# Output directories                                                                                                                        
BIN_DIR     = ./bin

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

INCLUDE   = -I$(INCLUDE_DIR) -I$(THRUST_DIR) -I$(TCLAP_DIR)/tclap/ -I${DEDISP_DIR}/include -I${CUDA_DIR}/include
LIBS = -L$(CUDA_DIR)/lib64 -lcuda -lcudart -L${DEDISP_DIR}/lib -ldedisp

NVCCFLAGS  = --compiler-options -Wall --machine 64 -arch=$(GPU_ARCH) -Xcompiler ${DEBUG}
FLAGS    = -fPIC -Wall ${OPTIMISE} ${DEBUG}

EXE_FILES = ${BIN_DIR}/dedisp_test


all: directories ${EXE_FILES}

${BIN_DIR}/dedisp_test: ${SRC_DIR}/dedisp_test.cpp
	${GXX} ${FLAGS} ${INCLUDE} ${LIBS} $< -o $@ 

directories:
	@mkdir -p ${BIN_DIR}

clean:
	@rm -rf ${BIN_DIR}/*