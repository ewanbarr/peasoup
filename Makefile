include Makefile.inc

# Output directories
BIN_DIR     = ./bin
OBJ_DIR     = ./obj

# Paths
SRC_DIR  = ./src
INCLUDE_DIR = ./include

# Compiler flags
OPTIMISE = -O3
DEBUG    =

# Includes and libraries
INCLUDE  = -I$(INCLUDE_DIR) -I$(THRUST_DIR) -I${DEDISP_DIR}/include -I${CUDA_DIR}/include/nvtx3 -I./tclap
LIBS = -L$(CUDA_DIR)/lib64 -lcudart -L${DEDISP_DIR}/lib -ldedisp -lcufft -lpthread -lnvToolsExt

FFASTER_DIR = /mnt/home/ebarr/Soft/FFAster
FFASTER_INCLUDES = -I${FFASTER_DIR}/include -L${FFASTER_DIR}/lib -lffaster

# Compiler flags (Uses dynamically set GPU_ARCH_FLAG)
NVCCFLAGS  = ${UCFLAGS} ${OPTIMISE} ${GPU_ARCH_FLAG} -lineinfo --machine 64
NVCCFLAGS_FFA  = ${UCFLAGS} ${OPTIMISE} -lineinfo --machine 64 -Xcompiler ${DEBUG}

CFLAGS    = ${UCFLAGS} -fPIC ${OPTIMISE} ${DEBUG}

OBJECTS   = ${OBJ_DIR}/kernels.o
EXE_FILES = ${BIN_DIR}/peasoup

all: directories ${OBJECTS} ${EXE_FILES}

${OBJ_DIR}/kernels.o: ${SRC_DIR}/kernels.cu
	${NVCC} -c ${NVCCFLAGS} ${INCLUDE} $<  -o $@

${BIN_DIR}/peasoup: ${SRC_DIR}/pipeline_multi.cu ${OBJECTS}
	${NVCC} ${NVCCFLAGS} ${INCLUDE} ${LIBS} $^ -o $@

${BIN_DIR}/ffaster: ${SRC_DIR}/ffa_pipeline.cu ${OBJECTS}
	${NVCC} ${NVCCFLAGS_FFA} ${INCLUDE} ${FFASTER_INCLUDES} ${LIBS} $^ -o $@

directories:
	@mkdir -p ${BIN_DIR}
	@mkdir -p ${OBJ_DIR}

clean:
	@rm -rf ${OBJ_DIR}/*

install: all
	cp $(BIN_DIR)/peasoup $(INSTALL_DIR)/bin

