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
INCLUDE  = -I$(INCLUDE_DIR) -I$(THRUST_DIR) -I${DEDISP_DIR}/include -I${CUDA_DIR}/include -I./tclap
LIBS = -L$(CUDA_DIR)/lib64 -lcuda -lcudart -L${DEDISP_DIR}/lib -ldedisp -lcufft -lpthread

# compiler flags
# --compiler-options -Wall
NVCC_COMP_FLAGS = -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30
NVCCFLAGS  = ${OPTIMISE} ${NVCC_COMP_FLAGS} --machine 64 -Xcompiler ${DEBUG} 
CFLAGS    = -fPIC ${OPTIMISE} ${DEBUG}

OBJECTS   = ${OBJ_DIR}/kernels.o
EXE_FILES = ${BIN_DIR}/peasoup #${BIN_DIR}/resampling_test ${BIN_DIR}/harmonic_sum_test

all: directories ${OBJECTS} ${EXE_FILES}

${OBJ_DIR}/kernels.o: ${SRC_DIR}/kernels.cu
	${NVCC} -c ${NVCCFLAGS} ${INCLUDE} $<  -o $@

${BIN_DIR}/peasoup: ${SRC_DIR}/pipeline_multi.cpp ${OBJECTS}
	${NVCC} ${NVCCFLAGS} ${INCLUDE} ${LIBS} $^ -o $@

${BIN_DIR}/harmonic_sum_test: ${SRC_DIR}/harmonic_sum_test.cpp ${OBJECTS}
	${NVCC} ${NVCCFLAGS} ${INCLUDE} ${LIBS} $^ -o $@

${BIN_DIR}/resampling_test: ${SRC_DIR}/resampling_test.cpp ${OBJECTS}
	${NVCC} ${NVCCFLAGS} ${INCLUDE} ${LIBS} $^ -o $@

${BIN_DIR}/coincidencer: ${SRC_DIR}/coincidencer.cpp ${OBJECTS}
	${NVCC} ${NVCCFLAGS} ${INCLUDE} ${LIBS} $^ -o $@

${BIN_DIR}/accmap: ${SRC_DIR}/accmap.cpp ${OBJECTS}
	${NVCC} ${NVCCFLAGS} ${INCLUDE} ${LIBS} $^ -o $@

${BIN_DIR}/rednoise: ${SRC_DIR}/rednoise_test.cpp ${OBJECTS}
	${NVCC} ${NVCCFLAGS} ${INCLUDE} ${LIBS} $^ -o $@

${BIN_DIR}/hcfft: ${SRC_DIR}/hcfft.cpp ${OBJECTS}
	${NVCC} ${NVCCFLAGS} ${INCLUDE} ${LIBS} $^ -o $@

${BIN_DIR}/folder_test: ${SRC_DIR}/folder_test.cpp ${OBJECTS}
	${NVCC} ${NVCCFLAGS} ${INCLUDE} ${LIBS} $^ -o $@

${BIN_DIR}/dedisp_test: ${SRC_DIR}/dedisp_test.cpp ${OBJECTS}
	${NVCC} ${NVCCFLAGS} ${INCLUDE} ${LIBS} $^ -o $@ 

directories:
	@mkdir -p ${BIN_DIR}
	@mkdir -p ${OBJ_DIR}

clean:
	@rm -rf ${BIN_DIR}/*	
	@rm -rf ${OBJ_DIR}/*