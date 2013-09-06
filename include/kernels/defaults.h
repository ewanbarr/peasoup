#pragma once
#include "cuda.h"
/*
cudaDeviceProp properties;
cudaGetDeviceProperties(&properties,0);
unsigned int MAX_THREADS = properties.maxThreadsPerBlock;
unsigned int MAX_BLOCKS = properties.maxGridSize[0];
*/
#define MAX_BLOCKS 65535
#define MAX_THREADS 512




