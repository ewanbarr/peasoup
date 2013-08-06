#pragma once
#include <cmath>
#include "cuda.h"
#include <string>
#include <utils/exceptions.hpp>
#include <fstream>

class Utils {
public:
  static unsigned int prev_power_of_two(unsigned int val){
    unsigned int n = 1;
    while (n*2<val){
      n*=2;
    }
    return n;
  }
  
  template <class T>
  static void device_malloc(T** ptr,unsigned int units){
    cudaError_t error;
    error = cudaMalloc((void**)ptr, sizeof(T)*units);
    ErrorChecker::check_cuda_error(error);
  }
  
  template <class T>
  static void host_malloc(T** ptr,unsigned int units){
    cudaError_t error;
    error = cudaMallocHost((void**)ptr, sizeof(T)*units);
    ErrorChecker::check_cuda_error(error);
  }

  template <class T>
  static void device_free(T* ptr){
    cudaError_t error = cudaFree(ptr);
    ErrorChecker::check_cuda_error(error);
  }
  
  template <class T>
  static void host_free(T* ptr){
    cudaError_t error = cudaFreeHost(ptr);
    ErrorChecker::check_cuda_error(error);
  }

  template <class T>
  static void h2dcpy(T* d_ptr, T* h_ptr, unsigned int units){
    cudaError_t error = cudaMemcpy(d_ptr,h_ptr,sizeof(T)*units,cudaMemcpyHostToDevice);
    ErrorChecker::check_cuda_error(error);
  }

  template <class T>
  static void d2hcpy(T* h_ptr, T* d_ptr, unsigned int units){
    cudaError_t error = cudaMemcpy(h_ptr,d_ptr,sizeof(T)*units,cudaMemcpyDeviceToHost);
    ErrorChecker::check_cuda_error(error);
  }

  template <class T>
  static void d2dcpy(T* d_ptr_dst, T* d_ptr_src, unsigned int units){
    cudaError_t error = cudaMemcpy(d_ptr_dst,d_ptr_src,sizeof(T)*units,cudaMemcpyDeviceToDevice);
    ErrorChecker::check_cuda_error(error);
  }

  template <class T>
  static void dump_device_buffer(T* buffer, unsigned int size, std::string filename){
    T* host_ptr;
    host_malloc<T>(&host_ptr,size);
    d2hcpy(host_ptr,buffer,size);
    std::ofstream infile;
    infile.open(filename.c_str(),std::ifstream::out | std::ifstream::binary);
    infile.write((char*)host_ptr ,size*sizeof(T));
    infile.close();
    host_free(host_ptr);
  }

  template <class T>
  static void dump_host_buffer(T* buffer, unsigned int size, std::string filename){
    std::ofstream infile;
    infile.open(filename.c_str(),std::ifstream::out | std::ifstream::binary);
    infile.write((char*)buffer ,size*sizeof(T));
    infile.close();
  }

};

class Block {
public:
  unsigned int blocks;
  unsigned int data_idx;
  
  Block(unsigned int blocks, unsigned int data_idx)
    :blocks(blocks),data_idx(data_idx){}
};

class BlockCalculator {
private:
  unsigned int gulps;
  unsigned int gulp_counter;
  unsigned int gulp_index;
  unsigned int gulp_size;
  unsigned int blocks;
  std::vector< Block > output;

public:
  BlockCalculator(unsigned int size,
                  unsigned int max_blocks,
                  unsigned int max_threads)
  {
    gulps = size/(max_blocks*max_threads)+1;
    for (gulp_counter = 0; gulp_counter<gulps; gulp_counter++){
      if (gulp_counter<gulps-1){
        gulp_size = max_blocks*max_threads;
        blocks = max_blocks;
      }
      else {
        gulp_size = size-gulp_counter*max_blocks*max_threads;
        blocks = (gulp_size-1)/max_threads+1;
      }
      gulp_index = gulp_counter*blocks*max_threads;
      output.push_back(Block(blocks,gulp_index));
    }
  }

  unsigned int size(void){
    return  output.size();
  }

  Block& operator[](int idx){
    return output[idx];
  }
};
