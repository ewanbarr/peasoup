#pragma once
#include <cmath>
#include "cuda.h"
#include <string>
#include <utils/exceptions.hpp>
#include <fstream>
#include <vector>
#include <iostream>

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
    cudaMalloc((void**)ptr, sizeof(T)*units);
    ErrorChecker::check_cuda_error("Error from device_malloc");
  }
  
  template <class T>
  static void host_malloc(T** ptr,unsigned int units){
    cudaMallocHost((void**)ptr, sizeof(T)*units);
    ErrorChecker::check_cuda_error("Error from host_malloc");
  }

  template <class T>
  static void device_free(T* ptr){
    cudaFree(ptr);
    ErrorChecker::check_cuda_error("Error from device_free");
  }
  
  template <class T>
  static void host_free(T* ptr){
    cudaFreeHost((void*) ptr);
    ErrorChecker::check_cuda_error("Error from host_free.");
  }

  template <class T>
  static void h2dcpy(T* d_ptr, T* h_ptr, unsigned int units){
    cudaMemcpy((void*)d_ptr, (void*) h_ptr, sizeof(T)*units, cudaMemcpyHostToDevice);
    ErrorChecker::check_cuda_error("Error from h2dcpy");
  }

  template <class T>
  static void d2hcpy(T* h_ptr, T* d_ptr, unsigned int units){
    cudaMemcpy((void*) h_ptr,(void*) d_ptr,sizeof(T)*units,cudaMemcpyDeviceToHost);
    ErrorChecker::check_cuda_error("Error from d2hcpy");
  }

  template <class T>
  static void d2dcpy(T* d_ptr_dst, T* d_ptr_src, unsigned int units){
    cudaMemcpy(d_ptr_dst,d_ptr_src,sizeof(T)*units,cudaMemcpyDeviceToDevice);
    ErrorChecker::check_cuda_error("Error from d2dcpy");
  }

  template <class T>
  static void dump_device_buffer(T* buffer, size_t size, std::string filename){
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
  static void dump_host_buffer(T* buffer, size_t size, std::string filename){
    std::ofstream infile;
    infile.open(filename.c_str(),std::ifstream::out | std::ifstream::binary);
    infile.write((char*)buffer ,size*sizeof(T));
    infile.close();
  }

  static int gpu_count(){
    int count;
    cudaGetDeviceCount(&count);
    return count;
  }

};

class Block {
public:
  unsigned int blocks;
  size_t data_idx;
  size_t gulp_size;
  
  Block(unsigned int blocks, size_t data_idx, size_t gulp_size)
    :blocks(blocks),data_idx(data_idx),gulp_size(gulp_size){}
};

class BlockCalculator {
private:
  size_t gulps;
  size_t gulp_counter;
  size_t gulp_index;
  size_t gulp_size;
  size_t blocks;
  std::vector< Block > output;

public:
  BlockCalculator(size_t size,
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
      gulp_index = gulp_counter*max_blocks*max_threads;
      output.push_back(Block(blocks,gulp_index,gulp_size));
      
      
    }
  }

  unsigned int size(void){
    return  output.size();
  }

  Block& operator[](int idx){
    return output[idx];
  }
};

class AccelerationPlan {
private:
  float acc_lo;
  float acc_hi;
  float tol;
  float pulse_width;
  unsigned int nsamps;
  float tsamp;
  float cfreq;
  float cfreq_GHz;
  float bw;
  float tsamp_us;
  float tobs;

public:
  AccelerationPlan(float acc_lo, float acc_hi, float tol,
		   float pulse_width, unsigned int nsamps,
		   float tsamp, float cfreq, float bw)
    :acc_lo(acc_lo),acc_hi(acc_hi),tol(tol),
     pulse_width(pulse_width),nsamps(nsamps),
     tsamp(tsamp),cfreq(cfreq),bw(fabs(bw))
  {
    tsamp_us = 1.0e6 * tsamp;
    tobs = nsamps*tsamp;
    cfreq_GHz = 1.0e-3 * cfreq;
    pulse_width /= 1.0e3;
  }
  
  void generate_accel_list(float dm,std::vector<float>& acc_list){
    if (acc_hi==acc_lo){
      acc_list.clear();
      acc_list.push_back(0.0);
      return;
    }

    float tdm = pow(8.3*bw/pow(cfreq,3.0)*dm,2.0);
    float tpulse = pulse_width * pulse_width;
    float ttsamp = tsamp * tsamp;
    float w_us = sqrt(tdm+tpulse+ttsamp);
    float alt_a = 2.0 * w_us * 1.0e-6 * 24.0 * 299792458.0/tobs/tobs * sqrt((tol*tol)-1.0);
    unsigned int naccels = (unsigned int)((float)(acc_hi-acc_lo))/alt_a;
    acc_list.clear();
    acc_list.reserve(naccels+3);
    if (acc_hi!=0 && acc_lo!=0)
      acc_list.push_back(0.0); //explicitly force zero acceleration.
    float acc = acc_lo;
    while (acc<acc_hi){
      acc_list.push_back(acc);
      acc+=alt_a;
    }
    acc_list.push_back(acc_hi);
    return;
  }
};


