#pragma once
#include <kernels/kernels.h>
#include <kernels/defaults.h>
#include <cmath>

namespace stats {

  template <class T>
  float rms(T* ptr,unsigned int nsamps,unsigned int first_samp=0)
  {
    return GPU_rms<T>(ptr,nsamps,first_samp);
  }

  template <class T>
  float mean(T* ptr,unsigned int nsamps,unsigned int first_samp=0)
  {
    return GPU_mean<T>(ptr,nsamps,first_samp);
  }

  float std(float mean, float rms)
  {
    return sqrt(rms*rms-mean*mean);
  }
  
  template <class T>
  void stats(T* ptr, unsigned int nsamps, float* mean_, float* rms_,
	     float* std_, unsigned int first_samp=0)
  {
    *rms_  = rms<T>(ptr,nsamps,first_samp);
    *mean_ = mean<T>(ptr,nsamps,first_samp);
    *std_  = std(*mean_,*rms_);
    return;
  } 

  void normalise(float* ptr, float mean, float std, 
		 unsigned int size, unsigned int max_blocks=MAX_BLOCKS,
		 unsigned int max_threads=MAX_THREADS){
    device_normalise(ptr, mean, std, size, max_blocks, max_threads);
    return;
  }

  
}
