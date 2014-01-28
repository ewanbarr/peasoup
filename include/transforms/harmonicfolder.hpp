#pragma once
#include <data_types/fourierseries.hpp>
#include <kernels/kernels.h>
#include <kernels/defaults.h>
#include <iostream>

class HarmonicFolder {
private:
  unsigned int max_blocks;
  unsigned int max_threads;

public:
  HarmonicFolder(unsigned int max_blocks=MAX_BLOCKS,
		 unsigned int max_threads=MAX_THREADS)
    :max_blocks(max_blocks),max_threads(max_threads){}
  
  void fold4(DevicePowerSpectrum<float>& fold0, HarmonicSums<float>& sums)
  {
    float** h_data_ptrs;
    float** d_data_ptrs;
    
    Utils::device_malloc<float*>(&d_data_ptrs,sums.size());
    Utils::host_malloc<float*>(&h_data_ptrs,sums.size());
    
    for (int ii=0;ii<sums.size();ii++)
      {
	h_data_ptrs[ii] = sums[ii]->get_data();
      }
    Utils::h2dcpy<float*>(d_data_ptrs,h_data_ptrs,sums.size());
    device_harmonic_sum_II(fold0.get_data(),d_data_ptrs,
			   fold0.get_nbins(),sums.size(),
			   max_blocks,max_threads);
    Utils::device_free(d_data_ptrs);
    Utils::host_free(h_data_ptrs);
  }

  
  void fold(DevicePowerSpectrum<float>& fold0, HarmonicSums<float>& sums)
  {
    for (int ii=0;ii<sums.size();ii++)
      {
	device_harmonic_sum(fold0.get_data(),
			    sums[ii]->get_data(),
			    fold0.get_nbins(),
			    pow(2,ii+1),
			    max_blocks,
			    max_threads);
      }

  }
  
  void fold(DevicePowerSpectrum<float>& fold0, 
	    DevicePowerSpectrum<float>& output, unsigned int harm)
  {
    device_harmonic_sum(fold0.get_data(),
			output.get_data(),
			fold0.get_nbins(),
			harm,
			max_blocks,
			max_threads);
  }

  
  
};
