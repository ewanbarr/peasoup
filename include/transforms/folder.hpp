#include <kernels/defaults.h>
#include <kernels/kernels.h>
#include <utils/exceptions.hpp>
#include "cuda.h"
#include <iostream>

class TimeSeriesFolder {
private:
  unsigned int size;
  unsigned int max_blocks;
  unsigned int max_threads;
  float* rebin_buffer;
    
public:
  TimeSeriesFolder(unsigned int size,
		   unsigned int max_blocks=MAX_BLOCKS,
		   unsigned int max_threads=MAX_THREADS)
    :size(size),max_blocks(max_blocks),max_threads(max_threads)
  {
    //Intentionally allocate much bigger buffers than needed
    //Can be refined if problematic
    cudaError_t error = cudaMalloc((void**)&rebin_buffer, sizeof(float)*size);
    ErrorChecker::check_cuda_error(error);
  }

  //TEMPORARY UNTIL FOLDED DATA CLASS IS WRITTEN

  void fold(DeviceTimeSeries<float>& input, float* output, float period, 
	    unsigned int nbins, unsigned int nints)
  {
    float tobs = input.get_nsamps() * input.get_tsamp();
    unsigned int rebinned_size = nbins*tobs/period;
    unsigned int nrots = rebinned_size/nbins;
    unsigned int nrots_per_subint = nrots/nints;
    device_rebin_time_series(input.get_data(),rebin_buffer,
			     period, input.get_tsamp(), 
			     input.get_nsamps(),rebinned_size,
			     nbins,max_blocks,max_threads);
    
    device_create_subints(rebin_buffer, output, nbins,
			  nbins*nints, nrots_per_subint, 
			  max_blocks,max_threads);
    
    ErrorChecker::check_cuda_error(cudaDeviceSynchronize());
  }
};

