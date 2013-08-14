#pragma once
#include "transforms/ffter.hpp"
#include "data_types/fourierseries.hpp"
#include "kernels/kernels.h"
#include "kernels/defaults.h"
#include "utils/utils.hpp"
#include "utils/exceptions.hpp"
#include <iostream>

class Dereddener {
private:
  unsigned int size;
  float boundary_5_freq;
  float boundary_25_freq;
  float* median_5;
  float* median_25;
  float* median_125;
  float* median;
  float* intermediate;
  
public:
  Dereddener(unsigned int size)
    :size(size)
  {
    Utils::device_malloc(&intermediate,size);
    Utils::device_malloc(&median,size);
    Utils::device_malloc(&median_5,size/5);
    Utils::device_malloc(&median_25,size/5/5);
    Utils::device_malloc(&median_125,size/5/5/5);
  }
  
  ~Dereddener()
  {
    Utils::device_free(median);
    Utils::device_free(median_5);
    Utils::device_free(median_25);
    Utils::device_free(median_125);
    Utils::device_free(intermediate);
  }

  void calculate_median(DevicePowerSpectrum<float>& powers, 
			float boundary_5_freq=0.05,
			float boundary_25_freq=0.5)
  {
    if (powers.get_nbins()!=size)
      ErrorChecker::throw_error("Bad data length given to running_median()");
  
    int pos5  = (int) (boundary_5_freq/powers.get_bin_width());
    int pos25 = (int) (boundary_25_freq/powers.get_bin_width());
    median_scrunch5(powers.get_data(),size,median_5);
    median_scrunch5(median_5,size/5,median_25);
    median_scrunch5(median_25,size/5/5,median_125);
    
    linear_stretch(median_5,size/5,intermediate,size);
    Utils::d2dcpy(median,intermediate,pos5);
    
    linear_stretch(median_25,size/5/5,intermediate,size);
    Utils::d2dcpy(median+pos5,intermediate+pos5,pos25);
    
    linear_stretch(median_125,size/5/5/5,intermediate,size);
    Utils::d2dcpy(median+pos25,intermediate+pos25,size-pos25);
  }
  
  void deredden(DeviceFourierSeries<cufftComplex>& spectrum){
    device_divide_c_by_f(spectrum.get_data(),median,spectrum.get_nbins(),MAX_BLOCKS,MAX_THREADS);
  }
  
};
