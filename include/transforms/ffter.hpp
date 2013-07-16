#pragma once
#include "cuda.h"
#include "cufft.h"
#include "data_types/timeseries.hpp"
#include "utils/exceptions.hpp"

template <class T>
class FFTer {
private:
  cufftHandle fft_plan;
  
public:
  void generate_r2c_plan(size_t size, size_t batch=1)
  {
    cufftResult error = cufftPlan1d(&fft_plan, size, CUFFT_R2C, 1);
    ErrorChecker::check_cufft_error(error);
  }
  
  void execute_r2c(DeviceTimeSeries& tim){
    cufftComplex* temp;
    cufftResult error = cufftExecR2C(fft_plan,
					 (cufftReal*) tim.get_data(),
					 (cufftComplex*) temp);
    ErrorChecker::check_cufft_error(error)
  }
};
