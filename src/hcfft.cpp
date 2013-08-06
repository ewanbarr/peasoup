#include <data_types/timeseries.hpp>
#include <data_types/folded.hpp>
#include <transforms/resampler.hpp>
#include <transforms/folder.hpp>
#include <utils/exceptions.hpp>
#include <utils/utils.hpp>
#include <utils/stopwatch.hpp>
#include <string>
#include <iostream>
#include <stdio.h>
#include "cuda.h"
#include "cufft.h"

int main(){
  
  float* tim;
  float* result;
  float* d_tim;
  float* d_result;
  cufftComplex* spec;
  
  cufftHandle forward_plan,reverse_plan;
  cufftPlan1d(&forward_plan,10,CUFFT_R2C,1);
  cufftPlan1d(&reverse_plan,10,CUFFT_C2R,1);

  Utils::host_malloc<float>(&tim,10);
  Utils::device_malloc<float>(&d_tim,10);
  Utils::host_malloc<float>(&result,10);
  Utils::device_malloc<float>(&d_result,10);
  Utils::device_malloc<cufftComplex>(&spec,6);

  for (int ii =0; ii<10; ii++)
    tim[ii] = ii;
    
  Utils::h2dcpy<float>(d_tim,tim,10);
  
  cufftExecR2C(forward_plan,d_tim,spec);
  
  cufftExecC2R(reverse_plan,spec,d_result);
  
  Utils::d2hcpy<float>(result,d_result,10);

  for (int ii =0; ii<10; ii++)
    std::cout << result[ii]/10. << std::endl;

  
}
