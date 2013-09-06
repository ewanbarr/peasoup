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
  unsigned int size = 8388608;
  int nloop=1000;
  cufftComplex* spec;
  Stopwatch timer;
  
  cufftHandle forward_plan,reverse_plan;
  cufftPlan1d(&forward_plan,size,CUFFT_R2C,1);
  cufftPlan1d(&reverse_plan,size,CUFFT_C2R,1);
  Utils::host_malloc<float>(&tim,size);
  Utils::device_malloc<float>(&d_tim,size);
  Utils::host_malloc<float>(&result,size);
  Utils::device_malloc<float>(&d_result,size);
  Utils::device_malloc<cufftComplex>(&spec,size/2+1);
  Utils::h2dcpy<float>(d_tim,tim,size);
  timer.start();
  for (int ii=0;ii<nloop;ii++){
    cufftExecR2C(forward_plan,d_tim,spec);
    cufftExecC2R(reverse_plan,spec,d_result);
  }
  timer.stop();
  std::cout << (float)timer.getTime()/nloop << std::endl;
  Utils::d2hcpy<float>(result,d_result,size);
}
