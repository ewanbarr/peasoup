#include <transforms/resampler.hpp>
#include <transforms/spectrumformer.hpp>
#include <data_types/timeseries.hpp>
#include <utils/exceptions.hpp>
#include <utils/utils.hpp>
#include <utils/stopwatch.hpp>
#include <string>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include "cuda.h"
#include "cufft.h"

#define NBINS 4194304
#define TSAMP 0.000064
#define ACC 125.5
int main()
{
  float* test_pattern;
  Utils::host_malloc<float>(&test_pattern,NBINS);
  for (int ii=0;ii<NBINS;ii++)
    {
      test_pattern[ii] = ii%451;
    }
  
  DeviceTimeSeries<float> d_tim(NBINS);
  Utils::h2dcpy<float>(d_tim.get_data(),test_pattern,NBINS);
  d_tim.set_tsamp(TSAMP);
  
  DeviceTimeSeries<float> d_tim_r0(NBINS);
  d_tim_r0.set_tsamp(TSAMP);

  DeviceTimeSeries<float> d_tim_r1(NBINS);
  d_tim_r1.set_tsamp(TSAMP);

  //SpectrumFormer specformer


  TimeDomainResampler resampler;
  
  resampler.resample(d_tim,d_tim_r0,NBINS,ACC);
  
  resampler.resampleII(d_tim,d_tim_r1,NBINS,ACC);
  
  
  float* test_block0;
  Utils::host_malloc<float>(&test_block0,NBINS);
      
  float* test_block1;
  Utils::host_malloc<float>(&test_block1,NBINS);

  Utils::d2hcpy(test_block0,d_tim_r0.get_data(),NBINS);
  Utils::d2hcpy(test_block1,d_tim_r1.get_data(),NBINS);

  for (int ii=0;ii<NBINS;ii++){
    if (fabs(test_block0[ii]-test_block1[ii])>0.0001)
      printf("[WRONG (%d)] %f != %f\n",ii,test_block0[ii],test_block1[ii]);
  }

  Utils::host_free(test_pattern);
  Utils::host_free(test_block0);
  Utils::host_free(test_block1);
  return 0;
}

