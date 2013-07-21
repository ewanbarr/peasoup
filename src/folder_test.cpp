#include <data_types/timeseries.hpp>
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

int main(void)
{
  unsigned int size = 8388608;
  std::cout << "Generating a time series on device" << std::endl;
  DeviceTimeSeries<float> d_tim(size);
  d_tim.set_tsamp(0.000064);
  
  TimeSeriesFolder folder(size);
  
  //DeviceTimeSeries<float> d_tim_r(fft_size); //<----for resampled data
  //TimeDomainResampler resampler;
  
  float* folded_buffer;
  cudaError_t error = cudaMalloc((void**)&folded_buffer, sizeof(float)*size);
  ErrorChecker::check_cuda_error(error);

  
  Stopwatch timer;
  
  timer.start();
  for (int ii=0;ii<1000;ii++){
    folder.fold(d_tim,folded_buffer,0.002432,16,16*16);
  }
  timer.stop();
  ErrorChecker::check_cuda_error();
  std::cout << "Total execution time (s): " << timer.getTime()<<std::endl;
  std::cout << "Average execution time (s): " << timer.getTime()/1000.0 << std::endl;



  return 0;
}
