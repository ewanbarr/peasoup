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

int main(void)
{
  //std::cout << "Generating a time series on device "<< tim.get_nsamps() << std::endl;
  //DeviceTimeSeries<float> d_tim(8388608);
  //d_tim.set_tsamp(0.000064);
  TimeSeries<float> tim;
  tim.from_file("/lustre/home/ebarr/Soft/peasoup/tmp5.tim");
  DeviceTimeSeries<float> d_tim(tim);
  
  unsigned int size = d_tim.get_nsamps();
  
  TimeSeriesFolder folder(size);
  
  //DeviceTimeSeries<float> d_tim_r(fft_size); //<----for resampled data
  //TimeDomainResampler resampler;
  

  float* folded_buffer;
  cudaError_t error;
  cufftResult result;
  error = cudaMalloc((void**)&folded_buffer, sizeof(float)*size);
  ErrorChecker::check_cuda_error(error);


  unsigned nints = 64;
  unsigned nbins = 32;

  cufftComplex* fft_out;
  error = cudaMalloc((void**)&fft_out, sizeof(cufftComplex)*nints*nbins);
  cufftHandle plan;
  result = cufftPlan1d(&plan,nbins,CUFFT_R2C, nints);
  ErrorChecker::check_cufft_error(result);
  Stopwatch timer;

  FoldedSubints<float> folded_array(nbins,nints);
  //folder.fold(d_tim,folded_array,0.007453079228);

  std::cout << "made it here" << std::endl;
  

  FoldOptimiser optimiser(nbins,nints);
  timer.start();
  for (int ii=0;ii<1;ii++){
    //FoldedSubints<float> folded_array(nbins,nints);
    folder.fold(d_tim,folded_array,0.007453099228);
    Utils::dump_device_buffer<float>(folded_array.get_data(),nints*nbins,"original_fold.bin");
    
    optimiser.optimise(folded_array);
  }
  timer.stop();
  
  /*
  float* temp = new float [nints*nbins];
  
  cudaMemcpy(temp,folded_buffer,nints*nbins*sizeof(float),cudaMemcpyDeviceToHost);
  ErrorChecker::check_cuda_error();

  for (int ii=0;ii<nints*nbins;ii++)
    std::cout << temp[ii] << std::endl;
  */

  
  std::cout << "Total execution time (s): " << timer.getTime()<<std::endl;
  std::cout << "Average execution time (s): " << timer.getTime()/1000.0 << std::endl;



  return 0;
}
