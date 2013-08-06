#include <data_types/timeseries.hpp>
#include <data_types/fourierseries.hpp>
#include <transforms/resampler.hpp>
#include <transforms/folder.hpp>
#include <transforms/ffter.hpp>
#include <transforms/dereddener.hpp>
#include <transforms/spectrumformer.hpp>
#include <transforms/birdiezapper.hpp>
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
  tim.from_file("/lustre/home/ebarr/Soft/peasoup/zerodm.tim");

  DeviceTimeSeries<float> d_tim(tim);
  
  unsigned int size = 8388608;
   
  float bin_width = 1.0/(size*d_tim.get_tsamp());


  DeviceFourierSeries<cufftComplex> fseries(size/2+1,bin_width);
  DevicePowerSpectrum<float> pspec(fseries);

  Zapper bzap("default_zaplist.txt");
  

  Dereddener rednoise(size/2+1);

  SpectrumFormer former;
  
  CuFFTerR2C r2cfft(size);
  CuFFTerC2R c2rfft(size);

  r2cfft.execute(d_tim.get_data(),fseries.get_data());

  former.form(fseries,pspec);
  rednoise.calculate_median(pspec);
  rednoise.deredden(fseries);

  Utils::dump_device_buffer<cufftComplex>(fseries.get_data(),size/2+1,"fseries_pre.bin");
  bzap.zap(fseries);
  Utils::dump_device_buffer<cufftComplex>(fseries.get_data(),size/2+1,"fseries_post.bin");

  Utils::dump_device_buffer(d_tim.get_data(),size,"tim1.bin");
  c2rfft.execute(fseries.get_data(),d_tim.get_data());
  
  Utils::dump_device_buffer(d_tim.get_data(),size,"tim2.bin");
  ErrorChecker::check_cuda_error();

  return 0;
}
