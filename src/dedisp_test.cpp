#include <data_types/filterbank.hpp>
#include <data_types/timeseries.hpp>
#include <data_types/fourierseries.hpp>
#include <transforms/dedisperser.hpp>
#include <transforms/ffter.hpp>
#include <transforms/resampler.hpp>
#include <transforms/spectrumformer.hpp>
#include <transforms/harmonicfolder.hpp>
#include <utils/exceptions.hpp>
#include <utils/utils.hpp>
#include <string>
#include <iostream>
#include <stdio.h>
#include "cuda.h"
#include "cufft.h"

int main(void)
{
  std::string filename("/lustre/projects/p002_swin/surveys/HTRU/medlat/2009-09-12-01:41:41/11/2009-09-12-01:41:41.fil");

  std::cout << "Creating filterbank object" << std::endl;
  SigprocFilterbank filobj(filename);

  std::cout << "Creating dedisperser object" << std::endl;
  Dedisperser dedisperser(filobj,2);
  
  std::cout << "Generating a DM list" << std::endl;
  dedisperser.generate_dm_list(0,10.0,0.4,1.1);
  
  std::cout << "Executing dedisperse" << std::endl;
  DispersionTrials<unsigned char> trials = dedisperser.dedisperse();
  
  unsigned int fft_size = Utils::prev_power_of_two(filobj.get_nsamps());
  std::cout << "Setting FFT size to " << fft_size << " points" << std::endl;

  std::cout << "Creating FFTer" << std::endl;
  CuFFTerR2C ffter(fft_size);

  std::cout << "Creating Fourier series on device" << std::endl;
  DeviceFourierSeries<cufftComplex> d_fseries(ffter.get_output_size(),
					      ffter.get_resolution(filobj.get_tsamp()));
  DedispersedTimeSeries<unsigned char> tim;

  std::cout << "Generating a time series on device" << std::endl;
  ReusableDeviceTimeSeries<float,unsigned char> d_tim(fft_size);
  DeviceTimeSeries<float> d_tim_r(fft_size); //<----for resampled data

  TimeDomainResampler resampler;
  SpectrumFormer spec_former;
  DevicePowerSpectrum<float> d_pspec(d_fseries);
  HarmonicFolder harm_folder;
  HarmonicSums<float> sums(d_pspec,4);
  
  //DevicePowerSpectrum<float> test = sums[3];


  for (int ii=0; ii < (int)trials.get_count(); ii++){    
    tim = trials[ii];
    
    std::cout << "Copying DM trial to device (" << tim.get_dm() << ")"<< std::endl;
    d_tim.copy_from_host(tim);

    std::cout << "Performing resampling" << std::endl;
    resampler.resample(d_tim,d_tim_r,12.0);
    
    std::cout << "Performing FFT" << std::endl;
    ffter.execute(d_tim_r.get_data(),d_fseries.get_data());
    
    std::cout << "Forming spectrum" << std::endl;
    spec_former.form_interpolated(d_fseries,d_pspec);
    
    std::cout << "Summing harmonics" << std::endl;
    harm_folder.fold(d_pspec,sums);
  }
  return 0;
}
