#include <data_types/timeseries.hpp>
#include <data_types/fourierseries.hpp>
#include <data_types/candidates.hpp>
#include <transforms/resampler.hpp>
#include <transforms/folder.hpp>
#include <transforms/ffter.hpp>
#include <transforms/dereddener.hpp>
#include <transforms/spectrumformer.hpp>
#include <transforms/birdiezapper.hpp>
#include <transforms/peakfinder.hpp>
#include <transforms/distiller.hpp>
#include <transforms/harmonicfolder.hpp>
#include <utils/exceptions.hpp>
#include <utils/utils.hpp>
#include <utils/stats.hpp>
#include <utils/stopwatch.hpp>
#include <string>
#include <iostream>
#include <stdio.h>
#include "cuda.h"
#include "cufft.h"
#include <cmath>

int main(void)
{
  //std::cout << "Generating a time series on device "<< tim.get_nsamps() << std::endl;
  //DeviceTimeSeries<float> d_tim(8388608);
  //d_tim.set_tsamp(0.000064);
  

  TimeSeries<float> tim;
  tim.from_file("/lustre/home/ebarr/Soft/peasoup/dblpsr.tim");
  //tim.from_file("/lustre/home/ebarr/Soft/peasoup/../test/tmp.tim");
  DeviceTimeSeries<float> d_tim(tim);
  DeviceTimeSeries<float> d_tim_r(tim);
  //unsigned int size = 187503;
  unsigned int size = pow(2,23);
  float tobs = size*d_tim.get_tsamp();
  float bin_width = 1.0/tobs;
  TimeDomainResampler resampler;
  
  //resampler.resample(d_tim,d_tim_r,size,200);
  //Utils::dump_device_buffer<float>(d_tim.get_data(),size,"tim.bin");                                                                                                          
  //Utils::dump_device_buffer<float>(d_tim_r.get_data(),size,"tim_r.bin");   
  
  DeviceFourierSeries<cufftComplex> fseries(size/2+1,bin_width);
  DevicePowerSpectrum<float> pspec(fseries);
  Zapper bzap("misc/default_zaplist.txt");
  Dereddener rednoise(size/2+1);
  SpectrumFormer former;
  CuFFTerR2C r2cfft(size);
  CuFFTerC2R c2rfft(size);
  PeakFinder cand_finder(6.0,0.1,1000.0);
  HarmonicFolder harm_folder;
  HarmonicSums<float> sums(pspec,4);
  HarmonicDistiller harm_finder(0.0001,16,false);

  r2cfft.execute(d_tim.get_data(),fseries.get_data());
  //Utils::dump_device_buffer<cufftComplex>(fseries.get_data(),size/2+1,"fseries_pre.bin");

  
  former.form(fseries,pspec);
  rednoise.calculate_median(pspec);
  rednoise.deredden(fseries);
  bzap.zap(fseries);
  former.form_interpolated(fseries,pspec);
  float mean,std,rms;
  stats::stats<float>(pspec.get_data(),size/2+1,&mean,&rms,&std);
  

  //Utils::dump_device_buffer(d_tim.get_data(),size,"tim1.bin");
  
  c2rfft.execute(fseries.get_data(),d_tim.get_data());
  //Utils::dump_device_buffer(d_tim.get_data(),size,"tim2.bin");
  ErrorChecker::check_cuda_error();

  CandidateCollection accel_trial_cands;

  //for (float ii=-20.0;ii<20.0;ii+=0.5){
  float ii=222.51;
    //std::cout << "---------   "<<ii<<"    -----------" << std::endl;
  resampler.resample(d_tim,d_tim_r,size,ii);
  //Utils::dump_device_buffer<float>(d_tim.get_data(),size,"tim.bin");
  Utils::dump_device_buffer<float>(d_tim_r.get_data(),size,"tim_r.bin");
    
  r2cfft.execute(d_tim_r.get_data(),fseries.get_data());
  //Utils::dump_device_buffer<cufftComplex>(fseries.get_data(),size/2+1,"fseries_re.bin");
  
  former.form(fseries,pspec);
  Utils::dump_device_buffer<float>(pspec.get_data(),size/2+1,"non_interp_spec.bin");
  former.form_interpolated(fseries,pspec);
  Utils::dump_device_buffer<float>(pspec.get_data(),size/2+1,"interp_spec.bin");
  stats::normalise(pspec.get_data(),mean*size,std*size,size/2+1);
  Utils::dump_device_buffer<float>(pspec.get_data(),size/2+1,"pspec_post.bin");
  
  harm_folder.fold(pspec,sums);
  
  Utils::dump_device_buffer<float>(pspec.get_data(),size/2+1,"harm0.bin");
  Utils::dump_device_buffer<float>(sums[0]->get_data(),size/2+1,"harm1.bin");
  Utils::dump_device_buffer<float>(sums[1]->get_data(),size/2+1,"harm2.bin");
  Utils::dump_device_buffer<float>(sums[2]->get_data(),size/2+1,"harm3.bin");
  Utils::dump_device_buffer<float>(sums[3]->get_data(),size/2+1,"harm4.bin");
  
  
  SpectrumCandidates trial_cands(24.7,0,ii);
  cand_finder.find_candidates(pspec,trial_cands);
  cand_finder.find_candidates(sums,trial_cands);
  trial_cands.print();
  //accel_trial_cands.append(harm_finder.distill(trial_cands.cands));
  //}
  //accel_trial_cands.print();
  //AccelerationDistiller acc_still(tobs,0.0001);
  //accel_trial_cands.cands = acc_still.distill(accel_trial_cands.cands);
  //accel_trial_cands.cands = harm_finder.distill(accel_trial_cands.cands);
  //accel_trial_cands.print();
  return 0;
}
