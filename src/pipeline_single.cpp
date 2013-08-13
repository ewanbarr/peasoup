#include <data_types/timeseries.hpp>
#include <data_types/fourierseries.hpp>
#include <data_types/candidates.hpp>
#include <data_types/filterbank.hpp>
#include <transforms/dedisperser.hpp>
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

  std::string filename("/lustre/projects/p002_swin/cflynn/products/linear/globular/2013-07-14-21:48:07/01/masked.fil");
  
  std::cout << "Creating filterbank object" << std::endl;
  SigprocFilterbank filobj(filename);

  std::cout << "Creating dedisperser object" << std::endl;
  Dedisperser dedisperser(filobj,2);

  std::cout << "Generating a DM list" << std::endl;
  dedisperser.generate_dm_list(24.0,25.0,0.4,1.05);

  std::cout << "Executing dedisperse" << std::endl;
  DispersionTrials<unsigned char> trials = dedisperser.dedisperse();
  
  unsigned int size = Utils::prev_power_of_two(filobj.get_nsamps());
  std::cout << "Setting FFT size to " << size << " points" << std::endl;

  std::cout << "Creating FFT instances" << std::endl;
  CuFFTerR2C r2cfft(size);
  CuFFTerC2R c2rfft(size);
  
  float tobs = size*filobj.get_tsamp();
  float bin_width = 1.0/tobs;  

  std::cout << "Creating Fourier series on device" << std::endl;
  DeviceFourierSeries<cufftComplex> d_fseries(size/2+1,bin_width);
  DedispersedTimeSeries<unsigned char> tim;

  std::cout << "Generating time series on device" << std::endl;
  ReusableDeviceTimeSeries<float,unsigned char> d_tim(size);
  DeviceTimeSeries<float> d_tim_r(size);

  TimeDomainResampler resampler;
  DevicePowerSpectrum<float> pspec(d_fseries);
  Zapper bzap("default_zaplist.txt");
  Dereddener rednoise(size/2+1);
  SpectrumFormer former;
  PeakFinder cand_finder(7.0,0.1,1000.0);
  HarmonicFolder harm_folder;
  HarmonicSums<float> sums(pspec,4);
  HarmonicDistiller harm_finder(0.0001,16);
  AccelerationDistiller acc_still(tobs,0.0001);
  DMDistiller dm_still(0.0001);
  float mean,std,rms;
  CandidateCollection dm_trial_cands;
  for (int ii=0; ii < (int)trials.get_count(); ii++){
    tim = trials[ii];
    std::cout << "Copying DM trial to device (" << tim.get_dm() << ")"<< std::endl;
    d_tim.copy_from_host(tim);

    r2cfft.execute(d_tim.get_data(),d_fseries.get_data());
    former.form(d_fseries,pspec);
    rednoise.calculate_median(pspec);
    rednoise.deredden(d_fseries);
    bzap.zap(d_fseries);
    former.form_interpolated(d_fseries,pspec);
    stats::stats<float>(pspec.get_data(),size/2+1,&mean,&rms,&std);
    c2rfft.execute(d_fseries.get_data(),d_tim.get_data());
    
    CandidateCollection accel_trial_cands;
    
    for (float jj=-2;jj<2;jj++){
      
      resampler.resample(d_tim,d_tim_r,size,jj);
      r2cfft.execute(d_tim_r.get_data(),d_fseries.get_data());
      former.form_interpolated(d_fseries,pspec);
      stats::normalise(pspec.get_data(),mean*size,std*size,size/2+1);
      harm_folder.fold(pspec,sums);
  
      SpectrumCandidates trial_cands(tim.get_dm(),ii,jj);
      cand_finder.find_candidates(pspec,trial_cands);
      cand_finder.find_candidates(sums,trial_cands);
      accel_trial_cands.append(harm_finder.distill(trial_cands.cands));
    }
    dm_trial_cands.append(acc_still.distill(accel_trial_cands.cands));
  }
  dm_trial_cands.cands = dm_still.distill(dm_trial_cands.cands);
  dm_trial_cands.print();
return 0;
}
