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
#include <transforms/coincidencer.hpp>
#include <utils/exceptions.hpp>
#include <utils/utils.hpp>
#include <utils/stats.hpp>
#include <utils/stopwatch.hpp>
#include <tclap/CmdLine.h>
#include <string>
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include "cuda.h"
#include "cufft.h"
#include <cmath>
#include <sstream>

struct CmdLineOptions {
  std::string samp_outfilename;
  std::string spec_outfilename;
  std::vector<std::string> multi;
  float boundary_5_freq;
  float boundary_25_freq;
  int nharmonics;
  float threshold;
  int beam_threshold;
  float min_freq;
  float max_freq;
  int max_harm;
  float freq_tol;
  bool verbose;
};

int main(int argc, char **argv)
{
  CmdLineOptions args;
  try
    {
      TCLAP::CmdLine cmd("Peasoup - a GPU pulsar search pipeline", ' ', "1.0");

      TCLAP::UnlabeledMultiArg<std::string> arg_multi("filterbanks","File names",
						      true, "string", cmd);

      TCLAP::ValueArg<std::string> arg_samp_outfilename("", "o",
							"Sample mask output filename",
							false, "rfi.eb_mask", "string", cmd);

      TCLAP::ValueArg<std::string> arg_spec_outfilename("", "o2",
                                                        "Birdie list output filename",
                                                        false, "birdies.txt", "string", cmd);

      TCLAP::ValueArg<float> arg_boundary_5_freq("l", "boundary_5_freq",
						 "Frequency at which to switch from median5 to median25",
						 false, 0.05, "float", cmd);
      
      TCLAP::ValueArg<float> arg_boundary_25_freq("a", "boundary_25_freq",
						 "Frequency at which to switch from median25 to median125",
                                                 false, 0.5, "float", cmd);
      
      TCLAP::ValueArg<int> arg_nharmonics("n", "nharmonics",
					  "Number of harmonic sums to perform",
					  false, 4, "int", cmd);

      TCLAP::ValueArg<float> arg_threshold("", "thresh", 
					 "The S/N threshold for a candidate to be considered for coincidencing matching",
					 false, 4.0, "float", cmd);
      
      TCLAP::ValueArg<int> arg_beam_threshold("", "beam_thresh",
						"The number of beams a candidate must appear in to be considered multibeam",
						false, 4, "int", cmd);

      TCLAP::ValueArg<float> arg_min_freq("L", "min_freq",
					  "Lowest Fourier freqency to consider",
					  false, 0.1, "float",cmd);
      
      TCLAP::ValueArg<float> arg_max_freq("H", "max_freq",
                                          "Highest Fourier freqency to consider",
                                          false, 1100.0, "float",cmd);

      TCLAP::ValueArg<int> arg_max_harm("b", "max_harm",
					"Maximum harmonic for related candidates",
                                          false, 16, "float",cmd);
      
      TCLAP::ValueArg<float> arg_freq_tol("f", "freq_tol",
                                          "Tolerance for distilling frequencies (0.0001 = 0.01%)",
                                          false, 0.0001, "float",cmd);

      TCLAP::SwitchArg arg_verbose("v", "verbose", "verbose mode", cmd);
      
      cmd.parse(argc, argv);
      args.multi             = arg_multi.getValue();
      args.samp_outfilename  = arg_samp_outfilename.getValue();
      args.spec_outfilename  = arg_spec_outfilename.getValue();
      args.boundary_5_freq   = arg_boundary_5_freq.getValue();   
      args.boundary_25_freq  = arg_boundary_25_freq.getValue();
      args.nharmonics        = arg_nharmonics.getValue();
      args.threshold         = arg_threshold.getValue();
      args.beam_threshold    = arg_beam_threshold.getValue();
      args.min_freq          = arg_min_freq.getValue();
      args.max_freq          = arg_max_freq.getValue();
      args.freq_tol          = arg_freq_tol.getValue();
      args.verbose           = arg_verbose.getValue();
      
    }catch (TCLAP::ArgException &e) {
    std::cerr << "Error: " << e.error() << " for arg " << e.argId()
	      << std::endl;
    return -1;
  }
  
  int ii;
  int nfiles = args.multi.size();
  std::vector<DedispersedTimeSeries<unsigned char> > tims;

  for (ii=0;ii<nfiles;ii++){
    if (args.verbose)
      std::cout << "Reading and dedispersing " << args.multi[ii] << std::endl;
    SigprocFilterbank filterbank(args.multi[ii]);
    Dedisperser dedisperser(filterbank,1);
    dedisperser.generate_dm_list(0.0,0.0,0.4,1.1);
    DispersionTrials<unsigned char> trial = dedisperser.dedisperse(); 
    tims.push_back(trial[0]);
  }
  
  size_t size = tims[0].get_nsamps();
  for (ii=0;ii<nfiles;ii++){
    if (tims[ii].get_nsamps() != size)
      ErrorChecker::throw_error("Not all filterbanks the same length");
  }
  
  float tsamp = tims[0].get_tsamp();
  CuFFTerR2C r2cfft(size);
  CuFFTerC2R c2rfft(size);
  float tobs = size*tsamp;
  float bin_width = 1.0/tobs;
    
  std::vector< ReusableDeviceTimeSeries<float,unsigned char>* > d_tims;
  for (ii=0;ii<nfiles;ii++){
    d_tims.push_back(new ReusableDeviceTimeSeries<float,unsigned char>(size));
  }
    
  DeviceFourierSeries<cufftComplex> d_fseries(size/2+1,bin_width);
  std::vector< DevicePowerSpectrum<float>* > pspecs;
  for (ii=0;ii<nfiles;ii++){
    pspecs.push_back(new  DevicePowerSpectrum<float>(d_fseries));
  }
  
  Dereddener rednoise(size/2+1);
  SpectrumFormer former;
  float mean,std,rms;

  for (ii=0;ii<nfiles;ii++){
    if (args.verbose)
      std::cout << "Baselining beam " << ii << std::endl;
    d_tims[ii]->copy_from_host(tims[ii]);
    r2cfft.execute(d_tims[ii]->get_data(),d_fseries.get_data());
    
    former.form(d_fseries,*pspecs[ii]);
    rednoise.calculate_median(*pspecs[ii]);
    rednoise.deredden(d_fseries);
    
    former.form_interpolated(d_fseries,*pspecs[ii]);
    stats::stats<float>(pspecs[ii]->get_data(),size/2+1,&mean,&rms,&std);
    stats::normalise(pspecs[ii]->get_data(),mean,std,size/2+1);

    c2rfft.execute(d_fseries.get_data(),d_tims[ii]->get_data());
    stats::stats<float>(d_tims[ii]->get_data(),size,&mean,&rms,&std);
    stats::normalise(d_tims[ii]->get_data(),mean,std,size);
  }

  if (args.verbose)
    std::cout << "Performing cross beam coincidence matching" << std::endl;

  std::vector<float*> tim_host_ptr_array;
  std::vector<float*> spec_host_ptr_array;
  for (ii=0;ii<nfiles;ii++){
    tim_host_ptr_array.push_back(d_tims[ii]->get_data());
    spec_host_ptr_array.push_back(pspecs[ii]->get_data());
  }

  DevicePowerSpectrum<float> spec_mask(d_fseries);
  DeviceTimeSeries<float> samp_mask(size);
  Coincidencer ccder(nfiles);
  
  ccder.match(&tim_host_ptr_array[0],samp_mask.get_data(),
	      size,args.threshold,args.beam_threshold);
  
  ccder.match(&spec_host_ptr_array[0],spec_mask.get_data(),
	      size/2+1,args.threshold,args.beam_threshold);
  
  ccder.write_samp_mask(samp_mask.get_data(),size,
			args.samp_outfilename);
  
  ccder.write_birdie_list(spec_mask.get_data(),size/2+1, 
			  spec_mask.get_bin_width(),
			  args.spec_outfilename);
  
  
  for (ii=0;ii<nfiles;ii++){
    delete d_tims[ii];
    delete pspecs[ii];
  }
  return 0;
}
