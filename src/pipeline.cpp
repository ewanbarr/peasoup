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
#include <tclap/CmdLine.h>
#include <string>
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include "cuda.h"
#include "cufft.h"
#include <cmath>

struct CmdLineOptions {
  std::string infilename;
  std::string outfilename;
  std::string killfilename;
  std::string zapfilename;
  unsigned int max_num_threads;
  unsigned int size; 
  float dm_start; 
  float dm_end;
  float dm_tol;
  float dm_pulse_width;
  float acc_start;
  float acc_end;
  float acc_tol;
  float acc_pulse_width;
  float boundary_5_freq;
  float boundary_25_freq;
  int nharmonics;
  float min_snr;
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
      
      TCLAP::ValueArg<std::string> arg_infilename("i", "inputfile",
						  "File to process (.fil)",
						  true, "", "string", cmd);
      
      TCLAP::ValueArg<std::string> arg_outfilename("o", "outputfile",
						   "The output filename",
						   false, "", "string",cmd);

      TCLAP::ValueArg<std::string> arg_killfilename("k", "killfile",
                                                   "Channel mask file",
                                                   false, "", "string",cmd);
      
      TCLAP::ValueArg<std::string> arg_zapfilename("z", "zapfile",
						   "Birdie list file",
						   false, "", "string",cmd);

      TCLAP::ValueArg<int> arg_max_num_threads("c", "num_threads", 
					       "The number of GPUs to use",
					       false, 14, "int", cmd);
      
      TCLAP::ValueArg<size_t> arg_size("x", "size",
				       "Transform size to use (defaults to lower power of two)",
				       false, 0, "size_t", cmd);
      
      TCLAP::ValueArg<float> arg_dm_start("s", "dm_start", 
					  "First DM to dedisperse to",
					  false, 0.0, "float", cmd);
      
      TCLAP::ValueArg<float> arg_dm_end("e", "dm_end",
					"Last DM to dedisperse to",
					false, 100.0, "float", cmd);
      
      TCLAP::ValueArg<float> arg_dm_tol("t", "dm_tol",
					"DM smearing tolerance (1.11=10%)",
					false, 1.10, "float",cmd);

      TCLAP::ValueArg<float> arg_dm_pulse_width("p", "dm_pulse_width",
						"Minimum pulse width for which dm_tol is valid",
						false, 0.4, "float (ms)",cmd);

      TCLAP::ValueArg<float> arg_acc_start("S", "acc_start",
                                          "First acceleration to resample to",
                                          false, 0.0, "float", cmd);

      TCLAP::ValueArg<float> arg_acc_end("E", "acc_end",
                                        "Last acceleration to resample to",
                                        false, 0.0, "float", cmd);

      TCLAP::ValueArg<float> arg_acc_tol("T", "acc_tol",
                                        "Acceleration smearing tolerance (1.11=10%)",
                                        false, 1.10, "float",cmd);

      TCLAP::ValueArg<float> arg_acc_pulse_width("P", "acc_pulse_width",
						 "Minimum pulse width for which acc_tol is valid",
                                                false, 0.4, "float (ms)",cmd);
            
      TCLAP::ValueArg<float> arg_boundary_5_freq("l", "boundary_5_freq",
						 "Frequency at which to switch from median5 to median25",
						 false, 0.05, "float", cmd);
      
      TCLAP::ValueArg<float> arg_boundary_25_freq("a", "boundary_25_freq",
						 "Frequency at which to switch from median25 to median125",
                                                 false, 0.5, "float", cmd);
      
      TCLAP::ValueArg<int> arg_nharmonics("n", "nharmonics",
					  "Number of harmonic sums to perform",
					  false, 4, "int", cmd);

      TCLAP::ValueArg<float> arg_min_snr("m", "min_snr", 
					 "The minimum S/N for a candidate",
					 false, 9.0, "float",cmd);
      
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
      args.infilename        = arg_infilename.getValue();
      args.outfilename       = arg_outfilename.getValue();
      args.killfilename      = arg_killfilename.getValue();
      args.zapfilename       = arg_zapfilename.getValue();
      args.max_num_threads   = arg_max_num_threads.getValue();
      args.size              = arg_size.getValue();
      args.dm_start          = arg_dm_start.getValue();
      args.dm_end            = arg_dm_end.getValue();
      args.dm_tol            = arg_dm_tol.getValue();
      args.dm_pulse_width    = arg_dm_pulse_width.getValue();
      args.acc_start         = arg_acc_start.getValue();
      args.acc_end           = arg_acc_end.getValue();
      args.acc_tol           = arg_acc_tol.getValue();
      args.acc_pulse_width   = arg_acc_pulse_width.getValue();
      args.boundary_5_freq   = arg_boundary_5_freq.getValue();   
      args.boundary_25_freq  = arg_boundary_25_freq.getValue();
      args.nharmonics        = arg_nharmonics.getValue();
      args.min_snr           = arg_min_snr.getValue();
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

  if (args.verbose)
    std::cout << "Using file: " << args.infilename << std::endl;
  std::string filename(args.infilename);
  SigprocFilterbank filobj(filename);
  Dedisperser dedisperser(filobj,2);

  if (args.killfilename!=""){
    if (args.verbose)
      std::cout << "Using killfile: " << args.killfilename << std::endl;
    dedisperser.set_killmask(args.killfilename);
  }
  
  if (args.verbose)
    std::cout << "Generating DM list" << std::endl;
  dedisperser.generate_dm_list(args.dm_start,args.dm_end,args.dm_pulse_width,args.dm_tol);
  
  if (args.verbose){
    std::vector<float> dm_list = dedisperser.get_dm_list();
    std::cout << dm_list.size() << " DM trials" << std::endl;
    for (ii=0;ii<dm_list.size();ii++)
      std::cout << dm_list[ii] << std::endl;
    std::cout << "Executing dedispersion" << std::endl;
  }
  DispersionTrials<unsigned char> trials = dedisperser.dedisperse();
  
  unsigned int size;
  if (args.size==0)
    size = Utils::prev_power_of_two(filobj.get_nsamps());
  else
    size = std::min(args.size,filobj.get_nsamps());
  if (args.verbose)
    std::cout << "Setting transform length to " << size << " points" << std::endl;

  CuFFTerR2C r2cfft(size);
  CuFFTerC2R c2rfft(size);
  float tobs = size*filobj.get_tsamp();
  float bin_width = 1.0/tobs;
  DeviceFourierSeries<cufftComplex> d_fseries(size/2+1,bin_width);
  DedispersedTimeSeries<unsigned char> tim;
  ReusableDeviceTimeSeries<float,unsigned char> d_tim(size);
  DeviceTimeSeries<float> d_tim_r(size);
  TimeDomainResampler resampler;
  DevicePowerSpectrum<float> pspec(d_fseries);
  Zapper* bzap;
  if (args.zapfilename!=""){
    if (args.verbose)
      std::cout << "Using zapfile: " << args.zapfilename << std::endl;
    bzap = new Zapper(args.zapfilename);
  }
  
  Dereddener rednoise(size/2+1);
  SpectrumFormer former;
  PeakFinder cand_finder(args.min_snr,args.min_freq,args.max_freq);
  
  HarmonicFolder harm_folder;
  HarmonicSums<float> sums(pspec,args.nharmonics);
  
  HarmonicDistiller harm_finder(args.freq_tol,args.max_harm);
  AccelerationDistiller acc_still(tobs,args.freq_tol);
  DMDistiller dm_still(args.freq_tol);
  float mean,std,rms;
  CandidateCollection dm_trial_cands;
  for (ii=0; ii < (int)trials.get_count(); ii++){
    tim = trials[ii];
    if (args.verbose)
      std::cout << "Copying DM trial to device (DM: " << tim.get_dm() << ")"<< std::endl;
    d_tim.copy_from_host(tim);
    
    if (args.verbose)
      std::cout << "Executing forward FFT" << std::endl;
    r2cfft.execute(d_tim.get_data(),d_fseries.get_data());
    
    if (args.verbose)
      std::cout << "Forming power spectrum" << std::endl;
    former.form(d_fseries,pspec);
    
    if (args.verbose)
      std::cout << "Finding running median" << std::endl;
    rednoise.calculate_median(pspec);
    
    if (args.verbose)
      std::cout << "Dereddening Fourier series" << std::endl;
    rednoise.deredden(d_fseries);
    
    if (args.zapfilename!=""){
      if (args.verbose)
	std::cout << "Zapping birdies" << std::endl;
      bzap->zap(d_fseries);
    }
    
    if (args.verbose)
      std::cout << "Forming interpolated power spectrum" << std::endl;
    former.form_interpolated(d_fseries,pspec);
    
    if (args.verbose)
      std::cout << "Finding statistics" << std::endl;
    stats::stats<float>(pspec.get_data(),size/2+1,&mean,&rms,&std);
    
    if (args.verbose)
      std::cout << "Executing inverse FFT" << std::endl;
    c2rfft.execute(d_fseries.get_data(),d_tim.get_data());
    
    CandidateCollection accel_trial_cands;
    
    for (float jj=args.acc_start;jj<args.acc_end;jj+=0.5){
      if (args.verbose)
	std::cout << "Resampling to "<<jj<< " m/s/s" << std::endl;
      resampler.resample(d_tim,d_tim_r,size,jj);
      
      if (args.verbose)
	std::cout << "Execute forward FFT" << std::endl;
      r2cfft.execute(d_tim_r.get_data(),d_fseries.get_data());

      if (args.verbose)
	std::cout << "Form interpolated power spectrum" << std::endl;
      former.form_interpolated(d_fseries,pspec);

      if (args.verbose)
	std::cout << "Normalise power spectrum" << std::endl;
      stats::normalise(pspec.get_data(),mean*size,std*size,size/2+1);

      if (args.verbose)
	std::cout << "Harmonic summing" << std::endl;
      harm_folder.fold(pspec,sums);
  
      if (args.verbose)
	std::cout << "Finding peaks" << std::endl;
      SpectrumCandidates trial_cands(tim.get_dm(),ii,jj);
      cand_finder.find_candidates(pspec,trial_cands);
      cand_finder.find_candidates(sums,trial_cands);
      
      if (args.verbose)
	std::cout << "Distilling harmonics" << std::endl;
      accel_trial_cands.append(harm_finder.distill(trial_cands.cands));
    }
    if (args.verbose)
      std::cout << "Distilling accelerations" << std::endl;
    dm_trial_cands.append(acc_still.distill(accel_trial_cands.cands));
  }
  if (args.verbose)
    std::cout << "Distilling DMs" << std::endl;
  dm_trial_cands.cands = dm_still.distill(dm_trial_cands.cands);
  dm_trial_cands.print();

  if (args.zapfilename!="")
    delete bzap;
  
  if (args.verbose)
    std::cout << "Setting up time series folder" << std::endl;
  
  MultiFolder folder(dm_trial_cands.cands,trials);
  folder.fold_n(3000);
  std::cout << "\n--------------\n" << std::endl;
  dm_trial_cands.print();

  return 0;
}
