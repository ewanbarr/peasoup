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
#include <transforms/scorer.hpp>
#include <utils/exceptions.hpp>
#include <utils/utils.hpp>
#include <utils/stats.hpp>
#include <utils/stopwatch.hpp>
#include <utils/progress_bar.hpp>
#include <utils/cmdline.hpp>
#include <utils/output_stats.hpp>
#include <string>
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include "cuda.h"
#include "cufft.h"
#include "pthread.h"
#include <cmath>
#include <map>

typedef unsigned int DedispOutputType; 

class DMDispenser {
private:
  DispersionTrials<DedispOutputType>& trials;
  pthread_mutex_t mutex;
  int dm_idx;
  int count;
  ProgressBar* progress;
  bool use_progress_bar;

public:
  DMDispenser(DispersionTrials<DedispOutputType>& trials)
    :trials(trials),dm_idx(0),use_progress_bar(false){
    count = trials.get_count();
    pthread_mutex_init(&mutex, NULL);
  }
  
  void enable_progress_bar(){
    progress = new ProgressBar();
    use_progress_bar = true;
  }

  int get_dm_trial_idx(void){
    pthread_mutex_lock(&mutex);
    int retval;
    if (dm_idx==0)
      if (use_progress_bar){
	printf("Releasing DMs to workers...\n");
	progress->start();
      }
    if (dm_idx >= trials.get_count()){
      retval =  -1;
      if (use_progress_bar)
	progress->stop();
    } else {
      if (use_progress_bar)
	progress->set_progress((float)dm_idx/count);
      retval = dm_idx;
      dm_idx++;
    }
    pthread_mutex_unlock(&mutex);
    return retval;
  }
  
  ~DMDispenser(){
    if (use_progress_bar)
      delete progress;
    pthread_mutex_destroy(&mutex);
  }
};

class Worker {
private:
  DispersionTrials<DedispOutputType>& trials;
  DMDispenser& manager;
  CmdLineOptions& args;
  AccelerationPlan& acc_plan;
  unsigned int size;
  int device;
  std::map<std::string,Stopwatch> timers;
  
public:
  CandidateCollection dm_trial_cands;

  Worker(DispersionTrials<DedispOutputType>& trials, DMDispenser& manager, 
	 AccelerationPlan& acc_plan, CmdLineOptions& args, unsigned int size, int device)
    :trials(trials),manager(manager),acc_plan(acc_plan),args(args),size(size),device(device){}
  
  void start(void)
  {
    //Generate some timer instances for benchmarking
    //timers["get_dm_trial"]      = Stopwatch();
    //timers["copy_to_device"] = Stopwatch();
    //timers["rednoise"]    = Stopwatch();
    //timers["search"]      = Stopwatch();

    cudaSetDevice(device);
    Stopwatch pass_timer;
    pass_timer.start();

    bool padding = false;
    if (size > trials.get_nsamps())
      padding = true;
    
    CuFFTerR2C r2cfft(size);
    CuFFTerC2R c2rfft(size);
    float tobs = size*trials.get_tsamp();
    float bin_width = 1.0/tobs;
    DeviceFourierSeries<cufftComplex> d_fseries(size/2+1,bin_width);
    DedispersedTimeSeries<DedispOutputType> tim;
    ReusableDeviceTimeSeries<float, DedispOutputType> d_tim(size);
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
    PeakFinder cand_finder(args.min_snr,args.min_freq,args.max_freq,size);
    HarmonicSums<float> sums(pspec,args.nharmonics);
    HarmonicFolder harm_folder(sums);
    std::vector<float> acc_list;
    HarmonicDistiller harm_finder(args.freq_tol,args.max_harm,false);
    AccelerationDistiller acc_still(tobs,args.freq_tol,true);
    float mean,std,rms;
    float padding_mean;
    int ii;

	PUSH_NVTX_RANGE("DM-Loop",0)
    while (true){
      //timers["get_trial_dm"].start();
      ii = manager.get_dm_trial_idx();
      //timers["get_trial_dm"].stop();

      if (ii==-1)
        break;
      trials.get_idx(ii,tim);
      
      if (args.verbose)
	std::cout << "Copying DM trial to device (DM: " << tim.get_dm() << ")"<< std::endl;

      d_tim.copy_from_host(tim);
      
      //timers["rednoise"].start()
      if (padding){
	    padding_mean = stats::mean<float>(d_tim.get_data(),trials.get_nsamps());
	    d_tim.fill(trials.get_nsamps(),d_tim.get_nsamps(),padding_mean);
      }

      if (args.verbose)
	    std::cout << "Generating accelration list" << std::endl;
      acc_plan.generate_accel_list(tim.get_dm(),acc_list);
      
      if (args.verbose)
	    std::cout << "Searching "<< acc_list.size()<< " acceleration trials for DM "<< tim.get_dm() << std::endl;

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
      PUSH_NVTX_RANGE("Acceleration-Loop",1)

      for (int jj=0;jj<acc_list.size();jj++){
	    if (args.verbose)
	      std::cout << "Resampling to "<< acc_list[jj] << " m/s/s" << std::endl;
	    resampler.resampleII(d_tim,d_tim_r,size,acc_list[jj]);

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
	    harm_folder.fold(pspec);
		
	    if (args.verbose)
	      std::cout << "Finding peaks" << std::endl;
	    SpectrumCandidates trial_cands(tim.get_dm(),ii,acc_list[jj]);
	    cand_finder.find_candidates(pspec,trial_cands);
	    cand_finder.find_candidates(sums,trial_cands);
	
	    if (args.verbose)
	      std::cout << "Distilling harmonics" << std::endl;
	      accel_trial_cands.append(harm_finder.distill(trial_cands.cands));
      }
	  POP_NVTX_RANGE
      if (args.verbose)
	    std::cout << "Distilling accelerations" << std::endl;
      dm_trial_cands.append(acc_still.distill(accel_trial_cands.cands));
    }
	POP_NVTX_RANGE
	
    if (args.zapfilename!="")
      delete bzap;
    
    if (args.verbose)
      std::cout << "DM processing took " << pass_timer.getTime() << " seconds"<< std::endl;
  }
  
};

void* launch_worker_thread(void* ptr){
  reinterpret_cast<Worker*>(ptr)->start();
  return NULL;
}


int main(int argc, char **argv)
{
  std::map<std::string,Stopwatch> timers;
  timers["reading"]      = Stopwatch();
  timers["dedispersion"] = Stopwatch();
  timers["searching"]    = Stopwatch();
  timers["folding"]      = Stopwatch();
  timers["total"]        = Stopwatch();
  timers["total"].start();

  CmdLineOptions args;
  if (!read_cmdline_options(args,argc,argv))
    ErrorChecker::throw_error("Failed to parse command line arguments.");

  int nthreads = std::min(Utils::gpu_count(),args.max_num_threads);
  nthreads = std::max(1,nthreads);

  if (args.verbose)
    std::cout << "Using file: " << args.infilename << std::endl;
  std::string filename(args.infilename);

  //Stopwatch timer;
  if (args.progress_bar)
    printf("Reading data from %s\n",args.infilename.c_str());
  
  timers["reading"].start();
  SigprocFilterbank filobj(filename);
  timers["reading"].stop();
    
  if (args.progress_bar){
    printf("Complete (execution time %.2f s)\n",timers["reading"].getTime());
  }

  Dedisperser dedisperser(filobj,nthreads);
  if (args.killfilename!=""){
    if (args.verbose)
      std::cout << "Using killfile: " << args.killfilename << std::endl;
    dedisperser.set_killmask(args.killfilename);
  }
  
  if (args.verbose)
    std::cout << "Generating DM list" << std::endl;
  dedisperser.generate_dm_list(args.dm_start,args.dm_end,args.dm_pulse_width,args.dm_tol);
  std::vector<float> dm_list = dedisperser.get_dm_list();
  
  if (args.verbose){
    std::cout << dm_list.size() << " DM trials" << std::endl;
    for (int ii=0;ii<dm_list.size();ii++)
      std::cout << dm_list[ii] << std::endl;
    std::cout << "Executing dedispersion" << std::endl;
  }

  if (args.progress_bar)
    printf("Starting dedispersion...\n");

  timers["dedispersion"].start();
  PUSH_NVTX_RANGE("Dedisperse",3)
  DispersionTrials<DedispOutputType> trials = dedisperser.dedisperse();
  POP_NVTX_RANGE
  timers["dedispersion"].stop();


//Write out a dedispersed time series file from the dedispersion tials
//  unsigned int* data_ptr = trials[0].get_data();
//  Utils::dump_host_buffer<unsigned int>(data_ptr,trials.get_nsamps(),"dedispersed_timeseries_new");

  if (args.progress_bar)
    printf("Complete (execution time %.2f s)\n",timers["dedispersion"].getTime());

  unsigned int size;
  if (args.size==0)
    size = Utils::prev_power_of_two(filobj.get_nsamps());
  else
    //size = std::min(args.size,filobj.get_nsamps());
    size = args.size;
  if (args.verbose)
    std::cout << "Setting transform length to " << size << " points" << std::endl;
  
  AccelerationPlan acc_plan(args.acc_start, args.acc_end, args.acc_tol,
			    args.acc_pulse_width, size, filobj.get_tsamp(),
			    filobj.get_cfreq(), filobj.get_foff()); 
  
  
  //Multithreading commands
  timers["searching"].start();
  std::vector<Worker*> workers(nthreads);
  std::vector<pthread_t> threads(nthreads);
  DMDispenser dispenser(trials);
  if (args.progress_bar)
    dispenser.enable_progress_bar();
  
  for (int ii=0;ii<nthreads;ii++){
    workers[ii] = (new Worker(trials,dispenser,acc_plan,args,size,ii));
    pthread_create(&threads[ii], NULL, launch_worker_thread, (void*) workers[ii]);
  }
  
  DMDistiller dm_still(args.freq_tol,true);
  HarmonicDistiller harm_still(args.freq_tol,args.max_harm,true,false);
  CandidateCollection dm_cands;
  for (int ii=0; ii<nthreads; ii++){
    pthread_join(threads[ii],NULL);
    dm_cands.append(workers[ii]->dm_trial_cands.cands);
  }
  timers["searching"].stop();
  
  if (args.verbose)
    std::cout << "Distilling DMs" << std::endl;
  dm_cands.cands = dm_still.distill(dm_cands.cands);
  dm_cands.cands = harm_still.distill(dm_cands.cands);
  
  CandidateScorer cand_scorer(filobj.get_tsamp(),filobj.get_cfreq(), filobj.get_foff(),
			      fabs(filobj.get_foff())*filobj.get_nchans());
  cand_scorer.score_all(dm_cands.cands);

  if (args.verbose)
    std::cout << "Setting up time series folder" << std::endl;
  
  MultiFolder folder(dm_cands.cands,trials);
  timers["folding"].start();
  if (args.progress_bar)
    folder.enable_progress_bar();

  if (args.npdmp > 0){
    if (args.verbose)
      std::cout << "Folding top "<< args.npdmp <<" cands" << std::endl;
    folder.fold_n(args.npdmp);
  }
  timers["folding"].stop();

  if (args.verbose)
    std::cout << "Writing output files" << std::endl;
  //dm_cands.write_candidate_file("./old_cands.txt");
  
  int new_size = std::min(args.limit,(int) dm_cands.cands.size());
  dm_cands.cands.resize(new_size);

  CandidateFileWriter cand_files(args.outdir);
  cand_files.write_binary(dm_cands.cands,"candidates.peasoup");
  
  OutputFileWriter stats;
  stats.add_misc_info();
  stats.add_header(filename);
  stats.add_search_parameters(args);
  stats.add_dm_list(dm_list);
  
  std::vector<float> acc_list;
  acc_plan.generate_accel_list(0.0,acc_list);
  stats.add_acc_list(acc_list);
  
  std::vector<int> device_idxs;
  for (int device_idx=0;device_idx<nthreads;device_idx++)
    device_idxs.push_back(device_idx);
  stats.add_gpu_info(device_idxs);
  stats.add_candidates(dm_cands.cands,cand_files.byte_mapping);
  timers["total"].stop();
  stats.add_timing_info(timers);
  
  std::stringstream xml_filepath;
  xml_filepath << args.outdir << "/" << "overview.xml";
  stats.to_file(xml_filepath.str());
  
  return 0;
}
