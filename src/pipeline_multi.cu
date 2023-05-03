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
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include "cuda.h"
#include "cufft.h"
#include "pthread.h"
#include <cmath>
#include <map>

typedef float DedispOutputType;

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

  virtual ~DMDispenser(){
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
    :trials(trials)
    ,manager(manager)
    ,acc_plan(acc_plan)
    ,args(args)
    ,size(size)
    ,device(device)
  {}

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
    // If filterbank file is coherently dedispersed at a non-zero value
    float cdm = args.cdm;
    if (args.verbose)
        if (cdm != 0.0)
	        std::cout << "Filterbank file is coherently dedispersed at DM: " << args.cdm << ". Will adjust acceleration trial step size accordingly. " << std::endl; 
        else
            std::cout << "Filterbank file is not coherently dedispersed. " << std::endl;
        
    
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
      trials.get_idx(ii, tim);

      if (args.verbose)
      {
          std::cout << "Copying DM trial to device (DM: " << tim.get_dm() << ")"<< std::endl;
          std::cout << "Transferring " << tim.get_nsamps() << " samples" << std::endl;
      }
      //Utils::dump_host_buffer<float>(tim.get_data(), tim.get_nsamps(), "raw_timeseries_before_baseline_removal_host.dump");
      d_tim.copy_from_host(tim);
      if (args.verbose) std::cout << "Copy from host complete\n";
      //Utils::dump_device_buffer<float>(d_tim.get_data(), d_tim.get_nsamps(), "raw_timeseries_before_baseline_removal.dump");
      if (args.verbose) std::cout << "Removing baseline\n";
      d_tim.remove_baseline(std::min(tim.get_nsamps(), d_tim.get_nsamps()));
      if (args.verbose) std::cout << "Baseline removed\n";
      //Utils::dump_device_buffer<float>(d_tim.get_data(), d_tim.get_nsamps(), "raw_timeseries_after_baseline_removal.dump");

      //timers["rednoise"].start()
      if (padding){
      if (args.verbose) std::cout << "Padding with zeros\n";
            if (tim.get_nsamps() >= d_tim.get_nsamps()){
                //NOOP
            } else {
                d_tim.fill(trials.get_nsamps(), d_tim.get_nsamps(), 0);
            }
	    //padding_mean = stats::mean<float>(d_tim.get_data(),trials.get_nsamps());
      }
      
      if (args.verbose)
	    std::cout << "Generating accelration list" << std::endl;
      acc_plan.generate_accel_list(tim.get_dm(), cdm, acc_list);

      if (args.verbose)
	    std::cout << "Searching "<< acc_list.size()<< " acceleration trials for DM "<< tim.get_dm() << std::endl;

      //Utils::dump_device_buffer<float>(d_tim.get_data(), d_tim.get_nsamps(), "raw_timeseries_after_padding.dump");


      if (args.verbose)
	    std::cout << "Executing forward FFT" << std::endl;
      r2cfft.execute(d_tim.get_data(),d_fseries.get_data());

      //Utils::dump_device_buffer<cufftComplex>(d_fseries.get_data(), d_fseries.get_nbins(), "fourier_series.dump");

      if (args.verbose)
	    std::cout << "Forming power spectrum" << std::endl;
      former.form(d_fseries,pspec);

      //Utils::dump_device_buffer<float>(pspec.get_data(), pspec.get_nbins(), "power_spec.dump");

      if (args.verbose)
	    std::cout << "Finding running median" << std::endl;
      rednoise.calculate_median(pspec);

      if (args.verbose)
	    std::cout << "Dereddening Fourier series" << std::endl;
      rednoise.deredden(d_fseries);

      //Utils::dump_device_buffer<cufftComplex>(d_fseries.get_data(), d_fseries.get_nbins(), "deredden_fourier_series.dump");


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

      //Utils::dump_device_buffer<float>(d_tim.get_data(), d_tim.get_nsamps(), "deredden_ifft.dump");


      CandidateCollection accel_trial_cands;
      PUSH_NVTX_RANGE("Acceleration-Loop",1)

      for (int jj=0;jj<acc_list.size();jj++){
  	    if (args.verbose)
  	      std::cout << "Resampling to "<< acc_list[jj] << " m/s/s" << std::endl;
  	    resampler.resampleII(d_tim,d_tim_r,size,acc_list[jj]);

        //Utils::dump_device_buffer<float>(d_tim_r.get_data(), d_tim_r.get_nsamps(), "resampler_out.dump");


  	    if (args.verbose)
  	      std::cout << "Execute forward FFT" << std::endl;
  	    r2cfft.execute(d_tim_r.get_data(),d_fseries.get_data());

        //Utils::dump_device_buffer<cufftComplex>(d_fseries.get_data(), d_fseries.get_nbins(), "search_fourier_series.dump");


  	    if (args.verbose)
  	      std::cout << "Form interpolated power spectrum" << std::endl;
  	    former.form_interpolated(d_fseries,pspec);

        //Utils::dump_device_buffer<float>(pspec.get_data(), pspec.get_nbins(), "search_power_spec.dump");



  	    if (args.verbose)
  	      std::cout << "Normalise power spectrum" << std::endl;
  	    stats::normalise(pspec.get_data(),mean*size,std*size,size/2+1);

        //Utils::dump_device_buffer<float>(pspec.get_data(), pspec.get_nbins(), "search_normalised_power_spec.dump");



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


bool getFileContent(std::string fileName, std::vector<float> & vecOfDMs)
{
    // Open the File
    std::ifstream in(fileName.c_str());
    // Check if object is valid
    if(!in)
    {
        std::cerr << "Cannot open the File : "<<fileName<<std::endl;
        return false;
    }
    std::string str;
    float fl;
    // Read the next line from File untill it reaches the end.
    while (std::getline(in, str))
    {
        // Line contains string of length > 0 then save it in vector
        if(str.size() > 0)
            fl = std::atof(str.c_str());
            //fl = std::stof(str); //c++11
            vecOfDMs.push_back(fl);
    }
    //Close The File
    in.close();
    return true;
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

  /* Could do a check on the GPU memory usage here */

  if (args.verbose)
    std::cout << "Using file: " << args.infilename << std::endl;
  std::string filename(args.infilename);

  //Stopwatch timer;
  if (args.progress_bar)
    printf("Reading header from %s\n",args.infilename.c_str());

  timers["reading"].start();
  SigprocFilterbank filobj(filename);
  timers["reading"].stop();

  if (args.progress_bar){
    printf("Complete (execution time %.2f s)\n",timers["reading"].getTime());
  }

  DMDistiller dm_still(args.freq_tol,true);
  HarmonicDistiller harm_still(args.freq_tol,args.max_harm,true,false);
  CandidateCollection dm_cands;

  unsigned int size;
  if (args.size==0)
    size = Utils::prev_power_of_two(filobj.get_nsamps());
  else
    //size = std::min(args.size,filobj.get_nsamps());
    size = args.size;
  if (args.verbose)
    std::cout << "Setting transform length to " << size << " points" << std::endl;

  AccelerationPlan acc_plan(
    args.acc_start, // m/s^2
    args.acc_end,   // m/s^2
    args.acc_tol,   // dimensionless
    args.acc_pulse_width * 1e-6, // cmd line arg is microseconds but needs to be passed as seconds
    size, // dimensionless
    filobj.get_tsamp(), // seconds
    filobj.get_cfreq() * 1e6, // from header in MHz needs converted to Hz
    filobj.get_foff() * 1e6 // from header in MHz needs converted to Hz
    );


  if (args.verbose)
    std::cout << "Generating DM list" << std::endl;
  std::vector<float> full_dm_list;

  if (args.dm_file=="none") {
    Dedisperser dedisperser(filobj, nthreads);
    dedisperser.generate_dm_list(args.dm_start, args.dm_end, args.dm_pulse_width, args.dm_tol);
    full_dm_list = dedisperser.get_dm_list();
  }
  else {
      bool result = getFileContent(args.dm_file, full_dm_list);
  }

  float nbytes = args.host_ram_limit_gb * 1e9;
  std::size_t ndm_trial_gulp = std::size_t(nbytes / (filobj.get_nsamps() * sizeof(float)));
  if (ndm_trial_gulp == 0)
  {
    throw std::runtime_error("Insufficient RAM specified to allow for dedispersion");
  }
  else if (ndm_trial_gulp > full_dm_list.size())
  {
    ndm_trial_gulp = full_dm_list.size();
  }
  for(std::size_t idx=0; idx < full_dm_list.size(); idx += ndm_trial_gulp){
    std::size_t start = idx;
    std::size_t end   = (idx + ndm_trial_gulp) > full_dm_list.size() ? full_dm_list.size(): (idx + ndm_trial_gulp) ;
    if(args.verbose) std::cout << "Gulp start: " << start << " end: " << end << std::endl;
    std::vector<float> dm_list_chunk(full_dm_list.begin() + start,  full_dm_list.begin() + end);
    Dedisperser dedisperser(filobj, nthreads);
    if (args.killfilename!=""){
      if (args.verbose)
        std::cout << "Using killfile: " << args.killfilename << std::endl;
      dedisperser.set_killmask(args.killfilename);
    }

    dedisperser.set_dm_list(dm_list_chunk);

    if (args.verbose){
    std::cout << dm_list_chunk.size() << " DM trials" << std::endl;
    for (std::size_t ii = 0; ii < dm_list_chunk.size(); ii++)
    {
      std::cout << dm_list_chunk[ii] << std::endl;
    }
    std::cout << "Executing dedispersion" << std::endl;
    }

    if (args.progress_bar) std::cout <<"Starting dedispersion: " << start << " to " << end << "..." << std::endl;

    timers["dedispersion"].start();
    PUSH_NVTX_RANGE("Dedisperse",3)
    DispersionTrials<DedispOutputType> trials(filobj.get_tsamp());
    std::cout <<"dedispersing...." <<std::endl;

    std::size_t gulp_size;
    if (args.dedisp_gulp == -1){
      gulp_size = filobj.get_nsamps();
    } else {
      gulp_size = args.dedisp_gulp;
    }

    dedisperser.dedisperse(trials, 0, filobj.get_nsamps(), gulp_size);
    POP_NVTX_RANGE
    timers["dedispersion"].stop();

    //Write out a dedispersed time series file from the dedispersion tials
    //  unsigned int* data_ptr = trials[0].get_data();
    //  Utils::dump_host_buffer<unsigned int>(data_ptr,trials.get_nsamps(),"dedispersed_timeseries_new");

    if (args.progress_bar)
      printf("Complete (execution time %.2f s)\n",timers["dedispersion"].getTime());

    std::cout <<"Starting searching..."  << std::endl;

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

    if(args.verbose)
      std::cout << "Joining worker threads" << std::endl;

    for (int ii=0; ii<nthreads; ii++){
      pthread_join(threads[ii],NULL);
      dm_cands.append(workers[ii]->dm_trial_cands.cands);
      delete workers[ii];
    }
    timers["searching"].stop();

    if (args.progress_bar)
      printf("Complete (execution time %.2f s)\n",timers["searching"].getTime());


  }

  if (args.verbose)
    std::cout << "Distilling DMs" << std::endl;
  dm_cands.cands = dm_still.distill(dm_cands.cands);
  dm_cands.cands = harm_still.distill(dm_cands.cands);

  CandidateScorer cand_scorer(filobj.get_tsamp(),filobj.get_cfreq(), filobj.get_foff(),
			      fabs(filobj.get_foff())*filobj.get_nchans());
  cand_scorer.score_all(dm_cands.cands);

  if (args.verbose)
    std::cout << "Setting up time series folder" << std::endl;

  // MultiFolder folder(dm_cands.cands,trials);
  // timers["folding"].start();
  // if (args.progress_bar)
  //   folder.enable_progress_bar();

  // if (args.npdmp > 0){
  //   if (args.verbose)
  //     std::cout << "Folding top "<< args.npdmp <<" cands" << std::endl;
  //   folder.fold_n(args.npdmp);
  // }
  // timers["folding"].stop();

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
  stats.add_dm_list(full_dm_list);

  std::vector<float> acc_list;
  // If filterbank file is coherently dedispersed at a non-zero value
  float cdm = args.cdm;
  //acc_plan.generate_accel_list(0.0,cdm,acc_list);
  acc_plan.generate_accel_list(cdm, cdm, acc_list);
  stats.add_acc_list(cdm, acc_list);

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

  std::cerr << "all done" << std::endl;

  return 0;
}
