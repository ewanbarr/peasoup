#pragma once
#include <kernels/defaults.h>
#include <kernels/kernels.h>
#include <utils/exceptions.hpp>
#include <utils/utils.hpp>
#include <utils/stats.hpp>
#include <utils/progress_bar.hpp>
#include <data_types/folded.hpp>
#include <data_types/candidates.hpp>
#include <transforms/dereddener.hpp>
#include <transforms/spectrumformer.hpp>
#include <transforms/ffter.hpp>
#include <transforms/resampler.hpp>
#include <data_types/fourierseries.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <map>
#include <stdio.h>
#include "cuda.h"
#include "cufft.h"
#include "cuComplex.h"
#include <iostream>

struct less_than_key
{
  inline bool operator() (const Candidate& x, const Candidate& y)
  {
    return (std::max(x.snr,x.folded_snr) > std::max(y.snr,y.folded_snr));
  }
};

class TimeSeriesFolder {
private:
  unsigned int size;
  unsigned int max_blocks;
  unsigned int max_threads;
    
public:
  TimeSeriesFolder(unsigned int size,
		   unsigned int max_blocks=MAX_BLOCKS,
		   unsigned int max_threads=MAX_THREADS)
    :size(size),max_blocks(max_blocks),max_threads(max_threads){}

  ~TimeSeriesFolder(){}
  

  void fold(DeviceTimeSeries<float>& input,FoldedSubints<float>& output, double period)
  {
    output.set_period(period);
    double tobs = input.get_nsamps() * input.get_tsamp();
    output.set_tobs((float)tobs);
    unsigned int nbins = output.get_nbins();
    unsigned int nints = output.get_nints();
    
    device_fold_timeseries(input.get_data(), output.get_data(),
			   input.get_nsamps(), nints,
			   period, input.get_tsamp(), nbins,
			   nints, nbins);

  }
};


class FoldOptimiser {
private:
  unsigned int nbins;
  unsigned int nints;
  
  //input buffer for folded device data
  cufftComplex* input_data;
  
  //phase shift parameters
  float* shift_mags;
  cufftComplex* shiftar;
  int nshifts;

  //collapsed subints after shifting
  cufftComplex* shifted_profiles;

  //final array of ntemplates * nbins * nshifts
  //this must be fft'd to get the optimisation
  cufftComplex* final_array_complex;
  float* final_array_float;
  
  //template parameters
  cufftComplex* templates;
  unsigned int ntemplates;

  //shifted_array
  cufftComplex* post_shift_input;

  //FFT plans
  CuFFTerC2C* forward_fft; //to make Fourier domain subints
  CuFFTerC2C* inverse_fft; //templated shifts to time domain (complex)
  CuFFTerC2C* inverse_fft_profile; //for selecting best profile

  //Kernel specifics
  unsigned int max_blocks;
  unsigned int max_threads;
 
  //Optimised profile/subints;
  cufftComplex* opt_subints_complex;
  float* opt_subints;
  cufftComplex* opt_prof_complex;
  float* opt_prof;
  
 
  void generate_templates(unsigned int step=1)
  {
    ntemplates = (int)(nbins/step - 1);
    unsigned int size = ntemplates*nbins;
    CuFFTerC2C template_ffter(nbins,ntemplates);
    Utils::device_malloc<cufftComplex>(&templates,size);   
    device_generate_template_array(templates, nbins, size, max_blocks, max_threads);
    template_ffter.execute(templates,templates,CUFFT_FORWARD);
  }
  
  void generate_shift_array(void)
  {
    nshifts = nbins;
    float* shift_mags_temp;
    Utils::host_malloc<float>(&shift_mags_temp,nshifts);
    Utils::device_malloc<float>(&shift_mags,nshifts);
    for (int ii=0;ii<nshifts;ii++)
      shift_mags_temp[ii] = ii-nshifts/2;
    
    Utils::h2dcpy<float>(shift_mags,shift_mags_temp,nshifts);
    Utils::host_free(shift_mags_temp);
    unsigned int size = nshifts*nbins*nints;
    //Utils::dump_device_buffer(shift_mags,nshifts,"shift_magnitudes.bin");

    Utils::device_malloc<cufftComplex>(&shiftar,size);
    device_generate_shift_array(shiftar, size, nbins, nints,
				nshifts, shift_mags,
				max_blocks, max_threads);
  }
    

  void calculate_sn(float* prof, int bin,
                    unsigned int width,
                    unsigned int nbins,
                    float* sn1, float* sn2)
  {
    int edge       = (int) (width*0.3 + 0.5);
    int width_by_2 = (int) (width/2.0 + 0.5);
    int ii,jj;
    std::vector<float> on_pulse;
    std::vector<float> off_pulse;
    std::vector<float> rprof;
    
    //centre the profile
    for (ii=0;ii<nbins;ii++){
      jj = (bin-nbins/2 + ii) % nbins;
      rprof.push_back(prof[jj]);
    }
    bin = nbins/2-1;

    int upper_edge = bin + (width_by_2+edge);
    int lower_edge = bin - (width_by_2+edge);
    
    for (ii=0;ii<nbins;ii++){
      if ((ii <= upper_edge) && (ii >= lower_edge))
	on_pulse.push_back(rprof[ii]);
      else
	off_pulse.push_back(rprof[ii]);
    }
    
    float on_mean  = std::accumulate(on_pulse.begin(),on_pulse.end(),0.0)/on_pulse.size();
    float off_mean = std::accumulate(off_pulse.begin(),off_pulse.end(),0.0)/off_pulse.size();
    float acc = 0;
    for (ii=0;ii<off_pulse.size();ii++)
      acc += std::pow(off_pulse[ii]-off_mean,2.0);
    float off_std = std::sqrt(acc/off_pulse.size());
    *sn1 = (on_mean-off_mean) * std::sqrt(width)/off_std;
    std::transform(&rprof[0], &rprof[0]+nbins, &rprof[0], std::bind2nd(std::minus<float>(),off_mean));
    std::transform(&rprof[0], &rprof[0]+nbins, &rprof[0], std::bind2nd(std::divides<float>(),off_std));
    *sn2 = std::accumulate(rprof.begin(),rprof.end(),0.0)/std::sqrt(width);
    if (*sn1>99999)
      *sn1 = 0.0;
    if (*sn2>99999)
      *sn2 = 0.0;
  }

public:
  FoldOptimiser(unsigned int nbins, unsigned int nints,
		unsigned int max_blocks=MAX_BLOCKS,
		unsigned int max_threads=MAX_THREADS)
    :nbins(nbins),nints(nints),
     max_blocks(max_blocks),
     max_threads(max_threads)
  {
    generate_templates();
    nshifts = nbins;
    generate_shift_array();
    Utils::device_malloc<cufftComplex>(&input_data,nbins*nints);
    Utils::device_malloc<cufftComplex>(&post_shift_input,nbins*nints*nshifts);
    Utils::device_malloc<cufftComplex>(&shifted_profiles,nbins*nshifts);
    Utils::device_malloc<cufftComplex>(&final_array_complex,nbins*nshifts*ntemplates);
    Utils::device_malloc<float>(&final_array_float,nbins*nshifts*ntemplates);
    Utils::host_malloc<cufftComplex>(&opt_prof_complex,nbins);
    Utils::host_malloc<cufftComplex>(&opt_subints_complex,nbins*nints);
    Utils::host_malloc<float>(&opt_prof,nbins);
    Utils::host_malloc<float>(&opt_subints,nbins*nints);
    forward_fft = new CuFFTerC2C(nbins,nints);
    inverse_fft = new CuFFTerC2C(nbins,nshifts*ntemplates);
    inverse_fft_profile = new CuFFTerC2C(nbins,1);
  }

  ~FoldOptimiser()
  {
    Utils::device_free(shift_mags);
    Utils::device_free(templates);
    Utils::device_free(shiftar);
    Utils::device_free(input_data);
    Utils::device_free(post_shift_input);
    Utils::device_free(final_array_complex);
    Utils::device_free(final_array_float);
    Utils::host_free(opt_prof);
    Utils::host_free(opt_prof_complex);
    Utils::host_free(opt_subints_complex);
    Utils::host_free(opt_subints);
    delete forward_fft;
    delete inverse_fft;
    delete inverse_fft_profile;
  }

  void dump_buffers(void){
    Utils::dump_host_buffer<float>(opt_prof,nbins,"opt_prof.bin");
    Utils::dump_device_buffer<cufftComplex>(post_shift_input,nbins*nints*nshifts,"shifted.bin");
    Utils::dump_device_buffer<float>(final_array_float,nshifts*nbins*ntemplates,"abs_templated.bin");
    Utils::dump_device_buffer<cufftComplex>(shifted_profiles,nshifts*nbins,"shifted_profiles.bin");
  }

  void optimise(FoldedSubints<float>& fold){
    if (nbins != fold.get_nbins() || nints != fold.get_nints())
      ErrorChecker::throw_error("FoldedSubints instance has wrong dimensions");
    
    float* tmp = fold.get_data();

    device_real_to_complex(tmp,input_data,
			   nbins*nints,max_blocks,max_threads);

    //Utils::dump_device_buffer<cuComplex>(input_data,nbins*nints,"prefft.bin");

    forward_fft->execute(input_data,input_data,CUFFT_FORWARD);

    //Utils::dump_device_buffer<cuComplex>(input_data,nbins*nints,"preshift.bin");

    device_multiply_by_shift(input_data, post_shift_input,
			     shiftar, nbins*nints*nshifts,
			     nbins*nints, max_blocks, max_threads);
    //Utils::dump_device_buffer<cuComplex>(post_shift_input,nbins*nints*nshifts,"shifted.bin");

    device_collapse_subints(post_shift_input,shifted_profiles,nbins,
			    nints,nbins*nshifts,max_blocks,max_threads);
    
    ///Utils::dump_device_buffer<cuComplex>(shifted_profiles,nbins*nshifts,"pretemplate.bin");

    //template normalisation is too steep

    device_multiply_by_templates(shifted_profiles, final_array_complex, templates,
				 nbins, nshifts, nshifts*nbins*ntemplates,
				 1,max_blocks,max_threads);

    //Utils::dump_device_buffer<cuComplex>(final_array_complex,ntemplates*nbins*nshifts,"posttemplate.bin");

    inverse_fft->execute(final_array_complex,final_array_complex,CUFFT_INVERSE);

    //Utils::dump_device_buffer<cuComplex>(final_array_complex,nshifts*nbins*ntemplates,"preabs.bin");

    device_get_absolute_value(final_array_complex,final_array_float,
			      nshifts*nbins*ntemplates,
			      max_blocks,max_threads);
    
    //Utils::dump_device_buffer<float>(final_array_float,nshifts*nbins*ntemplates,"pdmp.bin");

    int argmax = device_argmax(final_array_float,nshifts*nbins*ntemplates);
    unsigned int opt_template = argmax/(nbins*nshifts);
    int opt_bin = argmax%nbins-opt_template/2;
    unsigned int opt_shift = argmax/nbins%nbins;
    cufftComplex* prof = shifted_profiles+nbins*opt_shift;

    cufftComplex* subs = post_shift_input+nbins*nints*opt_shift;
    forward_fft->execute(subs,input_data,CUFFT_INVERSE);
    Utils::d2hcpy<cufftComplex>(opt_subints_complex,input_data,nbins*nints);
    for (int ii=0; ii<nbins*nints; ii++)
      opt_subints[ii] = (float) opt_subints_complex[ii].x;    

    
    //Utils::dump_device_buffer<cufftComplex>(prof,nbins,"prof.bin");

    //printf("opt_shift: %d   opt_bin: %d   opt_width: %d\n",opt_shift,opt_bin,opt_template);

    inverse_fft_profile->execute(prof,prof,CUFFT_INVERSE);

    //Utils::dump_device_buffer<cufftComplex>(prof,nbins,"prof_td.bin");
    
    Utils::d2hcpy<cufftComplex>(opt_prof_complex,prof,nbins);
    
    for (int ii=0; ii<nbins; ii++)
      opt_prof[ii] = opt_prof_complex[ii].x;
    
    //Utils::dump_device_buffer<float>(opt_prof,nbins,"prof_real.bin");
    
    //up to here is good for narrow pulse widths 

    float sn1 = 0;
    float sn2 = 0;

    calculate_sn(opt_prof, opt_bin, opt_template, nbins, &sn1, &sn2);

    fold.set_opt_sn(std::max(sn1,sn2));
    double p = fold.get_period();
    float tobs = fold.get_tobs();

    /*
    char buf[80];
    sprintf(buf,"fold_%.9f_%d.bin\0",p,nbins);
    Utils::dump_host_buffer<float>(opt_subints,nbins*nints,std::string(buf));
    //printf("fold_%.9f_%d.bin w:%d   sn1:%f   sn2:%f\n",p,nbins,opt_template+1,sn1,sn2);
    
    sprintf(buf,"prof_%.9f_%d.bin\0",p,nbins);
    Utils::dump_host_buffer<float>(opt_prof,nbins,std::string(buf));
    //printf("prof_%.9f_%d.bin w:%d   sn1:%f   sn2:%f\n",p,nbins,opt_template+1,sn1,sn2);
    */

    fold.set_opt_prof(opt_prof,nbins);
    fold.set_opt_fold(opt_subints,nbins*nints);
    fold.set_opt_period( p*((((32.0-opt_shift)*p)/(nbins*tobs))+1) );
    fold.set_opt_width(opt_template+1);
    fold.set_opt_bin(opt_bin);
        
  }  
};

class MultiFolder {
private:
  std::vector<Candidate>& cands;
  DispersionTrials<unsigned int>& dm_trials;
  TimeDomainResampler resampler;
  unsigned int nsamps;
  float tsamp;
  std::map< unsigned int, std::vector<unsigned int> > dm_to_cand_map;
  FoldedSubints<float>* subints;
  FoldOptimiser* optimiser;
  float min_period;
  float max_period;
  bool use_progress_bar;
  ProgressBar* progress_bar;

  void fold_all_mapped(void){
    std::map<unsigned int, std::vector<unsigned int> >::iterator iter;
    ReusableDeviceTimeSeries<float,unsigned int> device_tim(nsamps);
    DeviceTimeSeries<float> d_tim_r(nsamps);
    Dereddener rednoise(nsamps/2+1);
    TimeDomainResampler resampler;
    SpectrumFormer former;
    float tobs = nsamps*tsamp;
    CuFFTerR2C r2cfft(nsamps);
    CuFFTerC2R c2rfft(nsamps);
    DeviceFourierSeries<cufftComplex> d_fseries(nsamps/2+1,1.0/tobs);
    DevicePowerSpectrum<float> pspec(d_fseries);
    int nbins = 64;
    int nints = 16;
    TimeSeriesFolder folder(nsamps);
    float period;
    int cand_idx;
    TimeSeries<unsigned int> h_tim;
    float mean,std,rms;

    if (use_progress_bar){
      printf("Folding and optimising candidates...\n");
      progress_bar->start();
    }
    for(iter = dm_to_cand_map.begin(); iter != dm_to_cand_map.end(); iter++)
      {
	if (use_progress_bar)
	  progress_bar->set_progress((float)std::distance(dm_to_cand_map.begin(),iter)/dm_to_cand_map.size());
	mean = std = rms = 0.0;
	
	h_tim = dm_trials[iter->first];
        device_tim.copy_from_host(h_tim);
	d_tim_r.set_tsamp(h_tim.get_tsamp());
	r2cfft.execute(device_tim.get_data(),d_fseries.get_data());
	former.form(d_fseries,pspec);
	rednoise.calculate_median(pspec);
	rednoise.deredden(d_fseries);
	c2rfft.execute(d_fseries.get_data(),device_tim.get_data());
        
	for(int ii=0;ii<iter->second.size();ii++)
          {
	    
            cand_idx = iter->second[ii];
            period = 1.0/cands[cand_idx].freq;
	    resampler.resample(device_tim,d_tim_r,nsamps,cands[cand_idx].acc);
	    folder.fold(d_tim_r,*subints,period);
	    optimiser->optimise(*subints);
	    cands[cand_idx].folded_snr = subints->get_opt_sn();
	    cands[cand_idx].set_fold(&subints->opt_fold[0],nbins,nints);
	    cands[cand_idx].opt_period = subints->get_opt_period();
	  }
      }
    if (use_progress_bar)
      progress_bar->stop();
  }

public:
  MultiFolder(std::vector<Candidate>& cands, DispersionTrials<unsigned int>& dm_trials)
    :cands(cands),dm_trials(dm_trials),use_progress_bar(false){
    nsamps = Utils::prev_power_of_two(dm_trials.get_nsamps());
    tsamp = dm_trials.get_tsamp();
    subints = new FoldedSubints<float>(64,16);
    optimiser = new FoldOptimiser (64,16);
    min_period = 0.001;
    max_period = 10.00;
  }

  void enable_progress_bar(void){
    progress_bar = new ProgressBar;
    use_progress_bar = true;
  }
  
  void fold_n(unsigned int n_to_fold){
    int count = std::min(n_to_fold,(unsigned int) cands.size());
    float p;
    for (int ii=0;ii<count;ii++){
      p = 1.0/cands[ii].freq;
      if (p>min_period && p<max_period)
	dm_to_cand_map[cands[ii].dm_idx].push_back(ii);
    }
    fold_all_mapped();
    std::sort(cands.begin(),cands.end(),less_than_key());
  }
  
  ~MultiFolder(){
    delete subints;
    delete optimiser;
    if (use_progress_bar)
      delete progress_bar;
  }
};
