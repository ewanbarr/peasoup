#pragma once
#include <kernels/defaults.h>
#include <kernels/kernels.h>
#include <utils/exceptions.hpp>
#include <utils/utils.hpp>
#include <utils/stats.hpp>
#include <data_types/folded.hpp>
#include <data_types/candidates.hpp>
#include <transforms/ffter.hpp>
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
  //float* rebin_buffer;
    
public:
  float* rebin_buffer;
  TimeSeriesFolder(unsigned int size,
		   unsigned int max_blocks=MAX_BLOCKS,
		   unsigned int max_threads=MAX_THREADS)
    :size(size),max_blocks(max_blocks),max_threads(max_threads)
  {
    cudaError_t error = cudaMalloc((void**)&rebin_buffer, sizeof(float)*size);
    ErrorChecker::check_cuda_error(error);
  }

  void fold(DeviceTimeSeries<float>& input,FoldedSubints<float>& output, float period)
  {
    output.set_period(period);
    output.set_tobs(input.get_nsamps()*input.get_tsamp());
    unsigned int nbins = output.get_nbins();
    unsigned int nints = output.get_nints();
    float tobs = input.get_nsamps() * input.get_tsamp();
    unsigned int rebinned_size = nbins*tobs/period;
    unsigned int nrots = rebinned_size/nbins;
    unsigned int nrots_per_subint = nrots/nints;
    
    device_rebin_time_series(input.get_data(),rebin_buffer,
			     period, input.get_tsamp(), 
			     input.get_nsamps(),rebinned_size,
			     nbins,max_blocks,max_threads);
   
    device_create_subints(rebin_buffer, output.get_data(), nbins,
			  nbins*nints, nrots_per_subint, 
			  max_blocks,max_threads);

    ErrorChecker::check_cuda_error(cudaDeviceSynchronize());
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
 
  //Optimised profile;
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
    //edge of pulse that will not be taken into account
    int edge = 1;
    if (width<=3)
      edge = 0;

    int ii;
    float* onpulse;
    float* offpulse;
    int start = bin-width/2;
    unsigned int op_width = nbins-width;
    unsigned int op_width_t = op_width-2*edge;
    unsigned int width_t = width-2*edge;
    Utils::host_malloc<float>(&onpulse,width_t);
    Utils::host_malloc<float>(&offpulse,op_width_t);
    
    //here we give one bins grace either side of the optimal width
    for (ii=0;ii<width_t;ii++)
      onpulse[ii] = prof[(start+ii+edge)%nbins];
    for(ii=0;ii<op_width_t;ii++)
      offpulse[ii] = prof[(start+ii+width+edge)%nbins];
    float on_mean  = std::accumulate(onpulse,onpulse+width_t,0)/width_t;
    float off_mean = std::accumulate(offpulse,offpulse+op_width_t,0)/op_width_t;
    float acc = 0;
    for (ii=0;ii<op_width_t;ii++)
      acc += std::pow(offpulse[ii]-off_mean,2.0);
    float off_std = std::sqrt(acc/op_width_t);
    *sn1 = (on_mean-off_mean) * std::sqrt(width)/off_std;
    std::transform(prof, prof+nbins, prof, std::bind2nd(std::minus<float>(),off_mean));
    std::transform(prof, prof+nbins, prof, std::bind2nd(std::divides<float>(),off_std));
    *sn2 = std::accumulate(prof,prof+nbins,0)/std::sqrt(width);
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
    Utils::host_malloc<float>(&opt_prof,nbins);
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

    device_real_to_complex(fold.get_data(),input_data,
			   nbins*nints,max_blocks,max_threads);

    forward_fft->execute(input_data,input_data,CUFFT_FORWARD);

    
    device_multiply_by_shift(input_data, post_shift_input,
			     shiftar, nbins*nints*nshifts,
			     nbins*nints, max_blocks, max_threads);
    
    device_collapse_subints(post_shift_input,shifted_profiles,nbins,
			    nints,nbins*nshifts,max_blocks,max_threads);
    
    device_multiply_by_templates(shifted_profiles, final_array_complex, templates,
				 nbins, nshifts, nshifts*nbins*ntemplates,
				 1,max_blocks,max_threads);
    
    inverse_fft->execute(final_array_complex,final_array_complex,CUFFT_INVERSE);
    
    device_get_absolute_value(final_array_complex,final_array_float,
			      nshifts*nbins*ntemplates,
			      max_blocks,max_threads);
    
    int argmax = device_argmax(final_array_float,nshifts*nbins*ntemplates);
    unsigned int opt_template = argmax/(nbins*nshifts);
    int opt_bin = argmax%nbins-opt_template/2;
    unsigned int opt_shift = argmax/nbins%nbins;
    cufftComplex* prof = shifted_profiles+nbins*opt_shift;
    inverse_fft_profile->execute(prof,prof,CUFFT_INVERSE);
    Utils::d2hcpy<cufftComplex>(opt_prof_complex,prof,nbins);
    
    for (int ii=0; ii<nbins; ii++)
      opt_prof[ii] = cuCabsf(opt_prof_complex[ii]);
    
    float sn1 = 0;
    float sn2 = 0;
    calculate_sn(opt_prof, opt_bin, opt_template+1, nbins, &sn1, &sn2);
    fold.set_opt_sn(std::max(sn1,sn2));
    
    float p = fold.get_period();
    float tobs = fold.get_tobs();
    fold.set_opt_period(p*(((opt_shift*p)/(nbins*tobs))+1));
    fold.set_opt_width(opt_template+1);
    fold.set_opt_bin(opt_bin);
        
  }  
};

class MultiFolder {
private:
  std::vector<Candidate>& cands;
  DispersionTrials<unsigned char>& dm_trials;
  TimeDomainResampler resampler;
  unsigned int nsamps;
  float tsamp;
  std::map< unsigned int, std::vector<unsigned int> > dm_to_cand_map;
  
  int compute_nbins(float period){
    float nsamps_per_rot = period/tsamp;
    int nbins;
    if (nsamps_per_rot>64*2.0)
      nbins = 64;
    else if (nsamps_per_rot>32*2.0)
      nbins = 32;
    else if (nsamps_per_rot>16*2.0)
      nbins = 16;
    else if (nsamps_per_rot>8*2.0)
      nbins = 8;
    else
      nbins = 0;
    return nbins;
  }
  
  int compute_nints(float period){
    float nrots = nsamps*tsamp/period;
    int nints;
    if (nrots<16)
      nints = (int) nrots;
    else
      nints = 16;
    return nints;
  }
  
  void fold_all_mapped(void){
    std::map<unsigned int, std::vector<unsigned int> >::iterator iter;
    ReusableDeviceTimeSeries<float,unsigned char> device_tim(nsamps);
    DeviceTimeSeries<float> d_tim_r(nsamps);
    
    TimeDomainResampler resampler;
    TimeSeriesFolder folder(nsamps);
    int nbins,nints;
    float period;
    int cand_idx;
    TimeSeries<unsigned char> h_tim;

    for(iter = dm_to_cand_map.begin(); iter != dm_to_cand_map.end(); iter++)
      {
	h_tim = dm_trials[iter->first];
        device_tim.copy_from_host(h_tim);
	d_tim_r.set_tsamp(h_tim.get_tsamp());

        for(int ii=0;ii<iter->second.size();ii++)
          {
            cand_idx = iter->second[ii];
            period = 1.0/cands[cand_idx].freq;
            if (period<0.001)
              continue;
            nints = compute_nints(period);
            nbins = compute_nbins(period);
            if (!nbins || !nints)
              continue;
	    FoldedSubints<float> folded_array(nbins,nints);
            FoldOptimiser optimiser(nbins,nints);
	    resampler.resample(device_tim,d_tim_r,nsamps,cands[cand_idx].acc);
            folder.fold(d_tim_r,folded_array,period);
	    optimiser.optimise(folded_array);
            cands[cand_idx].folded_snr = folded_array.get_opt_sn();
	    cands[cand_idx].opt_period = folded_array.get_opt_period();
	  }
      }
  }

public:
  MultiFolder(std::vector<Candidate>& cands, DispersionTrials<unsigned char>& dm_trials)
    :cands(cands),dm_trials(dm_trials){
    nsamps = dm_trials.get_nsamps();
    tsamp = dm_trials.get_tsamp();
  }
  
  void fold_n(unsigned int n_to_fold){
    int count = std::min(n_to_fold,(unsigned int) cands.size());
    for (int ii=0;ii<count;ii++){
      dm_to_cand_map[cands[ii].dm_idx].push_back(ii);
    }
    fold_all_mapped();
    std::sort(cands.begin(),cands.end(),less_than_key());
  }
};
