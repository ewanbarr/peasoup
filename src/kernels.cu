#include <iostream>
#include "cuda.h"
#include "cufft.h"
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/distance.h>
#include <thrust/copy.h>
#include <thrust/tuple.h>
#include <thrust/fill.h>
#include <kernels/defaults.h>
#include <kernels/kernels.h>
#include <utils/exceptions.hpp>
#include <utils/utils.hpp>
#include <thrust/adjacent_difference.h>

//--------------Harmonic summing----------------//

__global__ 
void harmonic_sum_kernel_generic(float *d_idata, float *d_odata,
				 int gulp_index, int size, int harmonic, 
				 float one_over_sqrt_harm)
{
  int Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index<size) 
    {
      d_odata[gulp_index+Index] = d_idata[gulp_index+Index];
      for(int i = 1; i < harmonic; i++)
        {
          d_odata[gulp_index+Index] += d_idata[int((i*(gulp_index+Index))/harmonic+0.5)];
        }
      d_odata[gulp_index+Index] = d_odata[gulp_index+Index] * one_over_sqrt_harm;
    }
  return;
}

void device_harmonic_sum(float* d_input_array, float* d_output_array,
			 int original_size, int harmonic, 
			 unsigned int max_blocks, unsigned int max_threads)
{
  int gulps;
  int gulp_counter;
  int gulp_index = 0;
  int gulp_size;
  int blocks = 0;
  float one_over_sqrt_harm = 1.0f/sqrt((float)harmonic);
  gulps = original_size/(max_blocks*max_threads)+1;
  for(gulp_counter = 0; gulp_counter<gulps; gulp_counter++)
    {
      if(gulp_counter<gulps-1)
        {
          gulp_size = max_blocks*max_threads;
        }
      else
        {
          gulp_size = original_size - gulp_counter*max_blocks*max_threads;
        }
      blocks = (gulp_size-1)/MAX_THREADS + 1;
      harmonic_sum_kernel_generic<<<blocks,max_threads>>>(d_input_array,d_output_array,
							  gulp_index,gulp_size,harmonic,
							  one_over_sqrt_harm);
      gulp_index = gulp_index + blocks*max_threads;
    }
  ErrorChecker::check_cuda_error();
  return;
}

//------------spectrum forming--------------//


//Bad terminology, this forms the amplitudes
__global__ 
void power_series_kernel(cufftComplex *d_idata,float* d_odata, int size)
{
  float* d_idata_float = (float*)d_idata;
  int Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index<size)
    {
      d_odata[Index] = sqrtf(d_idata_float[2*Index]*d_idata_float[2*Index]
			     + d_idata_float[2*Index+1]*d_idata_float[2*Index+1]);
    }
  return;
}

__global__ void bin_interbin_series_kernel(cufftComplex *d_idata,float* d_odata, int size)
{
  float* d_idata_float = (float*)d_idata;
  int Index = blockIdx.x * blockDim.x + threadIdx.x;
  float re_l =0.0;
  float im_l =0.0;
  if (Index>0 && Index<size) {
    re_l = d_idata_float[2*Index-2];
    im_l = d_idata_float[2*Index-1];
  }
  if(Index<size)
    {
      float re = d_idata_float[2*Index];
      float im = d_idata_float[2*Index+1];
      float ampsq = re*re+im*im;
      float ampsq_diff = 0.5*((re-re_l)*(re-re_l) +
                              (im-im_l)*(im-im_l));
      d_odata[Index] = sqrtf(max(ampsq,ampsq_diff));
    }
  return;
}

void device_form_power_series(cufftComplex* d_array_in, float* d_array_out,
			      int size, int way)
{
  int gulps;
  int gulp_counter;
  cufftComplex* gulp_in_ptr = d_array_in;
  float* gulp_out_ptr = d_array_out;
  
  int gulp_size;
  
  gulps = (size-1)/(MAX_BLOCKS*MAX_THREADS)+1;

  for(gulp_counter = 0; gulp_counter<gulps; gulp_counter++)
    {
      if(gulp_counter<gulps-1)
        {
          gulp_size = MAX_BLOCKS*MAX_THREADS;
        }
      else
        {
          gulp_size = size - gulp_counter*MAX_BLOCKS*MAX_THREADS;
        }
      if (way==0)
        power_series_kernel<<<MAX_BLOCKS,MAX_THREADS>>>(gulp_in_ptr,gulp_out_ptr,gulp_size);
      if (way==1){
        bin_interbin_series_kernel<<<MAX_BLOCKS,MAX_THREADS>>>(gulp_in_ptr,gulp_out_ptr,gulp_size);
      }
      gulp_in_ptr = gulp_in_ptr + MAX_BLOCKS*MAX_THREADS;
      gulp_out_ptr = gulp_out_ptr + MAX_BLOCKS*MAX_THREADS;
    }
  return;
}

//-----------------time domain resampling---------------//

inline __device__ unsigned long getAcceleratedIndex(double accel_fact, double size_by_2, unsigned long id){
  return __double2ull_rn(id - accel_fact*( ((id-size_by_2)*(id-size_by_2)) - (size_by_2*size_by_2)));
}

__global__ void resample_kernel(float* input_d,
				float* output_d,
				double accel_fact,
				unsigned long size,
				double size_by_2,
				unsigned long start_idx)
{
  unsigned long idx = threadIdx.x + blockIdx.x * blockDim.x + start_idx;
  if (idx>=size)
    return;
  unsigned long idx_read = getAcceleratedIndex(accel_fact,size_by_2,idx);
  output_d[idx] = input_d[idx_read];
}

/*
inline __device__ double getAcceleratedIndex(double accel_fact, double size_by_2, unsigned long id){
  return id - accel_fact*( ((id-size_by_2)*(id-size_by_2)) - (size_by_2*size_by_2));
}


//With interpolation
__global__ void resample_kernel(float* input_d,
                                float* output_d,
                                double accel_fact,
                                unsigned long size,
                                double size_by_2,
                                unsigned long start_idx)
{
  unsigned long idx = threadIdx.x + blockIdx.x * blockDim.x + start_idx;
  if (idx>=size-1)
    return;
  double idx_read_frac = getAcceleratedIndex(accel_fact,size_by_2,idx);
  unsigned long idx_read = __double2ull_rd(idx_read_frac);
  double frac = idx_read_frac-idx_read;
  output_d[idx] = input_d[idx_read]*(1.0-frac) + input_d[idx_read+1]*frac;
}

//Non cetralised stretch
__global__ void GPU_resample_kernel(float *d_idata,float* d_odata, int size, float acc, double tsamp)
{
  double c = 2.99792458e8;
  int Index = blockIdx.x * blockDim.x + threadIdx.x;
  if (Index>=size-1)
    return;
  if (acc<=0.0) { // +ve acceleration                                                                                                       
    double earthtime = Index * tsamp;
    double delta = 0.5 * fabs(acc) * earthtime * earthtime/c/tsamp;
    double ddelta = delta - (int) delta;
    d_odata[Index]=d_idata[Index+(int)delta]*(1.0-ddelta)+
      d_idata[Index+(int)delta+1]*ddelta;
  }
  if (acc>0.0){   // -ve acceleration                                                                                                            
    double earthtime = Index * tsamp;
    double delta = 0.5 * fabs(acc) * earthtime * earthtime/c/tsamp;
    double ddelta = delta - (int) delta;
    d_odata[Index] = d_idata[Index-(int)delta]*(ddelta)+
      d_idata[Index-(int)delta+1]*(1.0-ddelta);
  }
  return;
}
*/

void device_resample(float * d_idata, float * d_odata,
		     unsigned int size, float a, 
		     float tsamp, unsigned int max_threads,
		     unsigned int max_blocks)
{
  double accel_fact = ((a*tsamp) / (2 * 299792458.0));
  double size_by_2  = (double)size/2.0;
  BlockCalculator calc(size,max_blocks,max_threads);
  for (int ii=0;ii<calc.size();ii++)
    resample_kernel<<< calc[ii].blocks,max_threads >>>(d_idata, d_odata, 
						       accel_fact,
						       (unsigned long) size,
						       size_by_2,
						       (unsigned long) ii*max_threads*max_blocks);
  ErrorChecker::check_cuda_error();
}

/*
void device_resample(float * d_idata, float * d_odata,
                     unsigned int size, float a,
                     float tsamp, unsigned int max_threads,
                     unsigned int max_blocks)
{
  BlockCalculator calc(size,max_blocks,max_threads);
  for (int ii=0;ii<calc.size();ii++)
    GPU_resample_kernel<<< calc[ii].blocks,max_threads >>>(d_idata, d_odata,
							   (int) size, a, (double)tsamp);
  ErrorChecker::check_cuda_error();
}
*/
//------------------peak finding-----------------//
//defined here as (although Thrust based) requires CUDA functors

struct greater_than_threshold : thrust::unary_function<thrust::tuple<int,float>,bool>
{
  float threshold;
  __device__ bool operator()(thrust::tuple<int,float> t) { return thrust::get<1>(t) > threshold; }
  greater_than_threshold(float thresh):threshold(thresh){}
};

int device_find_peaks(int n, int start_index, float * d_dat,
	     float thresh, int * indexes, float * snrs)
{
  using thrust::tuple;
  using thrust::counting_iterator;
  using thrust::zip_iterator;
  // Wrap the device pointer to let Thrust know                              
  thrust::device_ptr<float> dptr_dat(d_dat + start_index);
  thrust::device_vector<int> d_index(n-start_index);
  thrust::device_vector<float> d_snrs(n-start_index);
  typedef thrust::device_vector<float>::iterator snr_iterator;
  typedef thrust::device_vector<int>::iterator indices_iterator;
  thrust::counting_iterator<int> iter(start_index);
  zip_iterator<tuple<counting_iterator<int>,thrust::device_ptr<float> > > zipped_iter = make_zip_iterator(make_tuple(iter,dptr_dat));
  zip_iterator<tuple<indices_iterator,snr_iterator> > zipped_out_iter = make_zip_iterator(make_tuple(d_index.begin(),d_snrs.begin()));
  int num_copied = thrust::copy_if(zipped_iter, zipped_iter+n-start_index,
				   zipped_out_iter,greater_than_threshold(thresh)) - zipped_out_iter;
  thrust::copy(d_index.begin(),d_index.begin()+num_copied,indexes);
  thrust::copy(d_snrs.begin(),d_snrs.begin()+num_copied,snrs);

  ErrorChecker::check_cuda_error();
  return(num_copied);
}

//------------------rednoise----------------//

template<typename T>
struct square {
    __host__ __device__ inline
    T operator()(const T& x) { return x*x; }
};

template<typename T>
float GPU_rms(T* d_collection,int nsamps, int min_bin)
{
  T rms_sum;
  float rms;

  using thrust::device_ptr;
  rms_sum = thrust::transform_reduce(device_ptr<T>(d_collection)+min_bin,
				     device_ptr<T>(d_collection)+nsamps,
				     square<T>(),T(0),thrust::plus<T>());
  rms = sqrt(float(rms_sum)/float(nsamps-min_bin));
  return rms;
}

template<typename T>
float GPU_mean(T* d_collection,int nsamps, int min_bin)
{
  float mean;
  T m_sum;

  using thrust::device_ptr;
  m_sum = thrust::reduce(device_ptr<T>(d_collection)+min_bin,
			 device_ptr<T>(d_collection)+nsamps);

  cudaThreadSynchronize();
  mean = float(m_sum)/float(nsamps-min_bin);

  return mean;
}

template float GPU_rms<float>(float*,int,int);
template float GPU_mean<float>(float*,int,int);

__global__
void normalisation_kernel(float*d_powers, float mean, float sigma, unsigned int size, unsigned int gulp_idx)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x + gulp_idx;
  if (idx>=size)
    return;
  float val = d_powers[idx];
  val-=mean;
  val/=sigma;
  d_powers[idx] = val;
}

void device_normalise(float* d_powers,
		      float mean,
		      float sigma,
		      unsigned int size,
		      unsigned int max_blocks,
		      unsigned int max_threads)
{
  BlockCalculator calc(size, max_blocks, max_threads);
  for (int ii=0;ii<calc.size();ii++)
    normalisation_kernel<<<calc[ii].blocks,max_threads>>>(d_powers,mean,sigma,size,
							  ii*max_blocks*max_threads);
  ErrorChecker::check_cuda_error();
}


//old normalisation routine used after a different
//rednoise algorithm was applied
void device_normalise_spectrum(int nsamp,
			       float* d_power_spectrum,
			       float* d_normalised_power_spectrum,
			       int min_bin,
			       float * sigma)
{
  float mean;
  float rms;
  float meansquares;
  
  if (*sigma==0.0) {
    mean = GPU_mean(d_power_spectrum,nsamp,min_bin);
    rms = GPU_rms(d_power_spectrum,nsamp,min_bin);
    meansquares = rms*rms;
    *sigma = sqrt(meansquares - (mean*mean));
  }
  
  thrust::transform(thrust::device_ptr<float>(d_power_spectrum),
                    thrust::device_ptr<float>(d_power_spectrum)+nsamp,
                    thrust::make_constant_iterator(*sigma),
                    thrust::device_ptr<float>(d_normalised_power_spectrum),
                    thrust::divides<float>());
  ErrorChecker::check_cuda_error();
}


//--------------Time series folder----------------//


__global__ 
void rebin_time_series_kernel(float* i_data, float* o_data,
			      unsigned int size, float tsamp,
			      float period, unsigned int nbins,
			      unsigned int gulp_idx) 
{ 
  int ii;
  float val=0;
  int count=0;
  int idx = blockIdx.x * blockDim.x + threadIdx.x + gulp_idx;
  if (idx>=size)
    return;
  int start_idx = __float2int_rn(idx*period/(tsamp*nbins));
  int end_idx = __float2int_rn((idx+1)*period/(tsamp*nbins));
  if (start_idx==end_idx){
    o_data[idx] = i_data[start_idx];
  } else {
    for (ii=start_idx;ii<=end_idx;ii++)
      {
        val+=i_data[ii];
        count++;
      }
    o_data[idx] = val/count;
  }
}


__global__ 
void create_subints_kernel(float* input, float* output,
			   unsigned int nbins,
			   unsigned int output_size,
			   unsigned int nrots_per_subint)
{
  int ii;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx>=output_size)
    return;
  unsigned int bin = idx%nbins;
  unsigned int subint = idx/nbins;
  unsigned int offset = subint*nrots_per_subint*nbins;
  float val = 0;
  for (ii=0;ii<nrots_per_subint;ii++)
    {
      val+=input[(ii*nbins)+bin+offset];
    }
  output[idx] = val/nrots_per_subint;
}

void device_create_subints(float* input, float* output,
			   unsigned int nbins,
                           unsigned int output_size,
                           unsigned int nrots_per_subint,
			   unsigned int max_blocks,
			   unsigned int max_threads)
{
  unsigned int nblocks = output_size/max_threads + 1;
  create_subints_kernel<<<nblocks,max_threads>>>(input,output,nbins,
						 output_size,
						 nrots_per_subint);
}


void device_rebin_time_series(float* input, float* output,
			      float period, float tsamp,
			      unsigned int in_size, unsigned int out_size,
			      unsigned int nbins,
			      unsigned int max_blocks, unsigned int max_threads)
{
  unsigned int gulps;
  unsigned int gulp_counter;
  unsigned int gulp_index = 0;
  unsigned int gulp_size;
  unsigned int blocks = 0;
   
  gulps = out_size/(max_blocks*max_threads)+1;
  for (gulp_counter = 0; gulp_counter<gulps; gulp_counter++)
    {
      if (gulp_counter<gulps-1)
	{
	  gulp_size = max_blocks*max_threads;
	  blocks = max_blocks;
	}
      else
	{
	  gulp_size = out_size-gulp_counter*max_blocks*max_threads;
	  blocks = (gulp_size-1)/max_threads+1;
	}
      gulp_index = gulp_counter*blocks*max_threads;
      rebin_time_series_kernel<<<blocks,max_threads>>>(input,output,out_size,
						       tsamp,period,nbins,
						       gulp_index);
    }
}

//--------------FoldOptimiser------------//
  
__device__ inline cuComplex cuCexpf(cuComplex z)
{
  cuComplex res;
  float t = expf(z.x);
  sincosf(z.y, &res.y, &res.x);
  res.x *= t;
  res.y *= t;
  return res;
}

__global__
void shift_array_generator_kernel(cuComplex* shift_ar, unsigned int shift_ar_size,
				  unsigned int nbins, unsigned int nints,
				  unsigned int nshift, float* shifts,
				  float two_pi)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= shift_ar_size)
    return;
  float subint = idx/nbins%nints;
  unsigned int shift_idx = idx/(nbins*nints);
  unsigned int bin = idx%nbins;
  float shift = subint/nints * shifts[shift_idx];
  float ramp = bin*two_pi/nbins;
  if (bin>nbins/2)
    ramp-=two_pi;
  //printf("%f %f %f\n",ramp,shift,ramp*shift);
  cuComplex tmp1 = make_cuComplex(0.0,-1*ramp*shift);
  cuComplex tmp2 = cuCexpf(tmp1);
  shift_ar[idx] = tmp2;
}

__global__
void template_generator_kernel(cuComplex* templates, unsigned int nbins, unsigned int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx>=size)
    return;
  unsigned int bin = idx%nbins;
  unsigned int template_idx = idx/nbins;
  float val = (bin<=template_idx);
  templates[idx] = make_cuComplex(val,0.0);
}

__global__
void multiply_by_shift_kernel(cuComplex* input, cuComplex* output,
			      cuComplex* shift_array, unsigned int nbins_by_nints,
			      unsigned int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x; //NEED DATA IDX IF MULTIPLE GULPS
  if (idx>=size)
    return;
  unsigned int in_idx = idx%(nbins_by_nints);
  output[idx] = cuCmulf(input[in_idx],shift_array[idx]);
}

__global__
void collapse_subints_kernel(cuComplex* input, cuComplex* output, 
			     unsigned int nbins, unsigned int nints, 
			     unsigned int nbins_by_nints, unsigned int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx>=size)
    return;
  unsigned int bin = idx%nbins;
  unsigned int fold = idx/nbins;
  unsigned int in_idx = (fold*nbins_by_nints)+bin;
  cuComplex val =  make_cuComplex(0.0,0.0);
  for (int ii=0;ii<nints;ii++)
    val = cuCaddf(val,input[in_idx+ii*nbins]);  
  output[idx] = val;
}

__global__
void multiply_by_template_kernel(cuComplex* input, cuComplex* output,
				 cuComplex* templates, unsigned int nbins,
				 unsigned int nshifts, unsigned int nbins_by_nshifts,
				 unsigned int size, unsigned int step)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx>=size)
    return;
  unsigned int template_idx = idx/nbins_by_nshifts;
  unsigned int bin = idx%nbins;
  unsigned int shift = idx%nbins_by_nshifts;
  float width = (template_idx+1.0);
  cuComplex normalisation_factor = make_cuComplex(sqrtf(width),0.0);
  if (bin==0)
    output[idx] = make_cuComplex(0.0,0.0);
  else
    output[idx] = cuCdivf(cuCmulf(input[shift],templates[template_idx*nbins+bin]),normalisation_factor);
}

__global__
void cuCabsf_kernel(cuComplex* input, float* output, unsigned int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx>=size)
    return;
  output[idx] = cuCabsf(input[idx]);
}

__global__
void real_to_complex_kernel(float* input, cuComplex* output, unsigned int size) 
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx>=size)
    return;
  output[idx] = make_cuComplex(input[idx],0.0);
}

unsigned int device_argmax(float* input, unsigned int size)
{
  thrust::device_ptr<float> ptr(input);
  thrust::device_ptr<float> max_elem = thrust::max_element(ptr,ptr+size);
  return thrust::distance(ptr,max_elem);
}

void device_real_to_complex(float* input, cuComplex* output, unsigned int size, 
			    unsigned int max_blocks, unsigned int max_threads)
{
  BlockCalculator calc(size,max_blocks,max_threads);
  for (int ii=0;ii<calc.size();ii++)
    real_to_complex_kernel<<<calc[ii].blocks,max_threads>>>(input,output,size);
  ErrorChecker::check_cuda_error();
  return;
}


void device_get_absolute_value(cuComplex* input, float* output, unsigned int size,
			       unsigned int max_blocks, unsigned int max_threads)
{
  BlockCalculator calc(size,max_blocks,max_threads);
  for (int ii=0;ii<calc.size();ii++)
    cuCabsf_kernel<<<calc[ii].blocks,max_threads>>>(input,output,size);
  ErrorChecker::check_cuda_error();
  return;
}

void device_generate_shift_array(cuComplex* shifted_ar,
                                 unsigned int shifted_ar_size,
                                 unsigned int nbins, unsigned int nints,
                                 unsigned int nshift, float* shifts,
                                 unsigned int max_blocks, unsigned int max_threads)
{
  float two_pi = 2*3.14159265359;
  BlockCalculator calc(shifted_ar_size,max_blocks,max_threads);
  for (int ii=0;ii<calc.size();ii++)
    shift_array_generator_kernel<<<calc[ii].blocks,max_threads>>>(shifted_ar, shifted_ar_size, nbins,
								  nints, nshift, shifts, two_pi);
  ErrorChecker::check_cuda_error();
  return;
}

void device_generate_template_array(cuComplex* templates, unsigned int nbins, 
				    unsigned int size, unsigned int max_blocks,
				    unsigned int max_threads)
{
  BlockCalculator calc(size,max_blocks,max_threads);
  for (int ii=0;ii<calc.size();ii++){
    template_generator_kernel<<<calc[ii].blocks,max_threads>>>(templates, nbins, size);
    cudaDeviceSynchronize();
    ErrorChecker::check_cuda_error();
  }
  ErrorChecker::check_cuda_error();
  return;
}

void device_multiply_by_shift(cuComplex* input, cuComplex* output,
                              cuComplex* shift_array, unsigned int size,
			      unsigned int nbins_by_nints,
			      unsigned int max_blocks, unsigned int max_threads)
{
  BlockCalculator calc(size, max_blocks, max_threads);
  for (int ii=0;ii<calc.size();ii++){
    multiply_by_shift_kernel<<<calc[ii].blocks,max_threads>>>(input,output,shift_array,
							      nbins_by_nints,size);
  }
  ErrorChecker::check_cuda_error();
  return;
}

void device_collapse_subints(cuComplex* input, cuComplex* output,
			     unsigned int nbins, unsigned int nints,
			     unsigned int size, unsigned int max_blocks, 
			     unsigned int max_threads)
{
  unsigned int nbins_by_nints = nbins*nints;
  BlockCalculator calc(size, max_blocks, max_threads);
  for (int ii=0;ii<calc.size();ii++){
    collapse_subints_kernel<<<calc[ii].blocks,max_threads>>>(input,output,nbins,
							     nints,nbins_by_nints,size);
  }
  ErrorChecker::check_cuda_error();
  return;
}
  
void device_multiply_by_templates(cuComplex* input, cuComplex* output,
				  cuComplex* templates, unsigned int nbins,
				  unsigned int nshifts,
				  unsigned int size, unsigned int step,
				  unsigned int max_blocks, unsigned int max_threads)
{
  unsigned int nbins_by_nshifts = nbins*nshifts;
  BlockCalculator calc(size, max_blocks, max_threads);
  for (int ii=0;ii<calc.size();ii++){
    multiply_by_template_kernel<<<calc[ii].blocks,max_threads>>>(input,output,templates,
								 nbins,nshifts,nbins_by_nshifts,
								 size,step);
  }
  ErrorChecker::check_cuda_error();
  return;
}

//--------------Rednoise stuff--------------//

//Ben Barsdells median scrunching algorithm from Heimdall
/*
  Note: The implementations of median3-5 here can be derived from
          'sorting networks'.
*/

inline __host__ __device__
float median3(float a, float b, float c) {
	return a < b ? b < c ? b
	                      : a < c ? c : a
	             : a < c ? a
	                     : b < c ? c : b;
}
inline __host__ __device__
float median4(float a, float b, float c, float d) {
	return a < c ? b < d ? a < b ? c < d ? 0.5f*(b+c) : 0.5f*(b+d)
	                             : c < d ? 0.5f*(a+c) : 0.5f*(a+d)
	                     : a < d ? c < b ? 0.5f*(d+c) : 0.5f*(b+d)
	                             : c < b ? 0.5f*(a+c) : 0.5f*(a+b)
	             : b < d ? c < b ? a < d ? 0.5f*(b+a) : 0.5f*(b+d)
	                             : a < d ? 0.5f*(a+c) : 0.5f*(c+d)
	                     : c < d ? a < b ? 0.5f*(d+a) : 0.5f*(b+d)
	                             : a < b ? 0.5f*(a+c) : 0.5f*(c+b);
}
inline __host__ __device__
float median5(float a, float b, float c, float d, float e) {
	// Note: This wicked code is by 'DRBlaise' and was found here:
	//         http://stackoverflow.com/a/2117018
	return b < a ? d < c ? b < d ? a < e ? a < d ? e < d ? e : d
                                                 : c < a ? c : a
                                         : e < d ? a < d ? a : d
                                                 : c < e ? c : e
                                 : c < e ? b < c ? a < c ? a : c
                                                 : e < b ? e : b
                                         : b < e ? a < e ? a : e
                                                 : c < b ? c : b
                         : b < c ? a < e ? a < c ? e < c ? e : c
                                                 : d < a ? d : a
                                         : e < c ? a < c ? a : c
                                                 : d < e ? d : e
                                 : d < e ? b < d ? a < d ? a : d
                                                 : e < b ? e : b
                                         : b < e ? a < e ? a : e
                                                 : d < b ? d : b
	         : d < c ? a < d ? b < e ? b < d ? e < d ? e : d
                                                 : c < b ? c : b
                                         : e < d ? b < d ? b : d
                                                 : c < e ? c : e
                                 : c < e ? a < c ? b < c ? b : c
                                                 : e < a ? e : a
                                         : a < e ? b < e ? b : e
                                                 : c < a ? c : a
                         : a < c ? b < e ? b < c ? e < c ? e : c
                                                 : d < b ? d : b
                                         : e < c ? b < c ? b : c
                                                 : d < e ? d : e
                                 : d < e ? a < d ? b < d ? b : d
                                                 : e < a ? e : a
                                         : a < e ? b < e ? b : e
	                                         : d < a ? d : a;
}

struct median_scrunch5_kernel
	: public thrust::unary_function<hd_float,hd_float> {
	const hd_float* in;
	median_scrunch5_kernel(const hd_float* in_)
		: in(in_) {}
	inline __host__ __device__
	hd_float operator()(unsigned int i) const {
		hd_float a = in[5*i+0];
		hd_float b = in[5*i+1];
		hd_float c = in[5*i+2];
		hd_float d = in[5*i+3];
		hd_float e = in[5*i+4];
		return median5(a, b, c, d, e);
	}
};

hd_error median_scrunch5(const hd_float* d_in,
                         hd_size         count,
                         hd_float*       d_out)
{
	thrust::device_ptr<const hd_float> d_in_begin(d_in);
	thrust::device_ptr<hd_float>       d_out_begin(d_out);
	
	if( count == 1 ) {
		*d_out_begin = d_in_begin[0];
	}
	else if( count == 2 ) {
		*d_out_begin = 0.5f*(d_in_begin[0] + d_in_begin[1]);
	}
	else if( count == 3 ) {
		*d_out_begin = median3(d_in_begin[0],
		                       d_in_begin[1],
		                       d_in_begin[2]);
	}
	else if( count == 4 ) {
		*d_out_begin = median4(d_in_begin[0],
		                       d_in_begin[1],
		                       d_in_begin[2],
		                       d_in_begin[3]);
	}
	else {
		// Note: Truncating here is necessary
		hd_size out_count = count / 5;
		using thrust::make_counting_iterator;
		thrust::transform(make_counting_iterator<unsigned int>(0),
		                  make_counting_iterator<unsigned int>(out_count),
		                  d_out_begin,
		                  median_scrunch5_kernel(d_in));
	}
	return HD_NO_ERROR;
}

struct linear_stretch_functor
	: public thrust::unary_function<hd_float,hd_float> {
	const hd_float* in;
	hd_float        step;
	linear_stretch_functor(const hd_float* in_,
	                       hd_size in_count, hd_size out_count)
		: in(in_), step(hd_float(in_count-1)/(out_count-1)) {}
	inline __host__ __device__
	hd_float operator()(unsigned int i) const {
		hd_float     x = i * step;
		unsigned int j = x;
		return in[j] + ((x-j > 1e-5f) ? (x-j)*(in[j+1]-in[j]) : 0.f);
	}
};

hd_error linear_stretch(const hd_float* d_in,
                        hd_size         in_count,
                        hd_float*       d_out,
                        hd_size         out_count)
{
	using thrust::make_counting_iterator;
	thrust::device_ptr<hd_float> d_out_begin(d_out);
	
	thrust::transform(make_counting_iterator<unsigned int>(0),
	                  make_counting_iterator<unsigned int>(out_count),
	                  d_out_begin,
	                  linear_stretch_functor(d_in, in_count, out_count));
	return HD_NO_ERROR;
}

__global__ 
void divide_c_by_f_kernel(cuComplex* c, float* f, unsigned int size, unsigned int gulp_idx)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x + gulp_idx;
  if (idx>=size)
    return;
  if (idx<5)
    c[idx] = make_cuComplex(0.0,0.0);
  else
    c[idx] = cuCdivf(c[idx],make_cuComplex(f[idx],0.0));
}

void device_divide_c_by_f(cuComplex* c, float* f, unsigned int size,
			    unsigned int max_blocks, unsigned int max_threads)
{
  BlockCalculator calc(size, max_blocks, max_threads);
  for (int ii=0;ii<calc.size();ii++){
    divide_c_by_f_kernel<<<calc[ii].blocks,max_threads>>>(c,f,size,ii*max_threads*max_blocks);
  }
  ErrorChecker::check_cuda_error();
  return;
}

__global__
void zap_birdies_kernel(cuComplex* fseries, float* birdies, float* widths,
			float bin_width, unsigned int size,
			unsigned int fseries_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx>=size)
    return;
  int ii;
  float freq = birdies[idx];
  float width = widths[idx];
  int low_bin = __float2int_rd((freq-width)/bin_width);
  int high_bin = __float2int_ru((freq+width)/bin_width);
  //printf("%f  %f  %f   %d  %d  %d\n",freq,width,bin_width,idx,low_bin,high_bin);

  if (low_bin<0)
    low_bin = 0;
  if (low_bin>=fseries_size)
    return;
  if (high_bin>=fseries_size)
    high_bin = fseries_size-1;
  for (ii=low_bin;ii<high_bin;ii++)
    fseries[ii] = make_cuComplex(1.0,0.0);
}

void device_zap_birdies(cuComplex* fseries, float* d_birdies, float* d_widths, float bin_width,
			unsigned int birdies_size, unsigned int fseries_size,
			unsigned int max_blocks, unsigned int max_threads)
{
  BlockCalculator calc(birdies_size, max_blocks, max_threads);
  for (int ii=0;ii<calc.size();ii++)
    zap_birdies_kernel<<<calc[ii].blocks,max_threads>>>(fseries,d_birdies,d_widths,bin_width,birdies_size,fseries_size);
  ErrorChecker::check_cuda_error();
  return;
}
