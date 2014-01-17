#include <iostream>
#include "cuda.h"
#include "cufft.h"
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
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

//Could be optimised with registers
/*
__global__ 
void harmonic_sum_kernel_generic(float *d_idata, float *d_odata,
				 int gulp_index, size_t size, int harmonic, 
				 float one_over_sqrt_harm)
{
  float val;
  int idx = blockIdx.x * blockDim.x + threadIdx.x + gulp_index;
  if(idx<size) 
    {
      val = d_idata[idx];
      for(int i = 1; i < harmonic; i++)
	val += d_idata[int((i*idx)/harmonic+0.5)];
      d_odata[idx] = val * one_over_sqrt_harm;
    }
  return;
}
*/

__global__
void harmonic_sum_kernel(float *d_idata, float *d_odata,
			 int gulp_index, size_t size, int harmonic,
			 float one_over_sqrt_harm)
{
  for( int idx = blockIdx.x*blockDim.x + threadIdx.x ; idx < size ; idx += blockDim.x*gridDim.x )
    {
      double idx_div_harmonic = (double) idx / harmonic, i_as_dbl = 1.0;
      
      float val = d_idata[idx];
      for( int i = 1 ; i < harmonic ; ++i, i_as_dbl += 1.0 )
	val += d_idata[(int) (i_as_dbl*idx_div_harmonic + 0.5)];
      d_odata[idx] = val*one_over_sqrt_harm;
  }
  return;
}

/* Old routine
void device_harmonic_sum(float* d_input_array, float* d_output_array,
                         size_t size, int harmonic,
                         unsigned int max_blocks, unsigned int max_threads)
{
  float one_over_sqrt_harm = 1.0f/sqrt((float)harmonic);
  BlockCalculator calc(size, max_blocks, max_threads);
  for (int ii=0;ii<calc.size();ii++){
    harmonic_sum_kernel_generic<<<calc[ii].blocks,max_threads>>>
      (d_input_array,d_output_array,calc[ii].data_idx,
       size,harmonic,one_over_sqrt_harm);
  }
  ErrorChecker::check_cuda_error("Error from device_harmonic_sum");
}
*/

void device_harmonic_sum(float* d_input_array, float* d_output_array,
			   size_t size, int harmonic,
			   unsigned int max_blocks, unsigned int max_threads)
{
  float one_over_sqrt_harm = 1.0f/sqrt((float)harmonic);
  BlockCalculator calc(size, max_blocks, max_threads);
  harmonic_sum_kernel<<<calc[0].blocks,max_threads>>>
    (d_input_array,d_output_array,0,
     size,harmonic,one_over_sqrt_harm);
  ErrorChecker::check_cuda_error("Error from device_harmonic_sum");
}

//------------spectrum forming--------------//


//Could be optimised with shared memory

__global__ 
void power_series_kernel(cufftComplex *d_idata, float* d_odata, 
			 size_t size, size_t gulp_index)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x + gulp_index;
  cufftComplex& x = d_idata[idx];
  if(idx<size)
    {
      float z = x.x*x.x+x.y*x.y;
      d_odata[idx] = z*rsqrtf(z);
    }
  return;
}

//Could be optimised with shared memory

__global__ void bin_interbin_series_kernel(cufftComplex *d_idata,float* d_odata, 
					   size_t size, size_t gulp_index)
{
  float* d_idata_float = (float*)d_idata;
  int idx = blockIdx.x * blockDim.x + threadIdx.x + gulp_index;
  float re_l =0.0;
  float im_l =0.0;
  if (idx>0 && idx<size) {
    re_l = d_idata_float[2*idx-2];
    im_l = d_idata_float[2*idx-1];
  }
  if(idx<size)
    {
      float re = d_idata_float[2*idx];
      float im = d_idata_float[2*idx+1];
      float ampsq = re*re+im*im;
      float ampsq_diff = 0.5*((re-re_l)*(re-re_l) +
                              (im-im_l)*(im-im_l));
      d_odata[idx] = sqrtf(max(ampsq,ampsq_diff));
    }
  return;
}

 /*
__global__ void bin_interbin_series_kernel(cufftComplex *d_idata,float* d_odata, int size)
{
  int idx = blockIdx.x * (blockDim.x-1) + threadIdx.x;
  
  if (idx>=size-1)
    return;
    
  extern __shared__ cufftComplex s[];

  //blockIdx accounts for backshift by 1 sample to keep single write coalesence
  
  if (idx!=0)
    s[threadIdx.x] = d_idata[idx-1];
  else
    s[threadIdx.x] = make_cuComplex(0.0,0.0);
  __syncthreads();
  
  if (threadIdx.x+1 == blockDim.x)
    return;

  cufftComplex x = s[threadIdx.x+1];
  cufftComplex y = s[threadIdx.x];
  float ampsq = x.x*x.x+x.y*x.y;
  float ampsq_diff = 0.5*((x.x-y.x)*(x.x-y.x) +
			  (x.y-y.y)*(x.y-y.y));
  float val = max(ampsq,ampsq_diff);
  d_odata[idx] = val*rsqrtf(val);
  
  return;
}
 */

void device_form_power_series(cufftComplex* d_array_in, 
			      float* d_array_out,
			      size_t size, int way,
			      unsigned int max_blocks,
			      unsigned int max_threads)
{
  BlockCalculator calc(size,max_blocks,max_threads);
  for (int ii=0;ii<calc.size();ii++){
    if (way == 1)
      bin_interbin_series_kernel<<<calc[ii].blocks,max_threads>>>
        (d_array_in, d_array_out, size, calc[ii].data_idx);
    else
      power_series_kernel<<<calc[ii].blocks,max_threads>>>
        (d_array_in, d_array_out, size, calc[ii].data_idx);
  }
  ErrorChecker::check_cuda_error("Error from device_form_power_series");
  return;
}

//-----------------time domain resampling---------------//

inline __device__ unsigned long getAcceleratedIndex(double accel_fact, double size_by_2,
						    unsigned long id){
  return __double2ull_rn(id + accel_fact*( ((id-size_by_2)*(id-size_by_2)) - (size_by_2*size_by_2)));
}

__global__ void resample_kernel(float* input_d,
				float* output_d,
				double accel_fact,
				size_t size,
				double size_by_2,
				size_t start_idx)
{
  unsigned long idx = threadIdx.x + blockIdx.x * blockDim.x + start_idx;
  if (idx>=size)
    return;
  unsigned long idx_read = getAcceleratedIndex(accel_fact,size_by_2,idx);
  output_d[idx] = input_d[idx_read];
}

void device_resample(float * d_idata, float * d_odata,
		     size_t size, float a, 
		     float tsamp, unsigned int max_threads,
		     unsigned int max_blocks)
{
  double accel_fact = ((a*tsamp) / (2 * 299792458.0));
  double size_by_2  = (double)size/2.0;
  BlockCalculator calc(size,max_blocks,max_threads);
  for (int ii=0;ii<calc.size();ii++)
    resample_kernel<<< calc[ii].blocks,max_threads >>>(d_idata, d_odata, 
						       accel_fact,
						       size,
						       size_by_2,
						       calc[ii].data_idx);
  ErrorChecker::check_cuda_error("Error from device_resample");
}

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


  ErrorChecker::check_cuda_error("Error from device_find_peaks;");
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


template<typename T>
void GPU_fill(T* start, T* end, T value){
  thrust::device_ptr<T> ar_start(start);
  thrust::device_ptr<T> ar_end(end);
  thrust::fill(ar_start,ar_end,value);
  ErrorChecker::check_cuda_error("Error in GPU_fill");
}

template void GPU_fill<float>(float*, float*, float);
template float GPU_rms<float>(float*,int,int);
template float GPU_mean<float>(float*,int,int);

__global__
void normalisation_kernel(float*d_powers, float mean, float sigma, 
			  size_t size, size_t gulp_idx)
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
							  calc[ii].data_idx);
  ErrorChecker::check_cuda_error("Error from device_normalise");
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
  ErrorChecker::check_cuda_error("Error from device_normalise_spectrum");
}


//--------------Time series folder----------------//

/*
__global__ void reindexing_kernel(float* i_data, float* o_data, size_t size, 
				  size_t gulp_idx, double tsamp_by_period, 
				  float nrots_per_subint)
{
  int nbins = 64;
  __shared__ int idx_buffer_s[MAX_THREADS];
  __shared__ float i_data_s[MAX_THREADS];
  int idx = blockIdx.x * blockDim.x + threadIdx.x + gulp_idx;  
  if (idx>=size)
    return;
  int bin = __float2int_rd(idx*nbins*tsamp_by_period)%nbins;
  int subint = __float2int_rd((tsamp_by_period*idx)/nrots_per_subint);
  int oidx = subint*nbins+bin;
  idx_buffer_s[threadIdx.x] = oidx;
  i_data_s[threadIdx.x] = i_data[idx];
  __syncthreads();
  float val = 0.0;
  int count = 0;
  for (int ii=0;ii<MAX_THREADS;ii++){
    if (threadIdx.x==idx_buffer_s[ii]){
      val += i_data_s[ii];
      count++;
    }
  }
  if (count!=0)
    o_data[threadIdx.x] += val/count;
}
*/

__global__ void reindexing_kernel(size_t size, int* new_indexes, int nbins,
				   size_t gulp_idx, double tsamp_by_period,
				   double nrots_per_subint)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x + gulp_idx;
  if (idx>=size)
    return;

  int subint = __double2int_rd((tsamp_by_period*idx)/nrots_per_subint);
  int bin = __double2int_rd(idx*nbins*tsamp_by_period)%nbins;
  //int bin = __float2int_rd(idx*nbins*tsamp_by_period)%nbins;
  //int subint = __float2int_rd((tsamp_by_period*idx)/nrots_per_subint);
  int oidx = subint*nbins+bin;
  new_indexes[idx] = oidx;
}

__global__ void find_switch_points_kernel(int* indexes, size_t size, int* switch_ar, size_t gulp_idx)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x + gulp_idx;
  
  if (idx>=size-2)
    return;
  
  if (idx == 0){
    switch_ar[0] = 0;
    return;
  }
  
  if (indexes[idx] != indexes[idx+1]){
    switch_ar[indexes[idx+1]] = idx+1;
  } 
  
}

__global__ void bin_timeseries_kernel(float* tim, float* output, int* switch_points, unsigned int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx>=size)
    return;
  int end;
  int start = switch_points[idx];
  
  if (idx==size-1)
    end = size;
  else
    end = switch_points[idx+1];
  
  int count = end-start;
  float val = 0.0;
  for (int ii=start;ii<end;ii++)
    val += tim[ii];
  if (count!=0)
    output[idx] = val/count;
  else
    output[idx] = 0;
}

void device_fold_timeseries(float* tim_buffer, float* sorted_tim_buffer,
			    float* subints_buffer, int* new_indexes_buffer,
			    size_t size, unsigned int nbins, unsigned int nints,
			    double period, double tsamp, unsigned int max_blocks,
			    unsigned int max_threads)
{
  int ii;
  int* switch_points;
  Utils::device_malloc<int>(&switch_points,nbins*nints);
  Utils::d2dcpy(sorted_tim_buffer,tim_buffer,size);
  double tsamp_by_period = tsamp/period;
  double nrots_per_subint = (size*tsamp)/period/nints;
  
  BlockCalculator calc(size, max_blocks, max_threads);
  
  for (ii=0;ii<calc.size();ii++){
    reindexing_kernel<<<calc[ii].blocks,max_threads>>>(size,new_indexes_buffer,nbins,calc[ii].data_idx,
						       tsamp_by_period, nrots_per_subint);
  }
  ErrorChecker::check_cuda_error("Error from reindexing_kernel.");
  
  thrust::device_ptr<int>   th_new_idx_ptr(new_indexes_buffer);
  thrust::device_ptr<float> th_sorted_tim_ptr(sorted_tim_buffer);

  thrust::sort_by_key(th_new_idx_ptr,th_new_idx_ptr+size,th_sorted_tim_ptr);
  
  for (ii=0;ii<calc.size();ii++){
    find_switch_points_kernel<<<calc[ii].blocks,max_threads>>>(new_indexes_buffer, size,
							       switch_points, calc[ii].data_idx);
  }
  ErrorChecker::check_cuda_error("Error from find_switch_points_kernel.");
  
  
  BlockCalculator calc2(nbins*nints, max_blocks, max_threads);
  for (ii=0;ii<calc2.size();ii++)
    bin_timeseries_kernel<<<calc2[ii].blocks,max_threads>>>(sorted_tim_buffer, subints_buffer,
							    switch_points, nbins*nints);
  ErrorChecker::check_cuda_error("Error from bin_time_series_kernel.");
  Utils::device_free(switch_points);
}


/*
void device_fold_timeseries(float* i_data, float* o_data, 
			    size_t size, unsigned int nbins, unsigned int nints,
			    double period, double tsamp, unsigned int max_blocks,
			    unsigned int max_threads)
{
  thrust::device_ptr<float> ptr(o_data);
  thrust::fill(ptr,ptr+nbins*nints,0.0);
  double tsamp_by_period = tsamp/period;
  double nrots_per_subint = (size*tsamp)/period/nints;
  BlockCalculator calc(size, max_blocks, max_threads);
  for (int ii=0;ii<calc.size();ii++){
    reindexing_kernel<<<calc[ii].blocks,max_threads>>>(i_data, o_data, size, calc[ii].data_idx, 
						       tsamp_by_period, nrots_per_subint); 
  }
}
*/

/*
__global__ 
void interpolated_rebin_time_series_kernel(float* i_data, float* o_data,
					   unsigned int size, float tsamp,
					   float period, unsigned int nbins,
					   unsigned int gulp_idx, 
					   unsigned int in_size, float frac)
{
  int oidx = blockIdx.x * blockDim.x;
  int iidx = __float2int_rd(oidx*frac);
  
  extern __shared__ float i_data_s[];

  if (threadIdx.x < blockDim.x*frac+1){
    if (iidx+threadIdx.x < in_size)
      i_data_s[threadIdx.x] = i_data[iidx+threadIdx.x];
  }
  __syncthreads();
  if (oidx+threadIdx.x < size){ 
    float ipart;
    float x = threadIdx.x*frac + modff(oidx*frac,&ipart);
    int x0 = __float2int_rd(x);
    int x1 = x0+1;
    float y0 = i_data_s[x0];
    float y1 = i_data_s[x1];
    o_data[oidx+threadIdx.x] = y0 + (y1-y0)*((x-x0)/(x1-x0));
  }
}
*/
__global__
void interpolated_rebin_time_series_kernel(float* i_data, float* o_data,
                                           unsigned int size, float tsamp,
                                           float period, unsigned int nbins,
                                           unsigned int gulp_idx,
                                           unsigned int in_size, float frac)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x + gulp_idx;
    
  if (idx==0){
    o_data[idx]=i_data[idx];
    return;
  }
  else if (idx>=size)
    return;
  
  float x = idx*frac;
  int x0 = __float2int_rd(x);
  int x1 = x0+1;
  
  if (x1 >= in_size || x0 >= in_size)
    return;
  
  float y0 = i_data[x0];
  float y1 = i_data[x1];

  o_data[idx] = y0 + (y1-y0)*((x-x0)/(x1-x0));
  
}


__global__ 
void rebin_time_series_kernel(float* i_data, float* o_data,
			      unsigned int size, float tsamp,
			      float period, unsigned int nbins,
			      unsigned int gulp_idx, unsigned int in_size) 
{ 
  int ii;
  float val=0;
  int count=0;
  int idx = blockIdx.x * blockDim.x + threadIdx.x + gulp_idx;
  if (idx>=size)
    return;

  int start_idx = __float2int_rn(idx*period/(tsamp*nbins));
  int end_idx = __float2int_rn((idx+1)*period/(tsamp*nbins));

  if (end_idx>=in_size)
    end_idx=in_size-1;
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
  ErrorChecker::check_cuda_error("Error from device_create_subints");
}

void device_rebin_time_series(float* input, float* output,
                              double period, double tsamp,
                              size_t in_size, size_t out_size,
                              unsigned int nbins,
                              unsigned int max_blocks, unsigned int max_threads)
{
  double frac = period/tsamp/nbins;
  //the +4 below is a accounts for a potential rounding error in the
  //rebinning kernel and should be left as is. Saftey first.
  //int shared_mem_size = (int)(max_threads*frac+4)*sizeof(float);
  BlockCalculator calc(out_size, max_blocks, max_threads);
  for (int ii=0;ii<calc.size();ii++){
    if (frac < 1.0)
      interpolated_rebin_time_series_kernel
	<<<calc[ii].blocks,max_threads>>>(input,output,out_size,
					  (float)tsamp,(float)period,nbins,
					  calc[ii].data_idx,in_size,frac);
    else 
      rebin_time_series_kernel
	<<<calc[ii].blocks,max_threads>>>(input,output,out_size,
					  (float)tsamp,(float)period,nbins,
					  calc[ii].data_idx,in_size);
  }
  ErrorChecker::check_cuda_error("Error from device_rebin_time_series");
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
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
  ErrorChecker::check_cuda_error("Error from thrust::max_element in device_argmax");
  return thrust::distance(ptr,max_elem);
}

void device_real_to_complex(float* input, cuComplex* output, unsigned int size, 
			    unsigned int max_blocks, unsigned int max_threads)
{
  BlockCalculator calc(size,max_blocks,max_threads);
  for (int ii=0;ii<calc.size();ii++)
    real_to_complex_kernel<<<calc[ii].blocks,max_threads>>>(input,output,size);
  ErrorChecker::check_cuda_error("Error from device_real_to_complex");
  return;
}


void device_get_absolute_value(cuComplex* input, float* output, unsigned int size,
			       unsigned int max_blocks, unsigned int max_threads)
{
  BlockCalculator calc(size,max_blocks,max_threads);
  for (int ii=0;ii<calc.size();ii++)
    cuCabsf_kernel<<<calc[ii].blocks,max_threads>>>(input,output,size);
  ErrorChecker::check_cuda_error("Error from device_get_absolute_value");
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
  ErrorChecker::check_cuda_error("Error from device_generate_shift_array");
  return;
}

void device_generate_template_array(cuComplex* templates, unsigned int nbins, 
				    unsigned int size, unsigned int max_blocks,
				    unsigned int max_threads)
{
  BlockCalculator calc(size,max_blocks,max_threads);
  for (int ii=0;ii<calc.size();ii++){
    template_generator_kernel<<<calc[ii].blocks,max_threads>>>(templates, nbins, size);
  }
  ErrorChecker::check_cuda_error("Error from device_generate_template_array");
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
  ErrorChecker::check_cuda_error("Error from device_multiply_by_shift");
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
  ErrorChecker::check_cuda_error("Error from device_collapse_subints");
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
  ErrorChecker::check_cuda_error("Error from device_multiply_by_templates");
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
  ErrorChecker::check_cuda_error("Error from device_zap_birdies");
  return;
}

//--------------coincidence matching--------------//

__global__ 
void coincidence_kernel(float** arrays, float* out_array,
			int narrays, size_t size,
			float thresh, int beam_thresh)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int count = 0;
  for (int ii=0;ii<narrays;ii++)
    if (arrays[ii][idx]>thresh)
      count++;
  out_array[idx] = (count<beam_thresh);
}

void device_coincidencer(float** arrays, float* out_array, 
			 int narrays, size_t size,
			 float thresh, int beam_thresh,
			 unsigned int max_blocks, 
			 unsigned int max_threads)
{
  
  BlockCalculator calc(size, max_blocks, max_threads);
  for (int ii=0;ii<calc.size();ii++)
    coincidence_kernel<<<calc[ii].blocks,max_threads>>>
      (arrays,out_array,narrays,size,thresh,beam_thresh);
  ErrorChecker::check_cuda_error("Error from device_coincidencer");
  return;
  
}

//--------Correlation tools--------//

__global__ void conjugate_kernel(cufftComplex* x, unsigned int size, 
				 unsigned int gulp_idx){
  int idx = blockIdx.x * blockDim.x + threadIdx.x + gulp_idx;
  if (idx<size)
    x[idx].y *= -1.0;
}

void device_conjugate(cufftComplex* x, unsigned int size,
		      unsigned int max_blocks,
		      unsigned int max_threads)
{
  BlockCalculator calc(size, max_blocks, max_threads);
  for (int ii=0;ii<calc.size();ii++)
    conjugate_kernel<<<calc[ii].blocks,max_threads>>>(x,size,calc[ii].data_idx);
  ErrorChecker::check_cuda_error("Error from device_conjugate");
  return;
}

__global__ void cuCmulf_inplace_kernel(cufftComplex* x, cufftComplex* y, 
						unsigned int size, unsigned int gulp_idx){
  int idx = blockIdx.x * blockDim.x + threadIdx.x + gulp_idx;
  if (idx<size)
    y[idx] = cuCmulf(x[idx],y[idx]);
}

void device_cuCmulf_inplace(cufftComplex* x, cufftComplex* y,
			    unsigned int size,
			    unsigned int max_blocks,
			    unsigned int max_threads)
{
  BlockCalculator calc(size, max_blocks, max_threads);
  for (int ii=0;ii<calc.size();ii++)
    cuCmulf_inplace_kernel<<<calc[ii].blocks,max_threads>>>(x,y,size,calc[ii].data_idx);
  ErrorChecker::check_cuda_error("Error from device_cuCmulf_inplace");
  return;
}

//--------type converter--------//
//This is to get around the stupid thrust copy issue

template <class X,class Y> __global__
void conversion_kernel(X* x, Y* y, unsigned int size,
                       unsigned int gulp_idx)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x + gulp_idx;
  if (idx<size)
    y[idx] = x[idx];
  return;
}

template __global__ void conversion_kernel<char,float>(char*,float*,unsigned int,unsigned int);
template __global__ void conversion_kernel<unsigned char,float>(unsigned char*,float*,unsigned int,unsigned int);

template <class X,class Y>
void device_conversion(X* x, Y* y, unsigned int size,
                       unsigned int max_blocks,
                       unsigned int max_threads)
{
  BlockCalculator calc(size, max_blocks, max_threads);
  for (int ii=0;ii<calc.size();ii++)
    conversion_kernel<X,Y> <<<calc[ii].blocks,max_threads>>>(x,y,size,calc[ii].data_idx);
  ErrorChecker::check_cuda_error("Error from device_conversion");
  return;
}

template void device_conversion<char,float>(char*, float*, unsigned int, unsigned int, unsigned int);
template void device_conversion<unsigned char,float>(unsigned char*, float*, unsigned int, unsigned int, unsigned int);

//--------FIR delays--------//
/*
__global__
void fir_kernel(char* input, char* output, 
		size_t input_size, cuComplex* fir,
		unsigned int fir_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int data_idx = 2*(idx - (blockIdx.x * fir_size));
  unsigned int fsize_by_2 = fir_size/2;
    
  __shared__ cuComplex* cdata[MAX_THREADS];
  
  char real = input[data_idx];
  char imag = input[data_idx+1];
  cdata[threadIdx.x] = make_cuComplex((float) real+0.5, (float) imag+0.5);
  __syncthreads();
  
  if (data_idx >= input_size)
    return;
  
  int start,end;
  
  if (data_idx < fir_size){
    start = fir_size-data_idx;
    end = fir_size;
    

  }

  else if (threadIdx.x<fsize_by_2)
    return;
  
  else if (data_idx+fir_size >= input_size){
    start = 0;
    end = input_size-data_idx;
  }
  else if (threadIdx.x > MAX_THREADS-fsize_by_2)
    return;

  else {						
    start = 0;
    end = fir_size;
  }
  cuComplex val = make_cuComplex(0.0,0.0);
  
  for (int ii=start; ii<end; ii++)
    val += cuCmulf(fdata[threadIdx.x+ii],fir[ii]);
  
  //convert back to char*
  
  output[data_idx] = real_char;
  output[data_idx+1] = imag_char;
  

}

*/
