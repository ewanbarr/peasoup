#include <iostream>
#include "cuda.h"
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/tuple.h>
#include <thrust/fill.h>

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
			 int original_size, int harmonic)
{
  int gulps;
  int gulp_counter;
  int gulp_index = 0;
  int gulp_size;
  int blocks = 0;
  float one_over_sqrt_harm = 1.0f/sqrt((float)harmonic);
  gulps = original_size/(MAX_BLOCKS*MAX_THREADS)+1;
  for(gulp_counter = 0; gulp_counter<gulps; gulp_counter++)
    {
      if(gulp_counter<gulps-1)
        {
          gulp_size = MAX_BLOCKS*MAX_THREADS;
        }
      else
        {
          gulp_size = original_size - gulp_counter*MAX_BLOCKS*MAX_THREADS;
        }
      blocks = (gulp_size-1)/MAX_THREADS + 1;
      harmonic_sum_kernel_generic<<<blocks,MAX_THREADS>>>(d_input_array,d_output_array,
							  gulp_index,gulp_size,harmonic,
							  one_over_sqrt_harm);
      gulp_index = gulp_index + blocks*MAX_THREADS;
    }
  return;
}

//------------spectrum forming--------------//

__global__ void power_series_kernel(cufftComplex *d_idata,float* d_odata, int size)
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
      d_odata[Index] = sqrt(max(ampsq,ampsq_diff));
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
      if (way==1)
        bin_interbin_series_kernel<<<MAX_BLOCKS,MAX_THREADS>>>(gulp_in_ptr,gulp_out_ptr,gulp_size);
      gulp_in_ptr = gulp_in_ptr + MAX_BLOCKS*MAX_THREADS;
      gulp_out_ptr = gulp_out_ptr + MAX_BLOCKS*MAX_THREADS;
    }
  return;
}

//-----------------time domain resampling---------------//

__global__
void jstretch_kernel( float* d_odata, float* d_idata,
		      int start_index, int length,
		      float a, float timestep)
{
  double T = timestep*((float)length-1.0);
  double c = (float)299792458.0;

  double A = a/2.0;
  double B = -(a*T/2.0+c);

  double tobs;

  double xmax = -a*T*T/8.0;
  double dmax = (double)xmax/(double)c;


  unsigned int index = start_index + blockIdx.x*blockDim.x + threadIdx.x;

  if(index < length)
    {
      tobs = (double)index*timestep;
      double C = a*T*T/8.0 + c*tobs;
      float read_location;
      read_location = (dmax + (-B - sqrt(B*B - 4.0*A*C))/(2.0*A))/timestep;
      d_odata[index] = d_idata[(int)read_location] 
	+ (d_idata[1+(int)read_location] 
	   - d_idata[(int)read_location])
	*(read_location - (int)read_location);
    }
}

void device_resample(float * d_idata, float * d_odata,
		     unsigned int length, float a, 
		     float timestep, unsigned int block_size,
		     unsigned int max_blocks)
{
  dim3 dimBlock(block_size, 1, 1);
  int start_index;
  int gulp_length;
  start_index = 0;
  
  while(start_index < (int)length)
    {
      if(length - start_index >= max_blocks*block_size)
        {
          gulp_length = max_blocks*block_size;
        }
      else
        {
          gulp_length = length - start_index;
        }
      
      int blocks = (gulp_length - 1)/block_size + 1;
      
      dim3 dimGrid(blocks, 1, 1);
      
      jstretch_kernel<<< dimGrid, dimBlock, 0 >>>(d_odata+start_index, d_idata+start_index, start_index, gulp_length,(float)a,(float)timestep);
      start_index += gulp_length;
    }
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
  int num_copied = thrust::copy_if(zipped_iter, zipped_iter+n-start_index,zipped_out_iter,greater_than_threshold(thresh)) - zipped_out_iter;
  thrust::copy(d_index.begin(),d_index.begin()+num_copied,indexes);
  thrust::copy(d_snrs.begin(),d_snrs.begin()+num_copied,snrs);
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
  rms_sum = thrust::transform_reduce(device_ptr<T>(d_collection)+min_bin, device_ptr<T>(d_collection)+nsamps, square<T>(),T(0),thrust::plus<T>());
  rms = sqrt(float(rms_sum)/float(nsamps-min_bin));

  return rms;
}

template<typename T>
float GPU_mean(T* d_collection,int nsamps, int min_bin)
{
  float mean;
  T m_sum;

  using thrust::device_ptr;
  m_sum = thrust::reduce(device_ptr<T>(d_collection)+min_bin, device_ptr<T>(d_collection)+nsamps);

  cudaThreadSynchronize();
  mean = float(m_sum)/float(nsamps-min_bin);

  return mean;
}

void device_normalise_spectrum(int nsamp,
      float* d_power_spectrum,
      float* d_normalised_power_spectrum,
      int min_bin,
      float * sigma)
{
  float mean;
  float rms;
  float meansquares;
  cudaError_t error;
  
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
}

//--------------Time series folder----------------//


__global__ 
void time_series_folder_kernel(float* i_data, float* o_data, unsigned int size, float tsamp, float period){
  int Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index<size)
    {
      
    }

}


//--------------End--------------//