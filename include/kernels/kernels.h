#pragma once
#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/device_vector.h>
#include <map>

class cached_allocator
{
 public:
  // just allocate bytes
  typedef char value_type;

  cached_allocator() {}

  ~cached_allocator()
    {
      // free all allocations when cached_allocator goes out of scope
      free_all();
    }

  char *allocate(std::ptrdiff_t num_bytes)
  {
    char *result = 0;

    // search the cache for a free block
    free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

    if(free_block != free_blocks.end())
      {
        // get the pointer
        result = free_block->second;

        // erase from the free_blocks map
        free_blocks.erase(free_block);
      }
    else
      {
        // no allocation of the right size exists
        // create a new one with cuda::malloc
        // throw if cuda::malloc can't satisfy the request
        try
	  {
	    // allocate memory and convert cuda::pointer to raw pointer
	    result = thrust::cuda::malloc<char>(num_bytes).get();
	  }
        catch(std::runtime_error &e)
	  {
	    throw;
	  }
      }

    // insert the allocated pointer into the allocated_blocks map
    allocated_blocks.insert(std::make_pair(result, num_bytes));

    return result;
  }

  void deallocate(char *ptr, size_t n)
  {
    // erase the allocated block from the allocated blocks map
    allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
    std::ptrdiff_t num_bytes = iter->second;
    allocated_blocks.erase(iter);

    // insert the block into the free blocks map
    free_blocks.insert(std::make_pair(num_bytes, ptr));
  }

 private:
  typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
  typedef std::map<char *, std::ptrdiff_t>     allocated_blocks_type;

  free_blocks_type      free_blocks;
  allocated_blocks_type allocated_blocks;

  void free_all()
  {

    // deallocate all outstanding blocks in both lists
    for(free_blocks_type::iterator i = free_blocks.begin();
	i != free_blocks.end();
	++i)
      {
        // transform the pointer to cuda::pointer before calling cuda::free
	thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
      }

    for(allocated_blocks_type::iterator i = allocated_blocks.begin();
	i != allocated_blocks.end();
	++i)
      {
        // transform the pointer to cuda::pointer before calling cuda::free
	thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
      }
  }

};

typedef struct {
  unsigned subint;
  unsigned phasebin;
} peasoup_fold_plan;

void device_harmonic_sum(float* d_input_array, 
			 float** d_output_array,
			 size_t size, 
			 unsigned nharms,
			 unsigned int max_blocks,
			 unsigned int max_threads);

void device_form_power_series(cufftComplex* d_array_in,
			      float* d_array_out,
			      size_t size,
			      int way,
			      unsigned int max_blocks,
			      unsigned int max_threads);

void device_resample(float * d_idata,
		     float * d_odata,
		     size_t length,
		     float a,
		     float timestep,
		     unsigned int block_size,
		     unsigned int max_blocks);

void device_resampleII(float * d_idata,
                     float * d_odata,
                     size_t length,
                     float a,
                     float timestep,
                     unsigned int block_size,
                     unsigned int max_blocks);

int device_find_peaks(int n,
		      int start_index,
		      float * d_dat,
		      float thresh,
		      int * indexes,
		      float * snrs,
		      thrust::device_vector<int>&,
		      thrust::device_vector<float>&,
		      cached_allocator&);

void device_normalise(float* d_powers,
                      float mean,
                      float sigma,
                      unsigned int size,
                      unsigned int max_blocks,
                      unsigned int max_threads);

void device_normalise_spectrum(int nsamp,
			       float* d_power_spectrum,
			       float* d_normalised_power_spectrum,
			       int min_bin,
			       float * sigma);

//------Folding related------//

void device_fold_timeseries(float* input,
			    float* output,
			    size_t nsamps,
			    size_t nsubints,
			    double period,
			    double tsamp,
			    int nbins,
			    size_t max_blocks, 
			    size_t max_threads);

//------GPU fold optimisation related-----//

unsigned int device_argmax(float* input, 
			   unsigned int size);

void device_real_to_complex(float* input, 
			    cuComplex* output, 
			    unsigned int size,
			    unsigned int max_blocks,
			    unsigned int max_threads);

void device_get_absolute_value(cuComplex* input, 
			       float* output, 
			       unsigned int size,
                               unsigned int max_blocks, 
			       unsigned int max_threads);

void device_generate_shift_array(cuComplex* shifted_ar,
                                 unsigned int shifted_ar_size,
                                 unsigned int nbins, 
				 unsigned int nints,
                                 unsigned int nshift,
				 float* shifts,
                                 unsigned int max_blocks,
				 unsigned int max_threads);

void device_generate_template_array(cuComplex* templates, 
				    unsigned int nbins, 
				    unsigned int size,
				    unsigned int max_blocks,
				    unsigned int max_threads);

void device_multiply_by_shift(cuComplex* input, 
			      cuComplex* output,
                              cuComplex* shift_array,
			      unsigned int size,
			      unsigned int nbins_by_nints,
			      unsigned int max_blocks,
			      unsigned int max_threads);

void device_collapse_subints(cuComplex* input, 
			     cuComplex* output,
                             unsigned int nbins,
			     unsigned int nints,
                             unsigned int size,
			     unsigned int max_blocks, 
			     unsigned int max_threads);

void device_multiply_by_templates(cuComplex* input, 
				  cuComplex* output,
				  cuComplex* templates,
				  unsigned int nbins,
				  unsigned int nshifts,
				  unsigned int size,
				  unsigned int step,
				  unsigned int max_blocks, 
				  unsigned int max_threads);


//--------------median filter------------//

typedef unsigned char         hd_byte;
typedef size_t                hd_size;
typedef float                 hd_float;
typedef struct hd_pipeline_t* hd_pipeline;

typedef int hd_error;
enum {
  HD_NO_ERROR = 0,
  HD_MEM_ALLOC_FAILED,
  HD_MEM_COPY_FAILED,
  HD_INVALID_DEVICE_INDEX,
  HD_DEVICE_ALREADY_SET,
  HD_INVALID_PIPELINE,
  HD_INVALID_POINTER,
  HD_INVALID_STRIDE,
  HD_INVALID_NBITS,
  HD_PRIOR_GPU_ERROR,
  HD_INTERNAL_GPU_ERROR,
  HD_TOO_MANY_EVENTS,
  HD_UNKNOWN_ERROR
};

hd_error median_scrunch5(const hd_float* d_in,
                         hd_size         count,
                         hd_float*       d_out);

hd_error linear_stretch(const hd_float* d_in,
                        hd_size         in_count,
                        hd_float*       d_out,
                        hd_size         out_count);


void device_divide_c_by_f(cuComplex* c, 
			  float* f, 
			  unsigned int size,
			  unsigned int max_blocks, 
			  unsigned int max_threads);

void device_zap_birdies(cuComplex* fseries, 
			float* d_birdies,
			float* d_widths,
			float bin_width,
                        unsigned int birdies_size,
			unsigned int fseries_size,
                        unsigned int max_blocks,
			unsigned int max_threads);

//-----------stats-----------//

template <typename T>
float GPU_rms(T* d_collection,
	      int nsamps,
	      int min_bin);

template <typename T>
float GPU_mean(T* d_collection,
	       int nsamps,
	       int min_bin);  

template <typename T>
void GPU_fill(T* start,
	      T* end,
	      T value);

//----------coincidencer-----------//

void device_coincidencer(float** arrays, 
			 float* out_array,
                         int narrays,
			 size_t size,
                         float thresh,
			 int beam_thresh,
                         unsigned int max_blocks,
                         unsigned int max_threads);

//-------correlation-----///

void device_conjugate(cufftComplex* x, 
		      unsigned int size,
                      unsigned int max_blocks,
                      unsigned int max_threads);

void device_cuCmulf_inplace(cufftComplex* x, 
			    cufftComplex* y,
                            unsigned int size,
                            unsigned int max_blocks,
                            unsigned int max_threads);

template <class X, class Y>
void device_conversion(X*, Y*, unsigned int,
		       unsigned int, unsigned int);

