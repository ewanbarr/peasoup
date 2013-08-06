#pragma once


void device_harmonic_sum(float* d_input_array,
			 float* d_output_array,
			 int original_size,
			 int harmonic,
			 unsigned int max_blocks,
			 unsigned int max_threads);

void device_form_power_series(cufftComplex* d_array_in,
			      float* d_array_out,
			      int size,
			      int way);

void device_resample(float * d_idata,
		     float * d_odata,
		     unsigned int length,
		     float a,
		     float timestep,
		     unsigned int block_size,
		     unsigned int max_blocks);

int device_find_peaks(int n,
		      int start_index,
		      float * d_dat,
		      float thresh,
		      int * indexes,
		      float * snrs);

void device_normalise_spectrum(int nsamp,
			       float* d_power_spectrum,
			       float* d_normalised_power_spectrum,
			       int min_bin,
			       float * sigma);

//------Folding related------//

void device_rebin_time_series(float* input,
			      float* output,
                              float period,
			      float tsamp,
                              unsigned int in_size, 
			      unsigned int out_size,
                              unsigned int nbins,
                              unsigned int max_blocks,
			      unsigned int max_threads);

void device_create_subints(float* input,
			   float* output,
                           unsigned int nbins,
                           unsigned int output_size,
                           unsigned int nrots_per_subint,
                           unsigned int max_blocks,
                           unsigned int max_threads);

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
