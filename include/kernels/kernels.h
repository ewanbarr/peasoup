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

