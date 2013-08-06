#pragma once
#include <cstddef>

typedef unsigned char         hd_byte;
typedef size_t                hd_size;
//typedef unsigned int          hd_size;                                                                     
typedef float                 hd_float;
typedef struct hd_pipeline_t* hd_pipeline;

// Fundamental candidate quantities only                                                                     
struct RawCandidates {
  hd_float* peaks;
  hd_size*  inds;
  hd_size*  begins;
  hd_size*  ends;
  hd_size*  filter_inds;
  hd_size*  dm_inds;
  hd_size*  members;
};
struct ConstRawCandidates {
  const hd_float* peaks;
  const hd_size*  inds;
  const hd_size*  begins;
  const hd_size*  ends;
  const hd_size*  filter_inds;
  const hd_size*  dm_inds;
  const hd_size*  members;
};
// Full candidate info including derived quantities                                                          
struct Candidates : public RawCandidates {
  hd_float* dms;
  hd_size*  flags;
  hd_size*  beam_counts;
  hd_size*  beam_masks;
};

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
  // ...
  HD_UNKNOWN_ERROR
};

hd_error median_filter3(const hd_float* d_in,
                        hd_size         count,
                        hd_float*       d_out);

hd_error median_filter5(const hd_float* d_in,
                        hd_size         count,
                        hd_float*       d_out);

hd_error median_scrunch3(const hd_float* d_in,
                         hd_size         count,
                         hd_float*       d_out);

hd_error median_scrunch5(const hd_float* d_in,
                         hd_size         count,
                         hd_float*       d_out);

// Note: This can operate 'in-place'
hd_error mean_filter2(const hd_float* d_in,
                      hd_size         count,
                      hd_float*       d_out);

hd_error linear_stretch(const hd_float* d_in,
                        hd_size         in_count,
                        hd_float*       d_out,
                        hd_size         out_count);

// Median-scrunches the corresponding elements from a collection of arrays
// Note: This cannot (currently) handle count not being a multiple of 3
hd_error median_scrunch3_array(const hd_float* d_in,
                               hd_size         array_size,
                               hd_size         count,
                               hd_float*       d_out);

// Median-scrunches the corresponding elements from a collection of arrays
// Note: This cannot (currently) handle count not being a multiple of 5
hd_error median_scrunch5_array(const hd_float* d_in,
                               hd_size         array_size,
                               hd_size         count,
                               hd_float*       d_out);

// Mean-scrunches the corresponding elements from a collection of arrays
// Note: This cannot (currently) handle count not being a multiple of 2
hd_error mean_scrunch2_array(const hd_float* d_in,
                             hd_size         array_size,
                             hd_size         count,
                             hd_float*       d_out);
