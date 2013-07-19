#include <data_types/timeseries.hpp>
#include <kernels/kernels.h>
#include <kernels/defaults.h>
#include <utils/exceptions.hpp>

class TimeDomainResampler {
private:
  unsigned int block_size;
  unsigned int max_blocks;
  
public:
  TimeDomainResampler(unsigned int block_size=BLOCK_SIZE, unsigned int max_blocks=MAX_BLOCKS)
    :block_size(block_size),max_blocks(max_blocks)    
  {
  }
  
  //Force float until the kernel gets templated
  void resample(DeviceTimeSeries<float>& input, DeviceTimeSeries<float>& output, float acc)
  {
    device_resample(input.get_data(), output.get_data(), input.get_nsamps()
		    acc, input.get_tsamp(), block_size,  max_blocks);
    ErrorChecker::check_cuda_error();
  }
};


