#include <data_types/fourierseries.hpp>
#include <utils/exceptions.hpp>

class RednoiseRemover {
private:
  unsigned int nbins;
  float* buffer_5;
  float* buffer_25;
  float* buffer_125;
  float* median_array;

public:
  RednoiseRemover(unsigned int nbins)
    :nbins(nbins)
  {
    cudaError_t error;
    error = cudaMalloc((void**)&buffer_5,nbins/5);
    ErrorChecker::check_cuda_error(error);
    error = cudaMalloc((void**)&buffer_25,nbins/25);
    ErrorChecker::check_cuda_error(error);
    error = cudaMalloc((void**)&buffer_125,nbins/125);
    ErrorChecker::check_cuda_error(error);
    error = cudaMalloc((void**)&median_array,nbins);
    ErrorChecker::check_cuda_error(error);
  }

  void find_running_median(DevicePowerSpectrum& input)
  {
    unsigned int size = input.get_nbins();
    median_scrunch5(input.get_data(), size, buffer_5);
    median_scrunch5(buffer_5,  size/5, buffer_25);
    median_scrunch5(buffer_25, size/5/5,buffer_125);
    linear_stretch(buffer_125, size/5/5/5, median_array, size);
  }

  ~RednoiseRemover(){
    cudaFree(buffer_5);
    cudaFree(buffer_25);
    cudaFree(buffer_125);
    cudaFree(median_array);
  }

};
