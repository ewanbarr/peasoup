#pragma once
#include <complex>
#include "cufft.h"
#include "cuda.h"
#include "utils/exceptions.hpp"

template <class T>
class FourierSeries {
protected:
  T* data_ptr;
  unsigned int nbins;
  double bin_width;

public:
  FourierSeries(void)
    :data_ptr(0),nbins(0),bin_width(0){}
  
  FourierSeries(unsigned int nbins, double bin_width)
    :data_ptr(0),nbins(nbins),bin_width(bin_width){}

  FourierSeries(T* data_ptr, unsigned int nbins, double bin_width)
    :data_ptr(data_ptr),nbins(nbins),bin_width(bin_width){}

  T* get_data(void){return data_ptr;}
  void set_data(T* data_ptr){this->data_ptr = data_ptr;};
  double get_bin_width(void){return bin_width;}
  void set_bin_width(double bin_width){this->bin_width = bin_width;}
  unsigned int get_nbins(void){return nbins;}
  void set_nbins(unsigned int nbins){this->nbins = nbins;}
};

class DeviceFourierSeries: public FourierSeries<cufftComplex> {
public:
  DeviceFourierSeries(unsigned int nbins, double bin_width)
    :FourierSeries<cufftComplex>(nbins,bin_width)
  {
    cudaError_t error = cudaMalloc((void**)&data_ptr, sizeof(cufftComplex)*nbins);
    ErrorChecker::check_cuda_error(error);
  }

};


