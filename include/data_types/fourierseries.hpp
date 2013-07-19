#pragma once
#include <complex>
#include "cufft.h"
#include "cuda.h"
#include "utils/exceptions.hpp"
#include <vector>

template <class T>
class FrequencySeries {
protected:
  T* data_ptr;
  unsigned int nbins;
  double bin_width;

  FrequencySeries(void)
    :data_ptr(0),nbins(0),bin_width(0){}
  
  FrequencySeries(unsigned int nbins, double bin_width)
  {
    this->data_ptr = 0;
    this->nbins = nbins;
    this->bin_width = bin_width;
  }

  FrequencySeries(T* data_ptr, unsigned int nbins, double bin_width)
    :data_ptr(data_ptr),nbins(nbins),bin_width(bin_width){}

public:
  T* get_data(void){return data_ptr;}
  void set_data(T* data_ptr){this->data_ptr = data_ptr;};
  double get_bin_width(void){return bin_width;}
  void set_bin_width(double bin_width){this->bin_width = bin_width;}
  unsigned int get_nbins(void){return nbins;}
  void set_nbins(unsigned int nbins){this->nbins = nbins;}
};

template <class T>
class DeviceFrequencySeries: public FrequencySeries<T> {
protected:
  DeviceFrequencySeries(unsigned int nbins, double bin_width)
    :FrequencySeries<T>(nbins,bin_width)
  {
    cudaError_t error = cudaMalloc((void**)&this->data_ptr, sizeof(T)*nbins);
    ErrorChecker::check_cuda_error(error);
  }

  ~DeviceFrequencySeries()
  {
    cudaFree(this->data_ptr);
  }
  
};

//template class should be cufftComplex/cufftDoubleComplex
template <class T>
class DeviceFourierSeries: public DeviceFrequencySeries<T> {
public:
  DeviceFourierSeries(unsigned int nbins, double bin_width)
    :DeviceFrequencySeries<T>(nbins,bin_width){}
};

//template class should be real valued
template <class T>
class DevicePowerSpectrum: public DeviceFrequencySeries<T> {
private:
  unsigned char fold_no;
  
public:
  DevicePowerSpectrum(unsigned int nbins, double bin_width)
    :DeviceFrequencySeries<T>(nbins,bin_width){}
  
  template <class U>
  DevicePowerSpectrum(FrequencySeries<U>& other)
    :DeviceFrequencySeries<T>(other.get_nbins(),other.get_bin_width()){}

};


//
template <class T>
class HarmonicSums {
private:
  DevicePowerSpectrum<T>& fold0;
  std::vector<T*> folds;

public:
  HarmonicSums(DevicePowerSpectrum<T>& fold0, unsigned int nfolds)
    :fold0(fold0)
  {
    cudaError_t error;
    folds.resize(nfolds);
    for (int ii=0;ii<nfolds;ii++)
      {
	error = cudaMalloc((void**)&folds[ii], sizeof(T)*fold0.get_nbins());
	ErrorChecker::check_cuda_error(error);
      }
  }
  size_t size(void){
    return folds.size();
  }
  
  T* operator[](int ii){
    return folds[ii];
  }

  ~HarmonicSums()
  {
    for (int ii=0;ii<folds.size();ii++)
      {
	cudaFree(&folds[ii]);
      }
    folds.clear();
  }
  

};


