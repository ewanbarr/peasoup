/*
  Copyright 2014 Ewan Barr

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0  

  Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/
/*
  fourierseries.hpp
  
  By Ewan Barr (2013)
  ewan.d.barr@gmail.comm

  This file contains classes for the storage and manipulation of 
  both Fourier series and power spectra.
*/
#pragma once
#include <complex>
#include "cufft.h"
#include "cuda.h"
#include "utils/exceptions.hpp"
#include "utils/utils.hpp"
#include <vector>

/*!
  \brief Base class for storage of all frequency series.
*/
template <class T>
class FrequencySeries {
protected:
  T* data_ptr; /*!< Pointer to series data.*/
  unsigned int nbins; /*!< Number of bins in series.*/
  double bin_width; /*!< Width of each bin in frequency space (Hz).*/

  /*!
    \brief Construct a default FrequencySeries instance with all values set to zero.
  */
  FrequencySeries(void)
    :data_ptr(0),nbins(0),bin_width(0){}
  
  /*!
    \brief Construct a FrequencySeries instance with a NULL data pointer.
    
    \param nbins Number of bins in series.
    \param bin_width Width of each bin in frequency space (Hz).
  */
  FrequencySeries(unsigned int nbins, double bin_width)
    :data_ptr(0),nbins(nbins),bin_width(bin_width){}
  
  /*!
    \brief Construct a FrequencySeries instance with a NULL data pointer.

    \param data_ptr Pointer to frequency series data.
    \param nbins Number of bins in series.
    \param bin_width Width of each bin in frequency space (Hz).
  */
  FrequencySeries(T* data_ptr, unsigned int nbins, double bin_width)
    :data_ptr(data_ptr),nbins(nbins),bin_width(bin_width){}

public:
  
  /*!
    \brief Get pointer to frequency series data.
    
    \return Pointer to frequency series data.
  */
  T* get_data(void){return data_ptr;}
  
  /*!
    \brief Set pointer to frequency series data.
  
    \param Pointer tofrequency series data.
  */
  void set_data(T* data_ptr){this->data_ptr = data_ptr;};
  
  /*!
    \brief Get bin width in frequency.

    \return Bin width in frequency (Hz).
  */
  double get_bin_width(void){return bin_width;}
  
  /*!
    \brief Set bin width in frequency.
    
    \param Bin width in frequency (Hz).
  */
  void set_bin_width(double bin_width){this->bin_width = bin_width;}
 
  /*!
    \brief Get number of frequency bins.

    \return Number of frequency bins.
  */
  unsigned int get_nbins(void){return nbins;}
  
  /*!
    \brief Set number of frequency bins.
    
    \param Number of frequency bins.
  */
  void set_nbins(unsigned int nbins){this->nbins = nbins;}
};


/*!
  \brief Subclass for handling of frequency series on the GPU.
  
  Subclass of FrequencySeries that allocates and deallocates
  GPU memory for storage of frequency series data on the GPU.
*/
template <class T>
class DeviceFrequencySeries: public FrequencySeries<T> {
protected:
  DeviceFrequencySeries(unsigned int nbins, double bin_width)
    :FrequencySeries<T>(nbins,bin_width)
  {
    Utils::device_malloc<T>(&this->data_ptr,nbins);
  }

  ~DeviceFrequencySeries()
  {
    Utils::device_free(this->data_ptr);
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
  unsigned int nh;
  
public:
  DevicePowerSpectrum(unsigned int nbins, double bin_width,unsigned int nh=0)
    :DeviceFrequencySeries<T>(nbins,bin_width),nh(nh){}
  
  template <class U>
  DevicePowerSpectrum(FrequencySeries<U>& other,unsigned int nh=0)
    :DeviceFrequencySeries<T>(other.get_nbins(),other.get_bin_width()),nh(nh){}

  unsigned int get_nh(void){return nh;}
  void set_nh(unsigned int nh_){nh=nh_;}

};

template <class T>
class HarmonicSums {
private:
  std::vector< DevicePowerSpectrum<T>* > folds;

public:
  HarmonicSums(DevicePowerSpectrum<T>& fold0, unsigned int nfolds)
  {
    folds.reserve(nfolds);
    for (int ii=0;ii<nfolds;ii++)
      folds.push_back(new DevicePowerSpectrum<T>(fold0,ii+1));
  }
  
  size_t size(void){
    return folds.size();
  }

  DevicePowerSpectrum<T>* operator[](int ii){
    return folds[ii];
  }
  
  ~HarmonicSums()
  {
    for (int ii=0;ii<folds.size();ii++)
      delete folds[ii];
  }
};
