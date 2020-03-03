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
  timeseries.hpp
  
  By Ewan Barr (2013)
  ewan.d.barr@gmail.com
  
  This file contains classes for the storage and manipulation
  of timeseries. 
 */

#pragma once
#include <vector>
#include "cuda.h"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include "utils/exceptions.hpp"
#include "utils/utils.hpp"
#include <data_types/header.hpp>
#include <string>
#include "kernels/kernels.h"
#include "kernels/defaults.h"

//TEMP
//#include <stdio.h>
//#include <iostream>

/*!
  \brief Base class for handling timeseries data.

  Base class for all timeseries classes. This class contains
  set and get methods for timeseries parameters as well a 
  convenience method for reading time Sigproc timeseries 
  from file.
*/
template <class T> class TimeSeries {
protected:
  T* data_ptr; /*!< Pointer to timeseries data.*/
  unsigned int nsamps; /*!< Number of samples.*/
  float tsamp; /*!< Sampling time (seconds).*/
  
public:  
  /*!
    \brief Construct TimeSeries object from existing data.
    
    \param data_ptr Pointer to timeseries data.
    \param nsamps Number of samples.
    \param tsamp Sampling time (seconds).
  */
  TimeSeries(T* data_ptr,unsigned int nsamps,float tsamp)
    :data_ptr(data_ptr), nsamps(nsamps), tsamp(tsamp){}

  /*!
    \brief Create a default TimeSeries instance.
    
    Instantiate a TimeSeries instance with all parameters
    set to zero.
  */
  TimeSeries(void)
    :data_ptr(0), nsamps(0.0), tsamp(0.0) {}

  //Why does this exist?
  TimeSeries(unsigned int nsamps)
    :data_ptr(0), nsamps(nsamps), tsamp(0.0){}
  
  /*!
    \brief Return the nth sample from the data.
    
    \param n Index of sample.
    \return nth sample from timeseries.
  */
  T operator[](int n){
    return data_ptr[n];
  }
  
  /*!
    \brief Get data pointer.
    
    \return Pointer to timeseries data.
  */
  T* get_data(void){return data_ptr;}
  
  /*!
    \brief Set the data pointer.
    
    \param Pointer to timeseries data.
  */
  void set_data(T* data_ptr){this->data_ptr = data_ptr;};
  
  /*!
  \brief Get the number of samples.

  \return Number of samples in the time series.
  */
  unsigned int get_nsamps(void){return nsamps;}
  
  /*!
    \brief Set the number of samples.

    \param Number of samples in timeseries.
  */
  void set_nsamps(unsigned int nsamps){this->nsamps = nsamps;}
  
  /*!
    \brief Get sampling time.

    \return Sampling time (seconds).
  */
  float get_tsamp(void){return tsamp;}
  
  /*!
    \brief Set the sampling time.

    \param Sampling time (seconds).
  */
  void set_tsamp(float tsamp){this->tsamp = tsamp;}
  
  /*!
    \brief Read a Sigproc format timeseries file.
    
    \param filename The name of a Sigproc tim file.
  */
  virtual void from_file(std::string filename)
  {
    /*
      NOTE: This function has only been used for testing and is 
      not used as part of the peasoup application. Should 
      consider moving this method to a different class.
      Belongs to a HostTimeSeries class (if one existed).
    */
    std::ifstream infile;
    SigprocHeader hdr;
    infile.open(filename.c_str(),std::ifstream::in | std::ifstream::binary);
    ErrorChecker::check_file_error(infile, filename);
    // Read header
    read_header(infile,hdr);
    if (hdr.nbits/8!=sizeof(T))
      ErrorChecker::throw_error("Bad bit size in input time series");
    size_t input_size = (size_t) hdr.nsamples*sizeof(T);
    this->data_ptr = new T [hdr.nsamples];
    infile.seekg(hdr.size, std::ios::beg);
    // Read data
    infile.read(reinterpret_cast<char*>(this->data_ptr), input_size);
    this->nsamps = hdr.nsamples;
    this->tsamp = hdr.tsamp;
  }
};

/*!
  \brief Subclass of TimeSeries for handling dedispered timeseries.

  Subclass of TimeSeries which adds an attribute for storing the 
  dispersion measure of the timeseries.
*/
template <class T>
class DedispersedTimeSeries: public TimeSeries<T> {
private:
  float dm; /*!< Dispersion measure (pc cm^-3).*/

public:
  /*!
    \brief Construct default DedispersedTimeSeries instance.
    
    Instantiate DedispersedTimeSeries with all parameters set to zero.
  */
  DedispersedTimeSeries()
    :TimeSeries<T>(),dm(0.0){}

  /*!
    \brief Construct a new DedispersedTimeSeries instance.
    
    \param data_ptr Pointer to timeseries data.
    \param nsamps Number of samples.
    \param tsamp Sampling time (seconds).
    \param dm  Dispersion measure (pc cm^-3).
  */
  DedispersedTimeSeries(T* data_ptr, unsigned int nsamps, float tsamp, float dm)
    :TimeSeries<T>(data_ptr,nsamps,tsamp),dm(dm){}
  
  /*!
    \brief Get the dispersion measure.

    \return Dispersion measure (pc cm^-3).
  */
  float get_dm(void){return dm;}
  
  /*!
    \brief Set the dispersion measure.
    
    \param Dispersion measure (pc cm^-3).
  */
  void set_dm(float dm){this->dm = dm;}
};


//NOT USED
template <class T>
class FilterbankChannel: public TimeSeries<T> {
private:
  float freq;  
public:
  FilterbankChannel(T* data_ptr, unsigned int nsamps, float tsamp, float freq)
    :TimeSeries<T>(data_ptr,nsamps,tsamp),freq(freq){}
};


/*!
  \brief TimeSeries subclass for encapsulating on-GPU timeseries.
  
  This class is for the storage and manipulation of timeseries 
  data stored on the GPU. The lifetime of the GPU memory buffer 
  is tied to the life of the DeviceTimeSeries object.
*/
template <class OnDeviceType>
class DeviceTimeSeries: public TimeSeries<OnDeviceType> {
public:
  /*!
    \brief Construct a DeviceTimeSeries with N samples.
    
    Creates a DeviceTimeSeries with N samples. A buffer 
    is allocated in GPU memory.

    \param nsamps Number of samples.
  */
  DeviceTimeSeries(unsigned int nsamps)
    :TimeSeries<OnDeviceType>(nsamps)
  {
    Utils::device_malloc<OnDeviceType>(&this->data_ptr,nsamps);
  }

  /*!
    \brief Create a DeviceTimeSeries instance from TimeSeries instance.
    
    Constructor takes a TimeSeries instance (or subclass), allocates 
    space in GPU RAM and copies the data from the TimeSeries instance
    to the GPU. Data are automatically converted from type of input 
    TimeSeries object to type of DeviceTimeSeries (i.e. TimeSeries<char>
    can be converted to DeviceTimeSeries<float> automatically).
    
    \param host_tim A TimeSeries instance.
  */
  template <class OnHostType>
  DeviceTimeSeries(TimeSeries<OnHostType>& host_tim)
    :TimeSeries<OnDeviceType>(host_tim.get_nsamps())
  {
    OnHostType* copy_buffer;
    Utils::device_malloc<OnDeviceType>(&this->data_ptr,this->nsamps);
    Utils::device_malloc<OnHostType>(&copy_buffer,this->nsamps);
    
    Utils::h2dcpy(copy_buffer,host_tim.get_data(),this->nsamps);
    device_conversion<OnHostType,OnDeviceType>(copy_buffer, this->data_ptr,
                                               (unsigned int)this->nsamps,
                                               (unsigned int)MAX_BLOCKS,
                                               (unsigned int)MAX_THREADS);
    this->tsamp = host_tim.get_tsamp();
    Utils::device_free(copy_buffer);
  }

  /*!
    \brief Fill a range of samples with a value.
    
    \param start Index of first sample to fill.
    \param end Index of last sample to fill.
    \param value Value to fill range with.
  */
  void fill(size_t start, size_t end, OnDeviceType value){
    /*
      Note: GPU_fill is used rather than cudaMemset as it
      allows for arbitrary values to be set.
    */
    if (end > this->nsamps)
      ErrorChecker::throw_error("DeviceTimeSeries::fill bad end value requested");
    GPU_fill(this->data_ptr+start,this->data_ptr+end,value);
  }
 
  /*!
    \brief Destruct the DeviceTimeSeries instance.

    \note Memory allocated in the constructor is freed here.
  */
  ~DeviceTimeSeries()
  {
    Utils::device_free(this->data_ptr);
  }
};


/*!
  \brief DeviceTimeSeries subclass designed for efficient reusability.

  Similar to DeviceTimeSeries this class maintains a memory buffer on 
  the GPU used for conversion of data from host type to device type.
  This class is implemented to remove the need for reallocation of 
  GPU memory between different dm trials being passed to the GPU.
*/
template <class OnDeviceType,class OnHostType>
class ReusableDeviceTimeSeries: public DeviceTimeSeries<OnDeviceType> {
private:
  OnHostType* copy_buffer; //GPU memory buffer
  
public:

  /*!
    \brief Construct a ReusableDeviceTimeSeries with N samples.

    \param nsamps Number of samples.
  */
  ReusableDeviceTimeSeries(unsigned int nsamps)
    :DeviceTimeSeries<OnDeviceType>(nsamps)
  {
    Utils::device_malloc<OnHostType>(&copy_buffer,this->nsamps);
  }
  
  /*!
    \brief Copy a TimeSeries instance into ReusableDeviceTimeSeries.
    
    Copies the data from a TimeSeries instance into ReusableDeviceTimeSeries
    GPU memory buffer. Data undegoes automatic type conversion.
    
    \param host_tim A TimeSeries instance.
  */
  void copy_from_host(TimeSeries<OnHostType>& host_tim)
  {
    size_t size = std::min(host_tim.get_nsamps(),this->nsamps);
    this->tsamp = host_tim.get_tsamp();
    Utils::h2dcpy(copy_buffer, host_tim.get_data(), size);
    device_conversion<OnHostType,OnDeviceType>(copy_buffer, this->data_ptr,
                                               (unsigned int)size,
                                               (unsigned int)MAX_BLOCKS,
					       (unsigned int)MAX_THREADS);
  }
  
  /*!
    \brief Deconstruct the ReusableDeviceTimeSeries instance.

    \note Allocated GPU memory is freed here.
  */
  ~ReusableDeviceTimeSeries()
  {
    Utils::device_free(copy_buffer);
  }
};

  
/*!
  \brief A base wrapper class for multiple timeseries.
  
  Wrapper for class containing multiple timeseries (not TimeSeries instances). 
  Timeseries are stored in a contigous memory buffer.
*/

template <class T>
class TimeSeriesContainer {
protected:
  T* data_ptr; /*!< Pointer to timeseries.*/
  unsigned int nsamps; /*!< Number of samples in each timeseries.*/
  float tsamp; /*!< Sampling time of each timeseries (seconds).*/
  unsigned int count; /*!< Number of timeseries.*/
  
  /*!
    \brief Construct a new TimeSeriesContainer instance.
    
    \param data_ptr Pointer to timeseries data.
    \param nsamps Number of samples in each timeseries.
    \param tsamp Sampling time (seconds).
    \param count Number of timeseries.
  */
  TimeSeriesContainer(T* data_ptr, unsigned int nsamps, float tsamp, unsigned int count)
    :data_ptr(data_ptr),nsamps(nsamps),tsamp(tsamp),count(count){}
  
public:
  /*!
  \brief Get the number of timeseries in the container.
  
  \return Number of timeseries.
  */
  unsigned int get_count(void){return count;}
  
  /*!
  \brief Get the number of samples in each timeseries.

  \return Number of samples.
  */
  unsigned int get_nsamps(void){return nsamps;}
  
  /*!
    \brief Set the sampling time of each timeseries.
  
    \param Sampling time (seconds).
  */
  
  void set_tsamp(float tsamp){this->tsamp = tsamp;}
  
  /*!
    \brief Get the sampling time of each timeseries.
    
    \return Sampling time (seconds).
  */
  float get_tsamp(void){return tsamp;}
  
  /*!
    \brief Get a pointer to the timeseries data.
    
    \return Pointer to timeseries data.
  */
  T* get_data(void){return data_ptr;}
};


/*!
  \brief Subclass of TimeSeriesContainer for storing dedispersed timeseries.
*/
template <class T>
class DispersionTrials: public TimeSeriesContainer<T> {
  /*
    Note: These container classes were not wanted, but have 
    been created to conform with the output of the Dedisp 
    library. Preferably this would be a vector of 
    DedispersedTimeSeries instances.
  */
private:
  std::vector<float> dm_list; /*!< Dispersion measure of each timeseries.*/
  
public:
  /*!
    \brief Create a new DispersionTrials instance.
    
    \param data_ptr Pointer to dedispered data.
    \param nsamps Number of samples in each dedispersed timeseries.
    \param tsamp Sampling time (seconds).
    \param dm_list_in A vector of dispersion measures.
    \note The number of timeseries in the container is dm_list_in.size().
  */
  DispersionTrials(T* data_ptr, unsigned int nsamps, float tsamp, std::vector<float> dm_list_in)
    :TimeSeriesContainer<T>(data_ptr,nsamps,tsamp, (unsigned int)dm_list_in.size())
  {
    dm_list.swap(dm_list_in);
  }
  
  /*!
    \brief Select the Nth timeseries.
    
    \param idx Index of desired time series.
    \return DedispersedTimeSeries instance.
  */
  DedispersedTimeSeries<T> operator[](int idx)
  {
    T* ptr = this->data_ptr+idx*(size_t)this->nsamps;
    return DedispersedTimeSeries<T>(ptr, this->nsamps, this->tsamp, dm_list[idx]);
  }
  
  /*!
    \brief Set DedispersedTimeSeries instance from a DispersionTrials timeseries.
    
    \param idx Index of desired time series.
    \param tim DedispersedTimeSeries which will take the data.
    \note This function is implemented as an alternative to the 
    overloaded [] operator.
  */
  void get_idx(unsigned int idx, DedispersedTimeSeries<T>& tim){
    T* ptr = this->data_ptr+(size_t)idx*(size_t)this->nsamps;
    tim.set_data(ptr);
    tim.set_dm(dm_list[idx]);
    tim.set_nsamps(this->nsamps);
    tim.set_tsamp(this->tsamp);
  }
};



// Created through Channeliser
// NOT USED
template <class T>
class FilterbankChannels: public TimeSeriesContainer<T> {
public:
  FilterbankChannel<T> operator[](int idx);
  FilterbankChannel<T> nearest_chan(float freq);
};
