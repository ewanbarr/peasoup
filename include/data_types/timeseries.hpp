#pragma once
#include <vector>

//######################
template <class T> class TimeSeries {
protected:
  T* data_ptr;
  unsigned int nsamps;
  float tsamp;
  
public:  
  TimeSeries(T* data_ptr,unsigned int nsamps,float tsamp)
    :data_ptr(data_ptr), nsamps(nsamps), tsamp(tsamp){}
  
  T operator[](int idx){
    return data_ptr[idx];
  }
  
  unsigned int get_nsamps(void){return nsamps;}
  void set_nsamps(unsigned int nsamps){this->nsamps = nsamps;}
  float get_tsamp(void){return tsamp;}
  void set_tsamp(float tsamp){this->tsamp = tsamp;}
};

//#########################
//class DeviceTimeSeries 


//#########################

template <class T>
class DedispersedTimeSeries: public TimeSeries<T> {
private:
  float dm;

public:
  DedispersedTimeSeries(T* data_ptr, unsigned int nsamps, float tsamp, float dm)
    :TimeSeries<T>(data_ptr,nsamps,tsamp),dm(dm){}
  
  float get_dm(void){return dm;}
  void set_dm(float dm){this->dm = dm;}
};

//###########################

template <class T>
class FilterbankChannel: public TimeSeries<T> {
private:
  float freq;
  
public:
  FilterbankChannel(T* data_ptr, unsigned int nsamps, float tsamp, float freq)
    :TimeSeries<T>(data_ptr,nsamps,tsamp),freq(freq){}
};

//#############################

template <class T>
class TimeSeriesContainer {
protected:
  T* data_ptr;
  unsigned int nsamps;
  float tsamp;
  unsigned int count;
  
  TimeSeriesContainer(T* data_ptr, unsigned int nsamps, float tsamp, unsigned int count)
    :data_ptr(data_ptr),nsamps(nsamps),tsamp(tsamp),count(count){}
  
public:
  unsigned int get_count(void){return count;}
  unsigned int get_nsamps(void){return nsamps;}
  void set_tsamp(float tsamp){this->tsamp = tsamp;}
};

//created through Dedisperser
template <class T>
class DispersionTrials: public TimeSeriesContainer<T> {
private:
  std::vector<float> dm_list;
  
public:
  DispersionTrials(T* data_ptr, unsigned int nsamps, float tsamp, std::vector<float> dm_list_in)
    :TimeSeriesContainer<T>(data_ptr,nsamps,tsamp, (unsigned int)dm_list_in.size())
  {
    dm_list.swap(dm_list_in);
  }
  
  DedispersedTimeSeries<T> operator[](int idx){
    T* ptr = this->data_ptr+idx*this->nsamps;
    return DedispersedTimeSeries<T>(ptr, this->nsamps, this->tsamp, dm_list[idx]);
  }
  //DedispersedTimeSeries<T> nearest_dm(float dm){}
};



//created through Channeliser
template <class T>
class FilterbankChannels: public TimeSeriesContainer<T> {

public:
  FilterbankChannel<T> operator[](int idx);
  FilterbankChannel<T> nearest_chan(float freq);
  
};
