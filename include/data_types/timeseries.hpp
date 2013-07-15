#pragma once
#include <vector>

//######################
template <class T> class TimeSeries {
protected:
  unsigned int nsamps;
  float tsamp;
  T data_ptr;
  
public:  
  TimeSeries(T* data_ptr,unsigned int nsamps,float tsamp);
  T operator[](int idx);
  unsigned int get_nsamps(void);
  void set_nsamps(unsigned int nsamps);
  float get_tsamp(void);
  void set_tsamp(float tsamp);
};

//#########################

template <class T>
class DedispersedTimeSeries: public TimeSeries<T> {
private:
  float dm;

public:
  DedispersedTimeSeries(T* data_ptr, unsigned int nsamps, float tsamp, float dm);
  float get_dm(void);
  void set_dm(float dm);
};

//###########################

template <class T>
class FilterbankChannel: public TimeSeries<T> {

};

//#############################

template <class T>
class TimeSeriesContainer {
protected:
  unsigned int nsamps;
  unsigned int count;
  float tsamp;
  T* data_ptr;
  TimeSeriesContainer(T* data_ptr, unsigned int nsamps, float tsamp, unsigned int count);
};

//created through Dedisperser
template <class T>
class DispersionTrials: public TimeSeriesContainer<T> {
private:
  std::vector<float> dm_list;
  
public:
  DispersionTrials(T* data_ptr, unsigned int nsamps, float tsamp, std::vector<float> dm_list);
  DedispersedTimeSeries<T> operator[](int idx);
  DedispersedTimeSeries<T> nearest_dm(float dm);
};

//created through Channeliser
template <class T>
class FilterbankChannels: public TimeSeriesContainer<T> {

public:
  FilterbankChannel<T> operator[](int idx);
  FilterbankChannel<T> nearest_chan(float freq);
  
};
