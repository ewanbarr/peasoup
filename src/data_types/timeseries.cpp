#include "data_types/timeseries.hpp"

using namespace std;

//TimeSeries definition goes here
template <class T>
TimeSeries::TimeSeries(T* data_ptr, unsigned int nsamps, float tsamp)
:data_ptr(data_ptr), nsamps(nsamps), tsamp(tsamp)
{
}

T TimeSeries::operator[](int idx)
{
   return data_ptr[idx];	
}

void TimeSeries::set_nsamps(unsigned int nsamps_)
{
  nsamps = nsamps_;
}

unsigned int TimeSeries::get_nsamps(void)
{
  return nsamps;
}

void TimeSeries::set_tsamp(float tsamp_)
{
  tsamp = tsamp_;
}

float TimeSeries::get_tsamp(void)
{
  return tsamp;
}

template <class T>
DedisperedTimeSeries::DedispersedTimeSeries(T* data_ptr, unsigned int nsamps, float tsamp, float dm)
:TimeSeries<T>(data_ptr,nsamps,tsamp),dm(dm)
{
}

float DedisperedTimeSeries::get_dm(void)
{
  return dm;
}

void DedisperedTimeSeries::set_dm(float dm_)
{
  dm = dm_;
}


template <class T>
DispersionTrials::DispersionTrials(T* data_ptr, unsigned int nsamps, 
				   float tsamp, std::vector<float> dm_list_in)
  :TimeSeriesContainer<T>(data_ptr,nsamps,tsamp,dm_list_in.size())
{
  dm_list.swap(dm_list_in);
}

DedisperedTimeSeries DispersionTrials::operator[](int idx)
{
   return DedisperedTimeSeries<T>(data_ptr+idx*nsamps, nsamps, tsamp, dm_list[idx]);
}
















