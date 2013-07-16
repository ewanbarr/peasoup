#include "data_types/timeseries.hpp"
#include <vector>

using namespace std;

//TimeSeries definition goes here
template <class T>
TimeSeries<T>::TimeSeries(T* data_ptr, unsigned int nsamps, float tsamp)
:data_ptr(data_ptr), nsamps(nsamps), tsamp(tsamp)
{
}

template <class T>
T TimeSeries<T>::operator[](int idx)
{
   return data_ptr[idx];	
}

template <class T>
void TimeSeries<T>::set_nsamps(unsigned int nsamps_)
{
  nsamps = nsamps_;
}

template <class T>
unsigned int TimeSeries<T>::get_nsamps(void)
{
  return nsamps;
}

template <class T>
void TimeSeries<T>::set_tsamp(float tsamp_)
{
  tsamp = tsamp_;
}

template <class T>
float TimeSeries<T>::get_tsamp(void)
{
  return tsamp;
}

template <class T>
DedispersedTimeSeries<T>::DedispersedTimeSeries(T* data_ptr, unsigned int nsamps, float tsamp, float dm)
 :TimeSeries<T>(data_ptr,nsamps,tsamp),dm(dm)
{
}

template <class T>
float DedispersedTimeSeries<T>::get_dm(void)
{
  return dm;
}

template <class T>
void DedispersedTimeSeries<T>::set_dm(float dm_)
{
  dm = dm_;
}

template <class T> 
TimeSeriesContainer<T>::TimeSeriesContainer(T* data_ptr, unsigned int nsamps,
					 float tsamp, unsigned int count)
  :data_ptr(data_ptr),nsamps(nsamps),tsamp(tsamp),count(count)
{
}

template <class T>
DispersionTrials<T>::DispersionTrials(T* data_ptr, unsigned int nsamps, 
				      float tsamp, std::vector<float> dm_list_in)
  :TimeSeriesContainer<T>(data_ptr,nsamps,tsamp, (unsigned int)dm_list_in.size())
{
  dm_list.swap(dm_list_in);
}

template <class T>
DedispersedTimeSeries<T> DispersionTrials<T>::operator[](int idx)
{
  return DedispersedTimeSeries<T>(this->data_ptr+idx*this->nsamps, this->nsamps, this->tsamp, dm_list[idx]);
}















