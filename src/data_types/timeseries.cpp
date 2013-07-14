
using namespace std;

//TimeSeries definition goes here







//TimeSeries containers go here

template <class T>
DispersionTrials::DispersionTrials(T* data_ptr, unsigned int nsamps, 
				   std::vector<float> dm_list_in)
  :data_ptr(data_ptr),
   nsamps(nsamps),
   count(dm_list_in.size())
{
  dm_list.swap(dm_list_in);
}

DedisperedTimeSeries DispersionTrials::operator[](int idx)
{
  //Form DedisperedTimeSeries instance
}
















