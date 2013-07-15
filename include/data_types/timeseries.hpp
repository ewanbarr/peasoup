//Should move all data pointers to smart pointers std::tr1::shared_pointer
using namespace std;

//######################

template <class T> class TimeSeries {
protected:
  unsigned int nsamps;
  float tsamp;
  T data_ptr;
  
public:  
  template <class T> TimeSeries(T* data_ptr,unsigned int nsamps,float tsamp);
  operator[](int idx);
  unsigned int get_nsamps(void);
  void set_nsamps(unsigned int nsamps);
  float get_tsamp(void);
  void set_tsamp(float tsamp);
};

//#########################

class DedispersedTimeSeries: public TimeSeries {
private:
  float dm;

public:
  template<class T>
  DedispersedTimeSeries(T* data_ptr, unsigned int nsamps, float tsamp, float dm);
  float get_dm(void);
  void set_dm(float dm);
};

//###########################

class FilterbankChannel: public TimeSeries {

};

//#############################






template <class T>
class TimeSeriesContainer {
protected:
  unsigned int nsamps;
  unsigned int count;
  float tsamp;
  T* data_ptr;
  
};




//created through Dedisperser
template <class T>
class DispersionTrials: public TimeSeriesContainer <T> {
private:
  std::vector<float> dm_list;
  
public:
  DispersionTrials(T* data_ptr, unsigned int nsamps, std::vector<float> dm_list);
  DedisperedTimeSeries operator[](int idx);
  DedisperedTimeSeries nearest_dm(float dm);
};

//created through Channeliser
class FilterbankChannels: public TimeSeriesContainer{

public:
  FilterbankChannel operator[](int idx);
  FilterbankChannel nearest_chan(float freq);
  
}
