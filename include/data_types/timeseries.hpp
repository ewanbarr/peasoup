
//Should move all data pointers to smart pointers std::tr1::shared_pointer
using namespace std;


class TimeSeries {
  
};

class DedispersedTimeSeries: public TimeSeries {
  
};

class FilterbankChannel: public TimeSeries {

};

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
