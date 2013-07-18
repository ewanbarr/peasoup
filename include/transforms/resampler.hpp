#include <data_types/timeseries.hpp>

class DeviceResampler {
private:
  DeviceTimeSeries& input;
  DeviceTimeSeries output;

public:
  DeviceResampler(DeviceTimeSeries& tim)
    :input(tim)
  {
    output(tim.get_namps());
  }
  
  DeviceTimeSeries resample(float acc, float jerk=0)
  {
    return output;
  }
   
};
