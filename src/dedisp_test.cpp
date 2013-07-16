#include <data_types/filterbank.hpp>
#include <data_types/timeseries.hpp>
#include <transforms/dedisperser.hpp>
#include <utils/exceptions.hpp>
#include <string>
#include <iostream>

int main(void)
{
  std::string filename("/lustre/projects/p002_swin/surveys/HTRU/medlat/2009-09-12-01:41:41/11/2009-09-12-01:41:41.fil");
  SigprocFilterbank filobj(filename);
  Dedisperser factory(filobj,2);
  factory.generate_dm_list(0,2,0.4,1.1);
  DispersionTrials<unsigned char> trials = factory.dedisperse();
  return 0;
}
