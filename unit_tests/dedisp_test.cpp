#include <data_types/filterbank.hpp>
#include <data_types/timeseries.hpp>
#include <transforms/dedisperser.hpp>
#include <string>
#include <iostream>

int main(void)
{
  std::string filename("/lustre/projects/p002_swin/surveys/HTRU/medlat/2009-09-12-01:41:41/11/2009-09-12-01:41:41.fil");
  SigprocFilterbank filobj(filename);
  std::cout << (unsigned int) filobj.get_data()[0] << std::endl;
  Dedisperser factory(filobj,2);
  factory.generate_dm_list(0,2,0.4,1.1);
  std::cout << factory.get_dm_list().size() << std::endl;
  DispersionTrials<unsigned char> trials = factory.dedisperse();
  std::cout << (unsigned int) trials[1][345] << std::endl;

  return 0;
}
