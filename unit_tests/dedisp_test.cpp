#include <data_types/filterbank.hpp>
#include <data_types/timeseries.hpp>
#include <transforms/dedisperser.hpp>


int main(void)
{
  std::string filename("/lustre/projects/p002_swin/surveys/HTRU/medlat/2009-09-12-01:41:41/11/2009-09-12-01:41:41.fil");
  SigprocFilterbank filobj(filename);
  Dedisperser factory(filobj,7);
  factory.generate_dm_list()

  return 0;
}
