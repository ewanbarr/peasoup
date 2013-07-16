#include <data_types/filterbank.hpp>
#include <data_types/timeseries.hpp>
#include <transforms/dedisperser.hpp>
#include <string>


int main(void)
{
  std::string filename("/lustre/projects/p002_swin/surveys/HTRU/medlat/2009-09-12-01:41:41/11/2009-09-12-01:41:41.fil");



  SigprocFilterbank filobj(filename);



  //Dedisperser factory(filobj,7);
  //factory.generate_dm_list(0,1000.0,0.4,1.1);

  return 0;
}
