#include "utils/utils.hpp"
#include "stdio.h"
#include <vector>
#include "transforms/correlator.hpp"
#include "data_types/dada.hpp"
#include "data_types/header.hpp"
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

int main(void){
  size_t size = 65536;
  size_t nantennas = 2;
  uint channel = 0;
  char* arrays;
  //DadaFilterbank dada;
    
  Utils::host_malloc<char>(&arrays,size*nantennas*2);
  std::string filename("/lustre/projects/p002_swin/ebarr/CORRELATION_TESTS/tocopy.dada");
  DadaFile data(filename);
  
  
  data.extract_channel(channel, arrays, size);
  data.extract_channel(channel, arrays+size*2, size, 10);
  DelayFinder df(arrays,nantennas,size);
  df.find_delays(2048);

  Utils::host_free(arrays);
  return 0;

}
