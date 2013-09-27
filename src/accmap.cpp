#include "utils/utils.hpp"
#include "stdio.h"
#include <vector>
#include "transforms/correlator.hpp"

int main(void){
  size_t size = 33554432;
  size_t nantennas = 16;

  char* arrays;
  Utils::host_malloc<char>(&arrays,size*nantennas*2);

  DelayFinder df(arrays,nantennas,size);
  df.find_delays(2048);

  Utils::host_free(arrays);
  return 0;

}
