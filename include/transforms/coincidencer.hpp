#pragma once
#include <utils/utils.hpp>
#include <kernels/kernels.h>
#include <kernels/defaults.h>
#include <utils/exceptions.hpp>
#include <string>
#include <stdio.h>

struct Birdie {
public:
  float freq;
  float width;
  Birdie(float freq, float width)
    :freq(freq),width(width){}
};

class Coincidencer {
private:
  float** host_ptr_array;
  float** device_ptr_array;
  int narrays;
  unsigned int max_blocks;
  unsigned int max_threads;
  
public:
  Coincidencer(int narrays, 
	       unsigned int max_blocks=MAX_BLOCKS,
	       unsigned int max_threads=MAX_THREADS)
    :narrays(narrays), max_blocks(max_blocks), max_threads(max_threads)
  {
    Utils::device_malloc<float*>(&device_ptr_array,narrays);
  }
  
  void match(float** host_ptr_array, float* mask, size_t size, float threshold, int beam_threshold)
  {
    Utils::h2dcpy(device_ptr_array, host_ptr_array, narrays);
    device_coincidencer(device_ptr_array,mask,narrays,
			size,threshold,beam_threshold,
			max_blocks,max_threads);
  }

  void write_samp_mask(float* mask, size_t size, std::string filename)
  {
    std::vector<float> h_mask(size);
    Utils::d2hcpy(&h_mask[0],mask,size);
    FILE* fo = fopen(filename.c_str(),"w");
    fprintf(fo,"#0 1\n");
    for (int ii=0;ii<size;ii++)
      fprintf(fo,"%d\n",(int)h_mask[ii]);
    fclose(fo);
  }
    
  void write_birdie_list(float* mask, size_t size, float bin_width, std::string filename)
  {
    std::vector<float> h_mask(size);
    std::vector<Birdie> birdies;
    Utils::d2hcpy(&h_mask[0],mask,size);
    int count;
    int ii = 0;
    int jj = 0;
    while (ii<size){
      if (h_mask[ii]==0){
	count = 0;
	while(h_mask[ii]==0){
	  count++;
	  ii++;
	}
	birdies.push_back(Birdie(((ii-1)-(count/2.0))*bin_width, count*bin_width));
      } else {
	ii++;
      }
    }
    FILE* fo = fopen(filename.c_str(),"w");
    for (ii=0;ii<birdies.size();ii++){
      fprintf(fo,"%.9f\t%.6f\n",birdies[ii].freq,birdies[ii].width);
    }
    fclose(fo);
  }
  
  ~Coincidencer()
  {
    Utils::device_free(device_ptr_array);
  }

};
