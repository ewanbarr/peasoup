#pragma once
#include "data_types/fourierseries.hpp"
#include "kernels/kernels.h"
#include "kernels/defaults.h"
#include "utils/utils.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <iterator>

class Zapper {
private:
  bool d_mem_allocated;
  std::vector<float> birdies;
  std::vector<float> widths;
  float* d_birdies; //device memory
  float* d_widths; //device memory

  std::vector<std::string> split(std::string const &input) { 
    std::stringstream buffer(input);
    std::vector<std::string> ret;
    std::copy(std::istream_iterator<std::string>(buffer), 
              std::istream_iterator<std::string>(),
              std::back_inserter(ret));
    return ret;
  }
  
public:
  Zapper(std::string zaplist)
  {
    d_mem_allocated = false;
    append_from_file(zaplist);
  }
  
  void append_from_file(std::string zaplist){
    std::string line;
    std::ifstream infile(zaplist.c_str());
    ErrorChecker::check_file_error(infile, zaplist);
    while (std::getline(infile,line)){
      std::vector<std::string> split_line = split(line);
      if (split_line.size()>0){
	birdies.push_back(::atof(split_line[0].c_str()));
	widths.push_back(::atof(split_line[1].c_str()));
      }
    }
    infile.close();

    if (d_mem_allocated){
      Utils::device_free(d_birdies);
      Utils::device_free(d_widths);
    } else {
      Utils::device_malloc<float>(&d_birdies,birdies.size());
      Utils::device_malloc<float>(&d_widths,birdies.size());
      d_mem_allocated=true;
    }
    Utils::h2dcpy(d_birdies,&birdies[0],birdies.size());
    Utils::h2dcpy(d_widths,&widths[0],widths.size());

  }

  void zap(DeviceFourierSeries<cufftComplex>& fseries){
    float bin_width = fseries.get_bin_width();
    unsigned int nbins = fseries.get_nbins();
    zap(fseries.get_data(),bin_width,nbins);
  }
  
  void zap(cufftComplex* fseries, float bin_width, unsigned int nbins){
    device_zap_birdies(fseries, d_birdies, d_widths,
                       bin_width, birdies.size(), nbins,
                       MAX_BLOCKS, MAX_THREADS);
  }
    
};
