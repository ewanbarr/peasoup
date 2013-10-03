#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "data_types/header.hpp"
#include "utils/exceptions.hpp"
#include "utils/utils.hpp"
#include "data_types/timeseries.hpp"
#include "stdio.h"


class DadaData {
protected:
  char* data_ptr;
  
public:
  DadaHeader header;
  char* get_data(void){return this->data_ptr;}
  void set_data(char *data_ptr){this->data_ptr = data_ptr;}

  void extract_channel(int idx, char* ptr, size_t size, size_t offset=0)
  {
    char* offset_ptr = data_ptr + offset*header.nchan*2;
    for (int ii=0;ii<size;ii++){
      ptr[2*ii]   = offset_ptr[ii*header.nchan*2+2*idx];
      ptr[2*ii+1] = offset_ptr[ii*header.nchan*2+2*idx+1];
    }
  }
};


class DadaFile: public DadaData {
public:
  DadaFile(std::string filename){
    header.fromfile(filename);
    Utils::host_malloc<char>(&data_ptr,header.filesize);
    std::ifstream infile(filename.c_str(),std::ifstream::in | std::ifstream::binary);
    ErrorChecker::check_file_error(infile,filename);
    infile.seekg(header.header_size, infile.beg);
    infile.read(data_ptr,header.filesize);
    ErrorChecker::check_file_error(infile,filename);
  }
  
  ~DadaFile(){
    Utils::host_free(data_ptr);
  }
};


