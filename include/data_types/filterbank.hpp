#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "data_types/header.hpp"
#include "utils/exceptions.hpp"

class Filterbank {
protected:
  //Filterbank metadata
  unsigned char* data;
  unsigned int nsamps;
  unsigned int nchans;
  unsigned char nbits;
  float fch1;
  float foff;
  float tsamp;
  
  //Protected constructors
  Filterbank(unsigned char* data_ptr, unsigned int nsamps,
	     unsigned int nchans, unsigned char nbits,
	     float fch1, float foff, float tsamp)
    :data(data_ptr),nsamps(nsamps),nchans(nchans),
     nbits(nbits),fch1(fch1),foff(foff),tsamp(tsamp){}
  
  Filterbank(void)
    :data(0),nsamps(0),nchans(0),
     nbits(0),fch1(0.0),foff(0.0),tsamp(0.0){}

public:
  virtual float get_tsamp(void){return tsamp;}
  virtual void set_tsamp(float tsamp){this->tsamp = tsamp;}
  virtual float get_foff(void){return foff;}
  virtual void set_foff(float foff){this->foff = foff;}
  virtual float get_fch1(void){return fch1;}
  virtual void set_fch1(float fch1){this->fch1 = fch1;}
  virtual float get_nchans(void){return nchans;}
  virtual void set_nchans(unsigned int nchans){this->nchans = nchans;}
  virtual unsigned int get_nsamps(void){return nsamps;}
  virtual void set_nsamps(unsigned int nsamps){this->nsamps = nsamps;}
  virtual float get_nbits(void){return nbits;}
  virtual void set_nbits(unsigned char nbits){this->nbits = nbits;}
  virtual unsigned char* get_data(void){return this->data;}
  virtual void set_data(unsigned char *data){this->data = data;}
  virtual float get_cfreq(void){return fch1+foff*nchans/2;}

};


class SigprocFilterbank: public Filterbank {
private:
  bool from_file;

public:
  SigprocFilterbank(std::string filename)
    :from_file(true)
  {
    std::ifstream infile;
    SigprocHeader hdr;
    infile.open(filename.c_str(),std::ifstream::in | std::ifstream::binary);
    ErrorChecker::check_file_error(infile, filename);
    read_header(infile,hdr);
    size_t input_size = (size_t) hdr.nsamples*hdr.nbits*hdr.nchans/8;
    this->data = new unsigned char [input_size];
    infile.seekg(hdr.size, std::ios::beg);
    infile.read(reinterpret_cast<char*>(this->data), input_size);
    this->nsamps = hdr.nsamples;
    this->nchans = hdr.nchans;
    this->tsamp = hdr.tsamp;
    this->nbits = hdr.nbits;
    this->fch1 = hdr.fch1;
    this->foff  = hdr.foff;
  }
  
  SigprocFilterbank(unsigned char* data_ptr, unsigned int nsamps,
                    unsigned int nchans, unsigned char nbits,
                    float fch1, float foff, float tsamp)
    :Filterbank(data_ptr,nsamps,nchans,nbits,fch1,foff,tsamp),from_file(false){}
       
  ~SigprocFilterbank()
  {
    if (from_file)
      delete [] this->data;
  }
};
