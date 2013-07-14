#include <fstream>
#include "data_types/filterbank.hpp"
#include "data_types/header.hpp"

float Filterbank::get_tsamp(void){return tsamp;}
void  Filterbank::set_tsamp(float tsamp){this->tsamp = tsamp;}

float Filterbank::get_foff(void){return foff;}
void  Filterbank::set_foff(float foff){this->foff = foff;}

float Filterbank::get_cfreq(void){return cfreq;}
void  Filterbank::set_cfreq(float cfreq){this->cfreq = cfreq;}

float Filterbank::get_nchans(void){return nchans;}
void  Filterbank::set_nchans(unsigned int nchans){this->nchans = nchans;}

float Filterbank::get_nsamps(void){return nsamps;}
void  Filterbank::set_nsamps(unsigned int nsamps){this->nsamps = nsamps;}

float Filterbank::get_nbits(void){return nbits;}
void  Filterbank::set_nbits(unsigned char nbits){this->nbits = nbits;}

unsigned char* Filterbank::get_data(void){return data;}
void Filterbank::set_data(unsigned char*){this->data = data;}

SigprocFilterbank::SigprocFilterbank(std::string filename)
  :from_file(true)
{
  std::ifstream infile;
  SigprocHeader hdr;
  infile.open(filename.c_str(),std::ifstream::in | std::ifstream::binary);
  if(infile.bad())
    {
      std::cerr << "File "<< filename << " could not be opened " << std::endl;
      if ( (infile.rdstate() & std::ifstream::failbit ) != 0 )
	std::cerr << "Logical error on i/o operation" << std::endl;
      if ( (infile.rdstate() & std::ifstream::badbit ) != 0 )
	std::cerr << "Read/writing error on i/o operation" << std::endl;
      if ( (infile.rdstate() & std::ifstream::eofbit ) != 0 )
	std::cerr << "End-of-File reached on input operation" << std::endl;
      exit(-1);
    }
  read_header(infile,hdr);
  input_size = (size_t) hdr.nsamples * hdr.nbits * hdr.nchans / 8;
  this->data = new unsigned char [(size_t) hdr.nsamples*hdr.nbits*hdr.nchans/8];
  this->nsamps = hdr.nsamples;
  this->nchans = hdr.nchans;
  this->tsamp = hdr.tsamp;
  this->nbits = hdr.nbits;
  this->cfreq = hdr.cfreq;
  this->foff  = hdr.foff;
}

SigprocFilterbank::SigprocFilterbank(unsigend char* data_ptr, unsigned int nsamps,
				     unsigned int nchans, unsigned char nbits,
				     float cfreq, float foff, float tsamp)
  :from_file(false),data(data_ptr),nsamps(nsamps),nchans(nchans),
   nbits(nbits),cfreq(cfreq),foff(foff),tsamp(tsamp)
{
}

SigprocFilterbank::~SigprocFilterbank()
{
  if (from_file)
    delete [] this->data;
}
