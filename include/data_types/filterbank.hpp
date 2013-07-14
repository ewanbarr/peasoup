#pragma once
#include <string>
#include 

using namespace std;

class Filterbank {
protected:
  //Filterbank metadata
  float tsamp;
  float foff;
  float cfreq;
  unsigned int nchans;
  unsigned int nsamps;
  unsigned char nbits;
  
  //Filterbank data pointer
  unsigned char* data;
  
public:
  //Setter/Getters
  virtual float get_tsamp(void);
  virtual void set_tsamp(float tsamp);

  virtual float get_foff(void);
  virtual void set_foff(float foff);

  virtual float get_cfreq(void);
  virtual void set_cfreq(float cfreq);

  virtual float get_nchans(void);
  virtual void set_nchans(unsigned int nchans);

  virtual float get_nsamps(void);
  virtual void set_nsamps(unsigned int nsamps);

  virtual float get_nbits(void);
  virtual void set_nbits(unsigned char nbits);

  virtual unsigned char* get_data(void);
  virtual void set_data(unsigned char*);
};

class SigprocFilterbank: public Filterbank {
private:
  bool from_file;

public:
  SigprocFilterbank(std::string filename);
  SigprocFilterbank(unsigned char* data_ptr, unsigned int nsamps,
                    unsigned int nchans, unsigned char nbits,
                    float cfreq, float foff, float tsamp);    
  ~SigprocFilterbank();
};
