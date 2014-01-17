#pragma once
#include <utils/exceptions.hpp>
#include <utils/utils.hpp>
#include "cuda.h"
#include "cufft.h"
#include <iostream>
#include <vector>

template <class T>
class FoldedSubints {
private:
  T* data_ptr;
  unsigned int nbins;
  unsigned int nints;
  double period;
  float accel;
  double opt_period;
  int opt_width;
  int opt_bin;
  float opt_sn;
  float tobs;

public:
  std::vector<float> opt_fold;
  std::vector<float> opt_prof;
  FoldedSubints(unsigned int nbins, unsigned int nints)
    :data_ptr(0),nbins(nbins),nints(nints),period(0),accel(0)
  {
    Utils::device_malloc<T>(&data_ptr,nbins*nints);
  }
  
  T* get_data(void){return data_ptr;}
  void set_data(T* data_ptr_){data_ptr=data_ptr_;}
  unsigned int get_nbins(void){return nbins;}
  unsigned int get_nints(void){return nints;}
  double get_period(void){return period;}
  float get_accel(void){return accel;}
  void set_period(double p){period=p;}
  void set_accel(float a){accel=a;}
  void set_opt_period(double p){opt_period=p;}
  void set_opt_width(int w){opt_width=w;}
  void set_opt_bin(int bin){opt_bin=bin;}
  float get_opt_period(void){return opt_period;}
  int get_opt_width(void){return opt_width;}
  int get_opt_bin(void){return opt_bin;}
  void set_opt_sn(float sn){opt_sn=sn;}
  float get_opt_sn(void){return opt_sn;}
  void set_tobs(float tobs_){tobs=tobs_;}
  float get_tobs(void){return tobs;}

  void set_opt_fold(float* ar, size_t size){
    opt_fold.resize(size);
    for (int ii=0;ii<size;ii++)
      opt_fold[ii] = ar[ii];
  }

  void set_opt_prof(float* ar, size_t size){
    opt_prof.resize(size);
    for (int ii=0;ii<size;ii++)
      opt_prof[ii] = ar[ii];
  }

  ~FoldedSubints(){
    Utils::device_free(data_ptr);
  }
};
