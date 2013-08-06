#pragma once
#include <utils/exceptions.hpp>
#include "cuda.h"
#include "cufft.h"

template <class T>
class FoldedSubints {
private:
  T* data_ptr;
  unsigned int nbins;
  unsigned int nints;
  float period;
  float accel;
  float opt_period;
  int opt_width;
  int opt_bin;
  float opt_sn;

public:
  FoldedSubints(unsigned int nbins, unsigned int nints)
    :data_ptr(0),nbins(nbins),nints(nints),period(0),accel(0)
  {
    cudaError_t error = cudaMalloc((void**)&data_ptr, nbins*nints*sizeof(T));
    ErrorChecker::check_cuda_error(error);
  }
  
  T* get_data(void){return data_ptr;}
  void set_data(T* data_ptr_){data_ptr=data_ptr_;}
  unsigned int get_nbins(void){return nbins;}
  unsigned int get_nints(void){return nints;}
  float get_period(void){return period;}
  float get_accel(void){return accel;}
  void set_period(float p){period=p;}
  void set_accel(float a){accel=a;}
  void set_opt_period(float p){opt_period=p;}
  void set_opt_width(int w){opt_width=w;}
  void set_opt_bin(int bin){opt_bin=bin;}
  float get_opt_period(void){return opt_period;}
  int get_opt_width(void){return opt_width;}
  int get_opt_bin(void){return opt_bin;}
  void set_opt_sn(float sn){opt_sn=sn;}
  float get_opt_sn(void){return opt_sn;}
  

  void change_shape(unsigned int nbins_, unsigned int nints_){
    if ( nbins_*nints_ > nbins*nints ){
      cudaFree(data_ptr);
      cudaError_t error = cudaMalloc((void**)&data_ptr, nbins_*nints_*sizeof(T));
      ErrorChecker::check_cuda_error(error);
      this->nbins = nbins_;
      this->nints = nints_;
    }else{
      this->nbins = nbins_;
      this->nints = nints_;
    }
  }

};
