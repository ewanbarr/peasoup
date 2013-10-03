#pragma once
#include "cuda.h"
#include "cufft.h"
#include <vector>
#include <utility>
#include <cmath>
#include <complex>
#include "data_types/timeseries.hpp"
#include "utils/exceptions.hpp"
#include "utils/utils.hpp"
#include "transforms/ffter.hpp"
#include <kernels/defaults.h>
#include <kernels/kernels.h>
#include <iostream>
#include <algorithm>
#include <stdio.h>

class FringeFinder {
private:
  char* delay;


};









class DelayFinder {
private:
  char* arrays;
  uint size;
  uint complex_size;
  uint narrays;
  CuFFTerC2C c2cfft;
  std::vector< std::vector<uint> > baselines;

public:
  //chars here are actually complex16                                                                                                                                     
  DelayFinder(char* arrays, uint narrays, uint size)
    :arrays(arrays),complex_size(size),size(size*2),
     narrays(narrays),c2cfft(size){}

  void find_delays(uint max_delay){
    ReusableDeviceTimeSeries<float,char> x(size);
    ReusableDeviceTimeSeries<float,char> y(size);
    
    TimeSeries<char> host_tim(size);
    char* host_tim_ptr = host_tim.get_data();
    
    Utils::device_malloc<char>(&host_tim_ptr,size);
    cufftComplex* return_ptr;

    std::cout<<max_delay<<std::endl;
    Utils::host_malloc<cufftComplex>(&return_ptr,max_delay*2);
    float* abs_return_ptr;
    Utils::host_malloc<float>(&abs_return_ptr,max_delay*2);

    for (int ii=0;ii<narrays;ii++){
      std::cout << "["<<ii<<"] ";
      host_tim.set_data(&arrays[ii*size]);
      x.copy_from_host(host_tim);
      c2cfft.execute((cufftComplex*) x.get_data(),(cufftComplex*) x.get_data(),CUFFT_FORWARD);
      device_conjugate((cufftComplex*) x.get_data(), complex_size, MAX_BLOCKS, MAX_THREADS);

      for (int jj=ii+1;jj<narrays;jj++){
	std::cout << jj << "  ";
        host_tim.set_data(&arrays[jj*size]);
        y.copy_from_host(host_tim);
        c2cfft.execute((cufftComplex*) y.get_data(),(cufftComplex*) y.get_data(),CUFFT_FORWARD);
        device_cuCmulf_inplace((cufftComplex*) x.get_data(),(cufftComplex*) y.get_data(),complex_size, MAX_BLOCKS, MAX_THREADS);
        c2cfft.execute((cufftComplex*) y.get_data(),(cufftComplex*) y.get_data(),CUFFT_INVERSE);
	Utils::d2hcpy<cufftComplex>(return_ptr, (cufftComplex*)y.get_data(), max_delay);
	Utils::d2hcpy<cufftComplex>(return_ptr+max_delay, (cufftComplex*) y.get_data()+(complex_size-max_delay),max_delay);
        for (int kk=0;kk<max_delay*2;kk++){
          abs_return_ptr[kk] = std::pow(return_ptr[kk].x,2) + std::pow(return_ptr[kk].y,2);
	  //printf("%.5f\n", abs_return_ptr[kk]);
	}
	

        int distance = std::distance(abs_return_ptr,std::max_element(abs_return_ptr,abs_return_ptr+max_delay*2));
	std::cout << "Distance:" << distance << std::endl;

      }
      std::cout << std::endl;
    }
  }
};
