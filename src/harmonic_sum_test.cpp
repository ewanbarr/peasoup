#include <transforms/harmonicfolder.hpp>
#include <data_types/fourierseries.hpp>
#include <utils/exceptions.hpp>
#include <utils/utils.hpp>
#include <utils/stopwatch.hpp>
#include <string>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include "cuda.h"
#include "cufft.h"

#define NBINS 4194304
#define BINWIDTH 0.003725290
#define NFOLDS 4

int main()
{
  float* test_pattern;
  Utils::host_malloc<float>(&test_pattern,NBINS);
  for (int ii=0;ii<NBINS;ii++)
    {
      test_pattern[ii] = ii%451;
    }
  
  DevicePowerSpectrum<float> pspec(NBINS,BINWIDTH);
  Utils::h2dcpy<float>(pspec.get_data(),test_pattern,NBINS);
  
  HarmonicFolder folder;

  HarmonicSums<float> sums0(pspec, NFOLDS);
  folder.fold4(pspec,sums0);
  
  HarmonicSums<float> sums1(pspec, NFOLDS);
  folder.fold(pspec,sums1);

  float* sums0_test_block;
  Utils::host_malloc<float>(&sums0_test_block,NBINS);
  
  float* sums1_test_block;
  Utils::host_malloc<float>(&sums1_test_block,NBINS);

  for (int jj=0; jj<NFOLDS; jj++)
    {
      Utils::d2hcpy<float>(sums0_test_block,sums0[jj]->get_data(),NBINS);
      Utils::d2hcpy<float>(sums1_test_block,sums1[jj]->get_data(),NBINS);
      for (int ii=0;ii<NBINS;ii++)
	{
	  
	  if (fabs(sums0_test_block[ii] - sums1_test_block[ii]) > 0.001)
	    printf("[WRONG (%d,%d)] %.4f  !=  %.4f\n",ii,jj,sums0_test_block[ii],sums1_test_block[ii]);
	    
	}
      
    }
  


    

  Utils::host_free(test_pattern);
  Utils::host_free(sums0_test_block);
  Utils::host_free(sums1_test_block);


  return 0;
}

