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

#define NBINS 10000000
#define BINWIDTH 0.003725290
#define NFOLDS 4

int main()
{
  float* test_pattern;
  Utils::host_malloc<float>(&test_pattern,NBINS);
  for (int ii=0;ii<NBINS;ii++)
    {
      if (ii%32==0)
	test_pattern[ii] = 1;
      else
	test_pattern[ii] = 0;
    }
  
  DevicePowerSpectrum<float> pspec(NBINS,BINWIDTH);
  Utils::h2dcpy<float>(pspec.get_data(),test_pattern,NBINS);
  
  HarmonicSums<float> sums(pspec, NFOLDS);
  HarmonicFolder folder(sums);

  for (int jj=0;jj<100;jj++)
    folder.fold(pspec);
  
  //float* sums0_test_block;
  //Utils::host_malloc<float>(&sums0_test_block,NBINS);

  /*
  for (int jj=0; jj<NFOLDS; jj++)
    {
      Utils::d2hcpy<float>(sums0_test_block,sums[jj]->get_data(),NBINS);
      for (int ii=0;ii<NBINS;ii++)
	{
	  printf("[BIN (%d,%d)] %.4f \n",ii,jj,sums0_test_block[ii]);
	}
      
    }
  */
  Utils::host_free(test_pattern);
  //Utils::host_free(sums0_test_block);
  return 0;
}

