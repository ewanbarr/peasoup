#include <data_types/fourierseries.hpp>
#include <kernels/kernels.h>
#include <kernels/defaults.h>
#include <iostream>

class HarmonicFolder {
private:
  unsigned int max_blocks;
  unsigned int max_threads;

public:
  HarmonicFolder(unsigned int max_blocks=MAX_BLOCKS,
		 unsigned int max_threads=MAX_THREADS)
    :max_blocks(max_blocks),max_threads(max_threads){}
  
  void fold(DevicePowerSpectrum<float>& fold0, HarmonicSums<float>& sums)
  {
    for (int ii=0;ii<sums.size();ii++)
      {

	std::cout << fold0.get_data() 
		  << "\t" << sums[ii]
		  <<"\t"<<  fold0.get_nbins()
		  <<"\t"<<  pow(2,ii+1)
		  <<"\t"<< max_blocks
		  <<"\t"<< max_threads << std::endl;
	
	device_harmonic_sum(fold0.get_data(),
			    sums[ii],
			    fold0.get_nbins(),
			    pow(2,ii+1),
			    max_blocks,
			    max_threads);
      }

  }
  void fold(DevicePowerSpectrum<float>& fold0, DevicePowerSpectrum<float>& output, unsigned int harm)
  {
    device_harmonic_sum(fold0.get_data(),
			output.get_data(),
			fold0.get_nbins(),
			harm,
			max_blocks,
			max_threads);
  }

};
