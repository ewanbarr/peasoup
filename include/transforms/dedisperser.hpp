#pragma once
#include "dedisp.h"
#include <cstdlib>
#include <vector>
#include <string>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <data_types/timeseries.hpp>
#include <data_types/filterbank.hpp>
#include <utils/exceptions.hpp>

class Dedisperser {
private:
  dedisp_plan plan;
  Filterbank& filterbank;
  unsigned int num_gpus;
  std::vector<float> dm_list;
  std::vector<dedisp_bool> killmask;

public:
  Dedisperser(Filterbank& filterbank, unsigned int num_gpus=1)
    :filterbank(filterbank), num_gpus(num_gpus)
  {
    killmask.resize(filterbank.get_nchans(),1);
    dedisp_error error = dedisp_create_plan_multi(&plan,
						  filterbank.get_nchans(),
						  filterbank.get_tsamp(),
						  filterbank.get_fch1(),
						  filterbank.get_foff(),
						  num_gpus);
    ErrorChecker::check_dedisp_error(error,"create_plan_multi");
  }

  virtual ~Dedisperser()
  {
    if( plan ) {
      dedisp_destroy_plan(plan);
    }
  }

  void set_dm_list(float* dm_list_ptr, unsigned int ndms)
  {
    dm_list.resize(ndms);
    std::copy(dm_list_ptr, dm_list_ptr+ndms, dm_list.begin());
    dedisp_error error = dedisp_set_dm_list(plan,&dm_list[0],dm_list.size());
    ErrorChecker::check_dedisp_error(error,"set_dm_list");
  }

  void set_dm_list(std::vector<float> dm_list_vec)
  {
    dm_list.resize(dm_list_vec.size());
    std::copy(dm_list_vec.begin(), dm_list_vec.end(), dm_list.begin());
    dedisp_error error = dedisp_set_dm_list(plan,&dm_list[0],dm_list.size());
    ErrorChecker::check_dedisp_error(error,"set_dm_list");
  }

  std::vector<float> get_dm_list(void){
    return dm_list;
  }

  void generate_dm_list(float dm_start, float dm_end,
			float width, float tolerance)
  {
    dedisp_error error = dedisp_generate_dm_list(plan, dm_start, dm_end, width, tolerance);
    ErrorChecker::check_dedisp_error(error,"generate_dm_list");
    dm_list.resize(dedisp_get_dm_count(plan));
    const float* plan_dm_list = dedisp_get_dm_list(plan);
    std::copy(plan_dm_list,plan_dm_list+dm_list.size(),dm_list.begin());
  }

  void set_killmask(std::vector<int> killmask_in)
  {
    killmask.swap(killmask_in);
    dedisp_error error = dedisp_set_killmask(plan,&killmask[0]);
    ErrorChecker::check_dedisp_error(error,"set_killmask");
  }

  void set_killmask(std::string filename)
  {
    std::ifstream infile;
    std::string str;
    killmask.clear();
    infile.open(filename.c_str(),std::ifstream::in | std::ifstream::binary);
    ErrorChecker::check_file_error(infile,filename);

    int ii=0;
    while(!infile.eof()&&ii<filterbank.get_nchans()){
      std::getline(infile, str);
      killmask.push_back(std::atoi(str.c_str()));
      ii++;
    }

    if (killmask.size() != filterbank.get_nchans()){
      std::cerr << "WARNING: killmask is not the same size as nchans" << std::endl;
      std::cerr << killmask.size() <<" != " <<  filterbank.get_nchans() <<  std::endl;
      killmask.resize(filterbank.get_nchans(),1);
    } else {
      dedisp_error error = dedisp_set_killmask(plan,&killmask[0]);
      ErrorChecker::check_dedisp_error(error,"set_killmask");
    }

  }

  void dedisperse(DispersionTrials<float>& trials,
                  std::size_t start_sample, std::size_t nsamps,
                  std::size_t gulp)
  {
    std::vector<float> temp_buffer;
    std::size_t max_delay = dedisp_get_max_delay(plan);
    std::cout << "Max DM delay: " << max_delay << std::endl;
    if (gulp < 2 * max_delay)
    {
      gulp = 2 * max_delay;
      std::cerr << "WARNING: Gulp size < 2 x maximum DM delay, adjusting gulp size to "
                << gulp << " bytes"<< std::endl;
    }

    if ((start_sample + nsamps) > filterbank.get_effective_nsamps())
    {
      nsamps = filterbank.get_effective_nsamps() - start_sample;
      std::cerr << "WARNING: Number of sample requested exceeds input filterbank length "
                << "revising from" << (start_sample + nsamps) << "to " << nsamps << " samples" << std::endl;
    }

    // Calculated the total number of output samples expected
    std::size_t total_out_nsamps = nsamps - max_delay;
    std::cout << "Total Dedisp output samples: " << total_out_nsamps << std::endl;

    // Create a complete trials object to contain all trials at full length

    trials.resize(total_out_nsamps, dm_list);


    while (start_sample < total_out_nsamps)
    {
      std::cout << "Dedispersing samples " << start_sample
	        << " to " << start_sample + gulp << " of "
                << total_out_nsamps << std::endl;
      // Load a block of data from the filterbank
      std::size_t loaded_samples = filterbank.load_gulp(start_sample, gulp);
      if (loaded_samples == 0)
      {
          throw std::runtime_error("Failure on reading data during dedispersion, 0 bytes read.");
      }

      // Calculate the expected number of output samples from a dedisp call
      std::size_t dedisp_samples = loaded_samples - max_delay;
      //std::cout << "Dedisp output samples from block: " << dedisp_samples << std::endl;

      // Calculate the actual number of samples to memcpy
      std::size_t nsamps_to_copy;
      if (dedisp_samples + start_sample > total_out_nsamps){
        nsamps_to_copy = total_out_nsamps - start_sample;
      } else {
 	nsamps_to_copy = dedisp_samples;
      }
      // Resize the temporary buffer to handle the output of the next dedisp call
      temp_buffer.resize(gulp * dm_list.size());

      // Run Dedisp with output into the temporary buffer
      //std::cout << "Calling Dedisp" << std::endl;
      dedisp_error error = dedisp_execute(plan,
          loaded_samples,
          filterbank.get_data(),  //This pointer gets set in the filterband.load_gulp method
          filterbank.get_nbits(),
          reinterpret_cast<unsigned char*>(temp_buffer.data()),
          32, // Float output
          (unsigned) 0);
      ErrorChecker::check_dedisp_error(error,"execute");

      // Get a pointer to the final trials data
      std::vector<float> const& data = trials.get_data();
      std::cout << "Trials total size: " << data.size() * sizeof(float) << " bytes" << std::endl;
      float* ptr = reinterpret_cast<float*>(trials.get_data_ptr());

      // Loop over the trials and for each take the data from the temporary buffer
      // and memcpy it into the correct location in the final trials object
      std::cout << "Performing transpose/merge of Dedisp output samples" << std::endl;

      for (std::size_t trial_idx = 0; trial_idx < dm_list.size(); ++trial_idx)
      {
        // Calculate destination offset for trails pointer
        //std::cout << "Trial IDx " << trial_idx << std::endl;
        std::size_t offset = total_out_nsamps * trial_idx + start_sample;
        //std::cout << "Offset " << offset << std::endl;
        //std::cout << "Temp offset " << dedisp_samples * trial_idx << std::endl;
        //std::cout << "Trials size " << trials.get_data().size() << std::endl;
        //std::cout << "Temp size " << temp_buffer.size() << std::endl;
        //std::cout << "nsamps to copy " << nsamps_to_copy << std::endl;
        
        //std::cout << "Dest offset: " << offset * sizeof(float) 
        //          << " size: " << sizeof(float) * trials.get_count() * trials.get_nsamps()  
        //          << " remaining: " << sizeof(float) * trials.get_count() * trials.get_nsamps() - offset * sizeof(float)
        //          << " to_copy: " << sizeof(float) * nsamps_to_copy << std::endl;
        


        std::memcpy( reinterpret_cast<char*>(ptr + offset),
                     reinterpret_cast<char*>(temp_buffer.data() + dedisp_samples * trial_idx),
                     sizeof(float) * nsamps_to_copy);
      }

      // Update the start_sample based on the number of samples output by dedisp
      start_sample += dedisp_samples;
      //std::cout << "Updating start sample to " << start_sample << std::endl;
    }
    
  }
};














