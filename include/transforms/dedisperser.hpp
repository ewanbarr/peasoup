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
                std::size_t input_start_sample, std::size_t input_end_sample,
                std::size_t gulp)
{
    // Calculate the effective number of input samples to process
    std::size_t effective_nsamps = input_end_sample - input_start_sample;
    std::vector<float> temp_buffer;

    // Get the maximum delay introduced by dedispersion
    std::size_t max_delay = dedisp_get_max_delay(plan);

    // Adjust the gulp size if necessary
    if (gulp < 2 * max_delay)
    {
        gulp = 2 * max_delay;
        std::cerr << "WARNING: Gulp size adjusted to " << gulp << " samples." << std::endl;
    }

    // Calculate the total number of output samples after dedispersion
    std::size_t total_out_nsamps = effective_nsamps - max_delay;

    // Resize the trials object to hold the dedispersed data
    trials.resize(total_out_nsamps, dm_list);

    // Initialize input and output sample indices
    std::size_t input_current_sample = input_start_sample;
    std::size_t output_current_sample = 0;

    while (input_current_sample < input_end_sample - max_delay)
    {
        // Calculate the number of samples to load in this gulp
        std::size_t samples_to_load = std::min(gulp, input_end_sample - input_current_sample);

        // Load the data from the filterbank
        std::size_t loaded_samples = filterbank.load_gulp(input_current_sample, samples_to_load);
        if (loaded_samples == 0)
        {
            throw std::runtime_error("Failed to read data during dedispersion, 0 samples read.");
        }

        // Calculate the number of samples output by dedispersion
        std::size_t dedisp_samples = loaded_samples - max_delay;

        // Determine the number of samples to copy to the output buffer
        std::size_t nsamps_to_copy = std::min(dedisp_samples, total_out_nsamps - output_current_sample);

        // Resize the temporary buffer to hold the dedispersed data
        temp_buffer.resize(dm_list.size() * dedisp_samples);

        // Execute dedispersion
        dedisp_error error = dedisp_execute(
            plan,
            loaded_samples,
            filterbank.get_data(),
            filterbank.get_nbits(),
            reinterpret_cast<unsigned char*>(temp_buffer.data()),
            32, // Float output
            0);
        ErrorChecker::check_dedisp_error(error, "execute");

        // Get a pointer to the trials data
        float* ptr = reinterpret_cast<float*>(trials.get_data_ptr());

        // Loop over each DM trial
        for (std::size_t trial_idx = 0; trial_idx < dm_list.size(); ++trial_idx)
        {
            // Calculate the destination offset in the trials buffer
            std::size_t dest_offset = total_out_nsamps * trial_idx + output_current_sample;

            // Calculate the source offset in the temp_buffer
            std::size_t src_offset = dedisp_samples * trial_idx;

            // Copy the dedispersed data from temp_buffer to trials
            std::memcpy(
                ptr + dest_offset,
                temp_buffer.data() + src_offset,
                sizeof(float) * nsamps_to_copy);
        }

        // Update the input and output sample indices
        input_current_sample += dedisp_samples;
        output_current_sample += nsamps_to_copy;
    }
}

};














