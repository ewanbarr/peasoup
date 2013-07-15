#include "dedisp.h"
#include <copy>
#include <string>
#include <sstream>

using namespace std;

template <class FilterbankDerivative> 
Dedisperser::Dedisperser(FilterbankDerivative& filterbank, unsigned int num_gpus)
  :filterbank(filterbank),
   num_gpus(num_gpus)
{
  killmask.resize(filterbank.get_nchans(),1);
  dedisp_error error = dedisp_create_plan_multi(&plan,
						filterbank.get_nchans(),
						filterbank.get_tsamp(),
						filterbank.get_fch1(),
						filterbank.get_foff(),
						num_gpus);
  check_dedisp_error(error,"create_plan_multi");
}

void Dedisperser::set_dm_list(float* dm_list_ptr, unsigned int ndms)
{
  dm_list.resize(ndms);
  std::copy(dm_list_ptr, dm_list_ptr+ndms, dm_list.begin());
  dedisp_error error = dedisp_set_dm_list(plan,&dm_list[0],dm_list.size());
  check_dedisp_error(error,"set_dm_list");
}

void Dedisperser::set_dm_list(std::vector<float> dm_list_vec)
{
  dm_list.resize(dm_list_vec.size());
  std::copy(dm_list_vec.begin(), dm_list_vec.end(), dm_list.begin());
  dedisp_error error = dedisp_set_dm_list(plan,&dm_list[0],dm_list.size());
  check_dedisp_error(error,"set_dm_list");
}

std::vector<float> Dedisperser::get_dm_list(void)
{
  return dm_list;
}

void Dedisperser::generate_dm_list(float dm_start, float dm_end,
				   float width, float tolerance)
{
  dedisp_error error = dedisp_generate_dm_list(plan, dm_start, dm_end,
					       width, tolerance);
  check_dedisp_error(error,"generate_dm_list");
  dm_list.resize(plan->dm_count);
  std::copy(plan->dm_list.begin(),plan->dm_list.end(),dm_list.begin());
}

void Dedisperser::set_killmask(std::vector<int> killmask_in)
{
  killmask.swap(killmask_in);
  dedisp_error error = dedisp_set_killmask(plan,&killmask[0]);
  check_dedisp_error(error,"set_killmask");
}

void Dedisperser::set_killmask(std::string filename)
{
  std::ifstream infile;
  std::string str;
  killmask.clear();
  infile.open(filename.c_str(),std::ifstream::in | std::ifstream::binary);
  if(infile.bad())
    {
      std::stringstream error_msg;
      error_msg << "File "<< filename << " could not be opened: ";
      if ( (infile.rdstate() & std::ifstream::failbit ) != 0 )
        error_msg << "Logical error on i/o operation" << std::endl;
      if ( (infile.rdstate() & std::ifstream::badbit ) != 0 )
        error_msg << "Read/writing error on i/o operation" << std::endl;
      if ( (infile.rdstate() & std::ifstream::eofbit ) != 0 )
        error_msg << "End-of-File reached on input operation" << std::endl;
      throw std::runtime_error(error_msg.str());
    }  
  while(!infile.eof()){
    std::getline(infile, str);
    killmask.push_back(atoi(str.c_str()));
  }
  if (killmask.size() != filterbank.get_nchans()){
    std::cerr << "WARNING: killmask is not the same size as nchans" << std::endl;
    killmask.resize(filterbank.get_nchans(),1);
  } else {
    dedisp_error error = dedisp_set_killmask(plan,&killmask[0]);
    check_dedisp_error(error,"set_killmask");
  }
}

void Dedisperser::dedisperse(void){
  //Currently hardwired for non-subbanded dedispersion
  //with 8-bit output and no fancy flags.
  size_t max_delay = dedisp_get_max_delay(plan);
  unsigned int out_nsamps = filterbank.get_nsamps()-max_delay;
  size_t output_size = out_nsamps * dm_list.size();
  data_ptr = new unsigned char [output_size];
  dedisp_error error = dedisp_execute(plan,
				      filterbank.get_nsamps(),
				      filterbank.get_data()
				      filterbank.get_nbits(),
				      data_ptr,8,(unsigned)0);
  this->check_dedisp_error(error,"execute");
  DispersionTrials ddata(data_ptr,out_nsamps,dm_list);
  return ddata;
}

static void Dedisperser::check_dedisp_error(dedisp_error error,
					    std::string function_name);
{
  std::stringstream error_msg;
  if (error != DEDISP_NO_ERROR)
    {
      error_msg << function_name << " failed with DEDISP error: "
		<< dedisp_get_error_string(error) << std::endl;
      throw std::runtime_error(error_msg.str());
    }
}


