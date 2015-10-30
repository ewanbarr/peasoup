#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <ctime>
#include <iostream>
#include <map>
#include <fstream>
#include <utils/xml_util.hpp>
#include <utils/cmdline.hpp>
#include <utils/stopwatch.hpp>
#include <data_types/header.hpp>
#include "cuda.h"

class OutputFileWriter {
  XML::Element root;

public:
  OutputFileWriter()
    :root("peasoup_search"){}

  std::string to_string(void){
    return root.to_string(true);
  }

  void to_file(std::string filename){
    std::ofstream outfile;
    outfile.open(filename.c_str(),std::ifstream::out | std::ifstream::binary);
    ErrorChecker::check_file_error(outfile, filename);
    outfile << root.to_string(true);
    ErrorChecker::check_file_error(outfile, filename);
    outfile.close();
  }
  
  void add_header(std::string filename){
    std::ifstream infile;
    SigprocHeader hdr;
    infile.open(filename.c_str(),std::ifstream::in | std::ifstream::binary);
    ErrorChecker::check_file_error(infile, filename);
    read_header(infile,hdr);
    XML::Element header("header_parameters");
    header.append(XML::Element("source_name",hdr.source_name));
    header.append(XML::Element("rawdatafile",hdr.rawdatafile));
    header.append(XML::Element("az_start",hdr.az_start));
    header.append(XML::Element("za_start",hdr.za_start));
    header.append(XML::Element("src_raj",hdr.src_raj));
    header.append(XML::Element("src_dej",hdr.src_dej));
    header.append(XML::Element("tstart",hdr.tstart));
    header.append(XML::Element("tsamp",hdr.tsamp));
    header.append(XML::Element("period",hdr.period));
    header.append(XML::Element("fch1",hdr.fch1));
    header.append(XML::Element("foff",hdr.foff));
    header.append(XML::Element("nchans",hdr.nchans));
    header.append(XML::Element("telescope_id",hdr.telescope_id));
    header.append(XML::Element("machine_id",hdr.machine_id));
    header.append(XML::Element("data_type",hdr.data_type));
    header.append(XML::Element("ibeam",hdr.ibeam));
    header.append(XML::Element("nbeams",hdr.nbeams));
    header.append(XML::Element("nbits",hdr.nbits));
    header.append(XML::Element("barycentric",hdr.barycentric));
    header.append(XML::Element("pulsarcentric",hdr.pulsarcentric));
    header.append(XML::Element("nbins",hdr.nbins));
    header.append(XML::Element("nsamples",hdr.nsamples));
    header.append(XML::Element("nifs",hdr.nifs));
    header.append(XML::Element("npuls",hdr.npuls));
    header.append(XML::Element("refdm",hdr.refdm));
    header.append(XML::Element("signed",(int)hdr.signed_data));
    root.append(header);
  }

  void add_search_parameters(CmdLineOptions& args){
    XML::Element search_options("search_parameters");
    search_options.append(XML::Element("infilename",args.infilename));
    search_options.append(XML::Element("outdir",args.outdir));
    search_options.append(XML::Element("killfilename",args.killfilename));
    search_options.append(XML::Element("zapfilename",args.zapfilename));
    search_options.append(XML::Element("max_num_threads",args.max_num_threads));
    search_options.append(XML::Element("size",args.size));
    search_options.append(XML::Element("dm_start",args.dm_start));
    search_options.append(XML::Element("dm_end",args.dm_end));
    search_options.append(XML::Element("dm_tol",args.dm_tol));
    search_options.append(XML::Element("dm_pulse_width",args.dm_pulse_width));
    search_options.append(XML::Element("acc_start",args.acc_start));
    search_options.append(XML::Element("acc_end",args.acc_end));
    search_options.append(XML::Element("acc_tol",args.acc_tol));
    search_options.append(XML::Element("acc_pulse_width",args.acc_pulse_width));
    search_options.append(XML::Element("boundary_5_freq",args.boundary_5_freq));
    search_options.append(XML::Element("boundary_25_freq",args.boundary_25_freq));
    search_options.append(XML::Element("nharmonics",args.nharmonics));
    search_options.append(XML::Element("npdmp",args.npdmp));
    search_options.append(XML::Element("min_snr",args.min_snr));
    search_options.append(XML::Element("min_freq",args.min_freq));
    search_options.append(XML::Element("max_freq",args.max_freq));
    search_options.append(XML::Element("max_harm",args.max_harm));
    search_options.append(XML::Element("freq_tol",args.freq_tol));
    search_options.append(XML::Element("verbose",args.verbose));
    search_options.append(XML::Element("progress_bar",args.progress_bar));
    root.append(search_options);
  }

  void add_misc_info(void){
    XML::Element info("misc_info");
    char buf[128];
    getlogin_r(buf,128);
    info.append(XML::Element("username",buf));
    std::time_t t = std::time(NULL);
    std::strftime(buf, 128, "%Y-%m-%d-%H:%M", std::localtime(&t));
    info.append(XML::Element("local_datetime",buf));
    std::strftime(buf, 128, "%Y-%m-%d-%H:%M", std::gmtime(&t));
    info.append(XML::Element("utc_datetime",buf));
    root.append(info);
  }
  
  void add_timing_info(std::map<std::string,Stopwatch>& elapsed_times){
    XML::Element times("execution_times");
    typedef std::map<std::string,Stopwatch>::iterator it_type;
    for (it_type it=elapsed_times.begin(); it!=elapsed_times.end(); it++)
      times.append(XML::Element(it->first,it->second.getTime()));
    root.append(times);
  }
  
  void add_gpu_info(std::vector<int>& device_idxs){
    XML::Element gpu_info("cuda_device_parameters");
    int runtime_version,driver_version;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);
    gpu_info.append(XML::Element("runtime",runtime_version));
    gpu_info.append(XML::Element("driver",driver_version));
    cudaDeviceProp properties;
    for (int ii=0;ii<device_idxs.size();ii++){
      XML::Element device("cuda_device");
      device.add_attribute("id",device_idxs[ii]);
      cudaGetDeviceProperties(&properties,device_idxs[ii]);
      device.append(XML::Element("name",properties.name));
      device.append(XML::Element("major_cc",properties.major));
      device.append(XML::Element("minor_cc",properties.minor));
      gpu_info.append(device);
    }
    root.append(gpu_info);
  }
  
  void add_dm_list(std::vector<float>& dms){
    XML::Element dm_trials("dedispersion_trials");
    dm_trials.add_attribute("count",dms.size());
    for (int ii=0;ii<dms.size();ii++){
      XML::Element trial("trial");
      trial.add_attribute("id",ii);
      trial.set_text(dms[ii]);
      dm_trials.append(trial);
    }
    root.append(dm_trials);
  }
  
  void add_acc_list(std::vector<float>& accs){
    XML::Element acc_trials("acceleration_trials");
    acc_trials.add_attribute("count",accs.size());
    acc_trials.add_attribute("DM",0);
    for(int ii=0;ii<accs.size();ii++){
      XML::Element trial("trial");
      trial.add_attribute("id",ii);
      trial.set_text(accs[ii]);
      acc_trials.append(trial);
    }
    root.append(acc_trials);
  }

  void add_candidates(std::vector<Candidate>& candidates, 
		      std::map<unsigned,long int> byte_map)
  {
    XML::Element cands("candidates");
    for (int ii=0;ii<candidates.size();ii++){
      XML::Element cand("candidate");
      cand.add_attribute("id",ii);
      cand.append(XML::Element("period",1.0/candidates[ii].freq));
      cand.append(XML::Element("opt_period",candidates[ii].opt_period));
      cand.append(XML::Element("dm",candidates[ii].dm));
      cand.append(XML::Element("acc",candidates[ii].acc));
      cand.append(XML::Element("nh",candidates[ii].nh));
      cand.append(XML::Element("snr",candidates[ii].snr));
      cand.append(XML::Element("folded_snr",candidates[ii].folded_snr));
      cand.append(XML::Element("is_adjacent",candidates[ii].is_adjacent));
      cand.append(XML::Element("is_physical",candidates[ii].is_physical));
      cand.append(XML::Element("ddm_count_ratio",candidates[ii].ddm_count_ratio));
      cand.append(XML::Element("ddm_snr_ratio",candidates[ii].ddm_snr_ratio));
      cand.append(XML::Element("nassoc",candidates[ii].count_assoc()));
      cand.append(XML::Element("byte_offset",byte_map[ii]));
      cands.append(cand);
    }
    root.append(cands);
  }

  void add_candidates(std::vector<Candidate>& candidates,
		      std::map<int,std::string>& filenames){
    XML::Element cands("candidates");
    for (int ii=0;ii<candidates.size();ii++){
      XML::Element cand("candidate");
      cand.add_attribute("id",ii);
      cand.append(XML::Element("period",1.0/candidates[ii].freq));
      cand.append(XML::Element("opt_period",candidates[ii].opt_period));
      cand.append(XML::Element("dm",candidates[ii].dm));
      cand.append(XML::Element("acc",candidates[ii].acc));
      cand.append(XML::Element("nh",candidates[ii].nh));
      cand.append(XML::Element("snr",candidates[ii].snr));
      cand.append(XML::Element("folded_snr",candidates[ii].folded_snr));
      cand.append(XML::Element("is_adjacent",candidates[ii].is_adjacent));
      cand.append(XML::Element("is_physical",candidates[ii].is_physical));
      cand.append(XML::Element("ddm_count_ratio",candidates[ii].ddm_count_ratio));
      cand.append(XML::Element("ddm_snr_ratio",candidates[ii].ddm_snr_ratio));
      cand.append(XML::Element("nassoc",candidates[ii].count_assoc()));
      cand.append(XML::Element("results_file",filenames[ii]));
      cands.append(cand);
    }    
    root.append(cands);
  }
 
};


class CandidateFileWriter {
public:
  std::map<int,std::string> filenames;
  std::map<unsigned,long int> byte_mapping;
  std::string output_dir;
 
  CandidateFileWriter(std::string output_directory)
    :output_dir(output_directory)
  {
    struct stat st = {0};
    if (stat(output_dir.c_str(), &st) == -1) {
      if (mkdir(output_dir.c_str(), 0777) != 0)
	perror(output_dir.c_str());	
    }
  }

  void write_binary(std::vector<Candidate>& candidates,
		    std::string filename)
  {
    char actualpath [PATH_MAX];
    std::stringstream filepath;
    filepath << output_dir << "/" << filename;
    realpath(filepath.str().c_str(), actualpath);
    
    FILE* fo = fopen(actualpath,"w");
    if (fo == NULL) {
      perror(filepath.str().c_str());
      return;
    }
    
    for (int ii=0;ii<candidates.size();ii++)
      {
	byte_mapping[ii] = ftell(fo);
	if (candidates[ii].fold.size()>0)
	  {
	    size_t size = candidates[ii].nbins * candidates[ii].nints;
	    float* fold = &candidates[ii].fold[0];
	    fprintf(fo,"FOLD");
	    fwrite(&candidates[ii].nbins,sizeof(int),1,fo);
	    fwrite(&candidates[ii].nints,sizeof(int),1,fo);
	    fwrite(fold,sizeof(float),size,fo);
	  }
	std::vector<CandidatePOD> detections;
	candidates[ii].collect_candidates(detections);
	int ndets = detections.size();
	fwrite(&ndets,sizeof(int),1,fo);
	fwrite(&detections[0],sizeof(CandidatePOD),ndets,fo);
      }
    fclose(fo);
  }
  
  void write_binaries(std::vector<Candidate>& candidates)
  {
    char actualpath [PATH_MAX];
    char filename[1024];
    std::stringstream filepath;
    for (int ii=0;ii<candidates.size();ii++){
      filepath.str("");
      sprintf(filename,"cand_%04d_%.5f_%.1f_%.1f.peasoup",
              ii,1.0/candidates[ii].freq,candidates[ii].dm,candidates[ii].acc);
      filepath << output_dir << "/" << filename;

      char* ptr = realpath(filepath.str().c_str(), actualpath);
      filenames[ii] = std::string(actualpath);
      
      FILE* fo = fopen(filepath.str().c_str(),"w");
      if (fo == NULL) {
	perror(filepath.str().c_str());
	return;
      }
      
      if (candidates[ii].fold.size()>0){
	size_t size = candidates[ii].nbins * candidates[ii].nints;
	float* fold = &candidates[ii].fold[0];
	fprintf(fo,"FOLD");
	fwrite(&candidates[ii].nbins,sizeof(int),1,fo);
	fwrite(&candidates[ii].nints,sizeof(int),1,fo);
	fwrite(fold,sizeof(float),size,fo);
      }
      std::vector<CandidatePOD> detections;
      candidates[ii].collect_candidates(detections);
      int ndets = detections.size();
      fwrite(&ndets,sizeof(int),1,fo);
      fwrite(&detections[0],sizeof(CandidatePOD),ndets,fo);
      fclose(fo);
    }
  }
};

