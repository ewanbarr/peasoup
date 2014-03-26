#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctime>
#include <iostream>
#include <map>
#include <utils/xml_util.hpp>
#include <utils/cmdline.hpp>
#include <data_types/header.hpp>

class OutputFileWriter {
  XML::Element root;


public:
  OutputFileWriter()
    :root("peasoup_search"){}
  
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
    header.append(XML::Element("signed",hdr.signed));
    root.append(header)
  }

  
  void add_search_parameters(CmdLineOptions& args){
    XML::Element search_options("search_parameters");
    search_options.append(XML::Element("infilename",args.infilename));
    search_options.append(XML::Element("outfilename",args.outfilename));
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
    //hardware_parameters

  void add_misc_info(void){
    XML::Element info("misc_info");
    char buf[128];
    //Removed this call as it was generating invalid characters in XML output
    //getlogin_r(buf,128);
    //info.append(XML::Element("username",buf));
    std::time_t t = std::time(NULL);
    std::strftime(buf, 128, "%Y-%m-%d-%H:%M", std::localtime(&t));
    info.append(XML::Element("local_datetime",buf));
    std::strftime(buf, 128, "%Y-%m-%d-%H:%M", std::gmtime(&t));
    info.append(XML::Element("utc_datetime",buf));
    root.append(info)
  }
  
  void add_timing_info(std::map<std::string,float> elapsed_times){
    XML::Element times("execution_times");
    typedef std::map<std::string,float>::iterator it_type;
    for (it_type it=elapsed_times.begin(); it!=elapsed_times.end(); it++)
      times.append(XML::Element(it->first,it->second));
    root.append(times);
  }

  void add_hardware_info(void){
    
  }

  void add_candidates(std::vector<Candidate>& candidates){
    XML::Element cands("candidates");
    for (int ii=0;ii<candidates.size();ii++){
      XML::Element cand("candidate");
      cand.add_attribute("id",ii);
      cand.append(XML::Element("period",candidates[ii].period))
      
    }
    
  }

  
}




