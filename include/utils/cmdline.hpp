#pragma once
#include <tclap/CmdLine.h>
#include <string>
#include <iostream>

struct CmdLineOptions {
  std::string infilename;
  std::string outdir;
  std::string killfilename;
  std::string zapfilename;
  int max_num_threads;
  unsigned int size;
  float dm_start;
  float dm_end;
  float dm_tol;
  float dm_pulse_width;
  float acc_start;
  float acc_end;
  float acc_tol;
  float acc_pulse_width;
  float boundary_5_freq;
  float boundary_25_freq;
  int nharmonics;
  int npdmp;
  int limit;
  float min_snr;
  float min_freq;
  float max_freq;
  int max_harm;
  float freq_tol;
  bool verbose;
  bool progress_bar;
};

bool read_cmdline_options(CmdLineOptions& args, int argc, char **argv)
{
  try
    {
      TCLAP::CmdLine cmd("Peasoup - a GPU pulsar search pipeline", ' ', "1.0");

      TCLAP::ValueArg<std::string> arg_infilename("i", "inputfile",
						  "File to process (.fil)",
                                                  true, "", "string", cmd);

      TCLAP::ValueArg<std::string> arg_outdir("o", "outdir",
						   "The output directory",
						   false, "./candidates/", "string",cmd);

      TCLAP::ValueArg<std::string> arg_killfilename("k", "killfile",
						    "Channel mask file",
						    false, "", "string",cmd);

      TCLAP::ValueArg<std::string> arg_zapfilename("z", "zapfile",
                                                   "Birdie list file",
                                                   false, "", "string",cmd);

      TCLAP::ValueArg<int> arg_max_num_threads("t", "num_threads",
                                               "The number of GPUs to use",
                                               false, 14, "int", cmd);

      TCLAP::ValueArg<int> arg_limit("", "limit",
				     "upper limit on number of candidates to write out",
				     false, 1000, "int", cmd);

      TCLAP::ValueArg<size_t> arg_size("", "fft_size",
                                       "Transform size to use (defaults to lower power of two)",
                                       false, 0, "size_t", cmd);

      TCLAP::ValueArg<float> arg_dm_start("", "dm_start",
                                          "First DM to dedisperse to",
                                          false, 0.0, "float", cmd);

      TCLAP::ValueArg<float> arg_dm_end("", "dm_end",
                                        "Last DM to dedisperse to",
                                        false, 100.0, "float", cmd);

      TCLAP::ValueArg<float> arg_dm_tol("", "dm_tol",
                                        "DM smearing tolerance (1.11=10%)",
                                        false, 1.10, "float",cmd);

      TCLAP::ValueArg<float> arg_dm_pulse_width("", "dm_pulse_width",
                                                "Minimum pulse width for which dm_tol is valid",
                                                false, 64.0, "float (us)",cmd);

      TCLAP::ValueArg<float> arg_acc_start("", "acc_start",
					   "First acceleration to resample to",
					   false, 0.0, "float", cmd);

      TCLAP::ValueArg<float> arg_acc_end("", "acc_end",
					 "Last acceleration to resample to",
					 false, 0.0, "float", cmd);

      TCLAP::ValueArg<float> arg_acc_tol("", "acc_tol",
					 "Acceleration smearing tolerance (1.11=10%)",
					 false, 1.10, "float",cmd);

      TCLAP::ValueArg<float> arg_acc_pulse_width("", "acc_pulse_width",
                                                 "Minimum pulse width for which acc_tol is valid",
						 false, 64.0, "float (us)",cmd);

      TCLAP::ValueArg<float> arg_boundary_5_freq("", "boundary_5_freq",
                                                 "Frequency at which to switch from median5 to median25",
                                                 false, 0.05, "float", cmd);

      TCLAP::ValueArg<float> arg_boundary_25_freq("", "boundary_25_freq",
						  "Frequency at which to switch from median25 to median125",
						  false, 0.5, "float", cmd);

      TCLAP::ValueArg<int> arg_nharmonics("n", "nharmonics",
                                          "Number of harmonic sums to perform",
                                          false, 4, "int", cmd);

      TCLAP::ValueArg<int> arg_npdmp("", "npdmp",
                                     "Number of candidates to fold and pdmp",
                                     false, 0, "int", cmd);

      TCLAP::ValueArg<float> arg_min_snr("m", "min_snr",
                                         "The minimum S/N for a candidate",
                                         false, 9.0, "float",cmd);

      TCLAP::ValueArg<float> arg_min_freq("", "min_freq",
                                          "Lowest Fourier freqency to consider",
                                          false, 0.1, "float",cmd);

      TCLAP::ValueArg<float> arg_max_freq("", "max_freq",
                                          "Highest Fourier freqency to consider",
                                          false, 1100.0, "float",cmd);

      TCLAP::ValueArg<int> arg_max_harm("", "max_harm_match",
                                        "Maximum harmonic for related candidates",
                                        false, 16, "float",cmd);

      TCLAP::ValueArg<float> arg_freq_tol("", "freq_tol",
                                          "Tolerance for distilling frequencies (0.0001 = 0.01%)",
                                          false, 0.0001, "float",cmd);

      TCLAP::SwitchArg arg_verbose("v", "verbose", "verbose mode", cmd);

      TCLAP::SwitchArg arg_progress_bar("p", "progress_bar", "Enable progress bar for DM search", cmd);

      cmd.parse(argc, argv);
      args.infilename        = arg_infilename.getValue();
      args.outdir            = arg_outdir.getValue();
      args.killfilename      = arg_killfilename.getValue();
      args.zapfilename       = arg_zapfilename.getValue();
      args.max_num_threads   = arg_max_num_threads.getValue();
      args.limit             = arg_limit.getValue();
      args.size              = arg_size.getValue();
      args.dm_start          = arg_dm_start.getValue();
      args.dm_end            = arg_dm_end.getValue();
      args.dm_tol            = arg_dm_tol.getValue();
      args.dm_pulse_width    = arg_dm_pulse_width.getValue();
      args.acc_start         = arg_acc_start.getValue();
      args.acc_end           = arg_acc_end.getValue();
      args.acc_tol           = arg_acc_tol.getValue();
      args.acc_pulse_width   = arg_acc_pulse_width.getValue();
      args.boundary_5_freq   = arg_boundary_5_freq.getValue();
      args.boundary_25_freq  = arg_boundary_25_freq.getValue();
      args.nharmonics        = arg_nharmonics.getValue();
      args.npdmp             = arg_npdmp.getValue();
      args.min_snr           = arg_min_snr.getValue();
      args.min_freq          = arg_min_freq.getValue();
      args.max_freq          = arg_max_freq.getValue();
      args.max_harm          = arg_max_harm.getValue();
      args.freq_tol          = arg_freq_tol.getValue();
      args.verbose           = arg_verbose.getValue();
      args.progress_bar      = arg_progress_bar.getValue();

    }catch (TCLAP::ArgException &e) {
    std::cerr << "Error: " << e.error() << " for arg " << e.argId()
              << std::endl;
    return false;
  }
  return true;
}
