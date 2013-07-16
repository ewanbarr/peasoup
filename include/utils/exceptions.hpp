#pragma once
#include <fstream>
#include <stdexcept>
#include <string>
#include <sstream>
#include "dedisp.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"

class ErrorChecker {
public:
  static void check_dedisp_error(dedisp_error error,
			    std::string function_name)
  {
    if (error != DEDISP_NO_ERROR) {
      std::stringstream error_msg;
      error_msg << function_name 
		<< " failed with DEDISP error: "
		<< dedisp_get_error_string(error) 
		<< std::endl;
      throw std::runtime_error(error_msg.str());
    }
  }
  
  static void check_file_error(std::ifstream& infile, std::string filename){
    if(!infile.good()) {
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
  }

  static void check_cuda_error(cudaError_t error){
    if (error!=cudaSuccess){
      std::stringstream error_msg;
      error_msg << "CUDA failed with error: "
		<< cudaGetErrorString(error) << std::endl;
      throw std::runtime_error(error_msg.str());
    }
  }
  
  static void check_cufft_error(cufftResult error){
    if (error!=CUFFT_SUCCESS){
      std::stringstream error_msg;
      error_msg << "cuFFT failed with error: ";
      switch (error)
	{
	case CUFFT_INVALID_PLAN:
	  error_msg <<  "CUFFT_INVALID_PLAN";
	  break;

	case CUFFT_ALLOC_FAILED:
	  error_msg <<  "CUFFT_ALLOC_FAILED";
	  break;

	case CUFFT_INVALID_TYPE:
	  error_msg <<  "CUFFT_INVALID_TYPE";
	  break;

	case CUFFT_INVALID_VALUE:
	  error_msg <<  "CUFFT_INVALID_VALUE";
	  break;

	case CUFFT_INTERNAL_ERROR:
	  error_msg <<  "CUFFT_INTERNAL_ERROR";
	  break;

	case CUFFT_EXEC_FAILED:
	  error_msg <<  "CUFFT_EXEC_FAILED";
	  break;
	  
	case CUFFT_SETUP_FAILED:
	  error_msg <<  "CUFFT_SETUP_FAILED";
	  break;

	case CUFFT_INVALID_SIZE:
	  error_msg <<  "CUFFT_INVALID_SIZE";
	  break;

	case CUFFT_UNALIGNED_DATA:
	  error_msg <<  "CUFFT_UNALIGNED_DATA";
	  break;

	default:
	  error_msg <<  "<unknown>";
	}
      error_msg << std::endl;
      throw std::runtime_error(error_msg.str());
    }
  }
};
