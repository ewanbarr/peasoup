/*
  Copyright 2014 Ewan Barr

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

/*
  header.hpp
  
  By Ben Barsdell (2011)
  benbarsdell@gmail.com

  Modified by Ewan Barr (2013)
  ewan.d.barr@gmail.com

  This file contains classes and functions for the reading
  and writing of various data file header formats. Currently
  implemented header formats are:
  
  sigproc - used for peasoup filterbank input mode 
  psrdada - currently unused

*/

#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdlib.h>

#define DADA_HDR_SIZE 4096L

/*!
  \brief Psrdada header reader/storage class.

  This class implements methods for reading 
  of psrdada data file headers. The header 
  values are stored as attributes.
*/
class DadaHeader {
private:
  /*!                                                   
    \brief Extract a header value by key.               
                                                        
    Method to extract a header value given the psrdada 
    header keyword string. Method returns a null string
    if not instance of the keyword is found.            
                                                        
    \param name the name of a psrdada header keyword.
    \param header a stringstream containing a psrdada header.
    \return psrdada header value.
  */
  std::string get_value(std::string name,std::stringstream& header){
    size_t position = header.str().find(name);
    if (position!=std::string::npos){
      header.seekg(position+name.length());
      std::string value;
      header >> value;
      return value;
    } else {
      return "";
    }
  }

public:
  float header_version;
  uint header_size;
  double bw;
  double freq;
  uint nant;
  uint nchan;
  uint ndim;
  uint npol;
  uint nbit;
  double tsamp;
  double osamp_ratio;
  std::string source_name;
  std::string ra;
  std::string dec;
  std::string proc_file;
  std::string mode;
  std::string observer;
  std::string pid;
  size_t obs_offset;
  std::string telescope;
  std::string instrument;
  size_t dsb;
  size_t filesize;
  size_t dada_filesize;
  size_t nsamples;
  size_t bytes_per_sec;
  std::string utc_start;
  uint ant_id;
  uint file_no;

  /*!
    \brief Default constructor.
  */
  DadaHeader(){};

  /*!
    \brief Read a psrdada header from file.
    
    Method to read a psrdada header from a .dada format
    file. Method attempts to read all standard dada header
    keywords. Values are read in to class attributes.

    \param filename filename to read header from.
  */
  
  void fromfile(std::string filename){
    std::ifstream infile(filename.c_str());
    std::vector<char> buf(DADA_HDR_SIZE);
    infile.read(&buf[0],DADA_HDR_SIZE);
    std::stringstream header;
    header.rdbuf()->pubsetbuf(&buf[0],DADA_HDR_SIZE);
    infile.seekg(0,infile.end);
    filesize       = (size_t) infile.tellg() - (size_t) DADA_HDR_SIZE;
    header_version = atof(get_value("HDR_VERSION ",header).c_str());
    header_size    = atoi(get_value("HDR_SIZE ",header).c_str());
    bw             = atoi(get_value("BW ",header).c_str());
    freq           = atof(get_value("FREQ ",header).c_str());
    nant           = atoi(get_value("NANT ",header).c_str());
    nchan          = atoi(get_value("NCHAN ",header).c_str());
    ndim           = atoi(get_value("NDIM ",header).c_str());
    npol           = atoi(get_value("NPOL ",header).c_str());
    nbit           = atoi(get_value("NBIT ",header).c_str());
    tsamp          = atof(get_value("TSAMP ",header).c_str());
    osamp_ratio    = atof(get_value("OSAMP_RATIO ",header).c_str());
    source_name    = get_value("SOURCE ",header);
    ra             = get_value("RA ",header);
    dec            = get_value("DEC ",header);
    proc_file      = get_value("PROC_FILE ",header);
    mode           = get_value("MODE ",header);
    observer       = get_value("OBSERVER ",header);
    pid            = get_value("PID ",header);
    obs_offset     = atoi(get_value("OBS_OFFSET ",header).c_str());
    telescope      = get_value("TELESCOPE ",header);
    instrument     = get_value("INSTRUMENT ",header);
    dsb            = atoi(get_value("DSB ",header).c_str());
    dada_filesize  = atoi(get_value("FILE_SIZE ",header).c_str());
    nsamples       = filesize/nchan/nant/npol/2.;
    bytes_per_sec  = atoi(get_value("BYTES_PER_SECOND ",header).c_str());
    utc_start      = get_value("UTC_START ",header);
    ant_id         = atoi(get_value("ANT_ID ",header).c_str());
    file_no        = atoi(get_value("FILE_NUMBER ",header).c_str());
    infile.close();
  }
};  

/*!
  \brief Class for reading and writing of sigproc format headers.

  This class emulates the functionality of sigproc's
  header reading and writing functions.

  \note This class is implemented to emulate existing C code. 
*/
class SigprocHeader {
public:
  // Naming convention from sigproc header keys
  std::string source_name; /*!< Source name.*/
  std::string rawdatafile; /*!< Name of original data file.*/
  double az_start; /*!< Azimuth angle (deg).*/
  double za_start; /*!< Zenith angle (deg).*/
  double src_raj; /*!< Right ascension (hhmmss.ss format).*/
  double src_dej; /*!< Declination (ddmmss.ss format).*/
  double tstart; /*!< Modified Julian date of first sample.*/
  double tsamp; /*!< Sampling time (seconds).*/
  double period; 
  double fch1; /*!< Frequency of top channel (MHz).*/
  double foff; /*!< Channel bandwith (MHz).*/
  int    nchans; /*!< Number of frequency channels.*/
  int    telescope_id; /*!< Sigproc telescope ID.*/
  int    machine_id; /*!< Sigproc backend ID.*/
  int    data_type; /*!< Sigproc data type ID.*/ 
  int    ibeam; /*!< Beam number.*/
  int    nbeams; /*!< Number of beams.*/
  int    nbits; /*!< Number of bits per sample.*/
  int    barycentric; 
  int    pulsarcentric;
  int    nbins;  
  int    nsamples; /*!< Number of time samples.*/
  int    nifs; 
  int    npuls;
  double refdm; /*!< Reference DM of data.*/
  unsigned char signed_data; /*!< char or unsigned char.*/
  unsigned int size; /*!< Header size in bytes.*/
  
  /*!
    \brief Default constructor, initializes all values to zero.
  */
  SigprocHeader() : 
    az_start(0.0), za_start(0.0), src_raj(0.0), src_dej(0.0),
    tstart(0.0), tsamp(0.0), period(0.0), fch1(0.0), foff(0.0),
    nchans(0), telescope_id(0), machine_id(0), data_type(0),
    ibeam(0), nbeams(0), nbits(0), barycentric(0),
    pulsarcentric(0), nbins(0), nsamples(0), nifs(0), npuls(0),
    refdm(0.0), signed_data(0), size(0) {}
};

/*!
  \brief Write a string in sigproc format.
  
  Write a string to a binary stream in sigrpoc header format.
  
  \param stream a binary stream to which data is written.
  \param str the string to be written.
*/
template<class String, class BinaryStream>
void header_write(BinaryStream& stream, const String& str) {
  std::string s = str;
  int len = s.size();
  // TODO: Apply byte swapping for endian-correctness
  stream.write((char*)&len, sizeof(int));
  // TODO: Apply byte swapping for endian-correctness
  stream.write(s.c_str(), len*sizeof(char));
}

/*!
  \brief Write an int in sigproc format.

  Write an int to a binary stream in sigproc header format.

  \param stream a binary stream to which data is written.
  \param name keyword name for value being written.
  \param val integer value to be written.
*/
template<class String, class BinaryStream>
void header_write(BinaryStream& stream, String name, int val) {
  header_write(stream, name);
  // TODO: Apply byte swapping for endian-correctness
  stream.write((char*)&val, sizeof(int));
}

// WAR ambiguous conversion to int or double
template<class String, class BinaryStream>
void header_write(BinaryStream& stream, String name, unsigned val) {
  header_write(stream, name, (int)val);
}

/*!
  \brief Write a float in sigproc format.

  Write a float to a binary stream in sigproc header format.

  \param stream a binary stream to which data is written.
  \param name keyword name for value being written.
  \param val floating-point value to be written.
*/
template<class String, class BinaryStream>
void header_write(BinaryStream& stream, String name, double val) {
  header_write(stream, name);
  // TODO: Apply byte swapping for endian-correctness
  stream.write((char*)&val, sizeof(double));
}

/*!
  \brief Write coordinates in sigproc format.

  Write an equatorial and horizontal coordinates
  to file in sigproc header format. Coordinates are
  represented as doubles such that:

  12:34:56.78 is represented by 123456.78

  \param stream a binary stream to which data is written.
  \param raj R.A. in floating-point format.
  \param dej Dec. in floating-point format.
  \param az Azimuthal angle in floating-point format.
  \param za Zenith angle in floating-point format.
*/
template<class BinaryStream>
void header_write(BinaryStream& stream,
		  double raj, double dej,
		  double az, double za) {
  header_write(stream, "src_raj",  raj);
  header_write(stream, "src_dej",  dej);
  header_write(stream, "az_start", az);
  header_write(stream, "za_start", za);
}

/*!
  \brief Write a single byte in sigproc format.
  
  Write a single char to file in sigproc header format.

  \param stream a binary stream to which data is written.
  \param name keyword name for value being written.
  \param val unsigned byte value to be written.
*/
template<class String, class BinaryStream>
void header_write(BinaryStream& stream, String name, unsigned char val) {
  header_write(stream, name);
  stream.write((char*)&val, sizeof(unsigned char));
}

// Internal functions
namespace detail {
  
// Note: This is an internal function and is not intended to be used externally
template<class BinaryStream, class String>
bool header_read(BinaryStream& stream,
		 String& str) {
  int len;
  char c_str[80];
  stream.read((char*)&len, sizeof(int));
  if( len <= 0 || len >= 80 ) return false;
  stream.read(c_str, len*sizeof(char));
  c_str[len] = '\0';
  str = c_str;
  return true;
}
  
} // namespace detail

/*! 
  \brief Read header data into a SigprocHeader (or similar) structure.
  
  Function attempts to read all standard sigproc header keywords. 
  Only header attributes with matching keywords are updated in the
  given Header object.

  \param stream Binary stream to read header from.
  \param header A SigprocHeader-like storage class to hold header values.
*/
template<class BinaryStream, class Header>
bool read_header(BinaryStream& stream, Header& header) {
	
  std::string s;
  bool ret = detail::header_read(stream, s);
  if( !ret || s != "HEADER_START" ) {
    stream.seekg(0, std::ios::beg);
    return false;
  }
  
  bool expecting_source_name = false;
  bool expecting_rawdatafile = false;
  while( true ) {
    ret = detail::header_read(stream, s);
    
    if( s == "HEADER_END" ) break;
    else if( s == "source_name" )   expecting_source_name = true;
    else if( s == "rawdatafile")    expecting_rawdatafile = true;
    else if( s == "az_start" )      stream.read((char*)(char*)&header.az_start, sizeof(double));
    else if( s == "za_start" )      stream.read((char*)&header.za_start, sizeof(double));
    else if( s == "src_raj" )       stream.read((char*)&header.src_raj, sizeof(double));
    else if( s == "src_dej" )       stream.read((char*)&header.src_dej, sizeof(double));
    else if( s == "tstart" )        stream.read((char*)&header.tstart, sizeof(double));
    else if( s == "tsamp" )         stream.read((char*)&header.tsamp, sizeof(double));
    else if( s == "period" )        stream.read((char*)&header.period, sizeof(double));
    else if( s == "fch1" )          stream.read((char*)&header.fch1, sizeof(double));
    else if( s == "foff" )          stream.read((char*)&header.foff, sizeof(double));
    else if( s == "nchans" )        stream.read((char*)&header.nchans, sizeof(int));
    else if( s == "telescope_id" )  stream.read((char*)&header.telescope_id, sizeof(int));
    else if( s == "machine_id" )    stream.read((char*)&header.machine_id, sizeof(int));
    else if( s == "data_type" )     stream.read((char*)&header.data_type, sizeof(int));
    else if( s == "ibeam" )         stream.read((char*)&header.ibeam, sizeof(int));
    else if( s == "nbeams" )        stream.read((char*)&header.nbeams, sizeof(int));
    else if( s == "nbits" )         stream.read((char*)&header.nbits, sizeof(int));
    else if( s == "barycentric" )   stream.read((char*)&header.barycentric, sizeof(int));
    else if( s == "pulsarcentric" ) stream.read((char*)&header.pulsarcentric, sizeof(int));
    else if( s == "nbins" )         stream.read((char*)&header.nbins, sizeof(int));
    else if( s == "nsamples" )      stream.read((char*)&header.nsamples, sizeof(int));
    else if( s == "nifs" )          stream.read((char*)&header.nifs, sizeof(int));
    else if( s == "npuls" )         stream.read((char*)&header.npuls, sizeof(int));
    else if( s == "refdm" )         stream.read((char*)&header.refdm, sizeof(double));
    else if( s == "signed" )        stream.read((char*)&header.signed_data, sizeof(unsigned char));
    else if( expecting_source_name ) {
      header.source_name = s;
      expecting_source_name = false;
    }
    else if( expecting_rawdatafile ) {
      header.rawdatafile = s;
      expecting_rawdatafile = false;
    }
    else {
      std::cerr << "Warning: read_header: unknown parameter " << s << std::endl;
    }
  }
  header.size = stream.tellg();
  if( 0 == header.nsamples ) {
    // Compute the number of samples from the file size
    stream.seekg(0, std::ios::end);
    size_t total_size = stream.tellg();
    header.nsamples = (total_size-header.size) / header.nchans * 8 / header.nbits;
    // Seek back to the end of the header
    stream.seekg(header.size, std::ios::beg);
  }
  return NULL;
}
