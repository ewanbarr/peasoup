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
  filterbank.hpp

  By Ewan Barr (2013)
  ewan.d.barr@gmail.com

  This file contians classes and methods for the reading, storage
  and manipulation of filterbank format data. Filterbank format
  can be any time-frequency data block. Time must be the slowest
  changing dimension.
*/

#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "data_types/header.hpp"
#include "utils/exceptions.hpp"

/*!
  \brief Base class for handling filterbank data.

  All time and frequency resolved data types should inherit
  from this class. Class presents virtual set and get methods
  for various requrired meta data. The filterbank data itself
  is stred in the *data pointer as unsigend chars.
*/
class Filterbank {
protected:
  //Filterbank metadata
  unsigned char* data; /*!< Pointer to filterbank data.*/
  unsigned int nsamps; /*!< Number of time samples. */
  unsigned int nchans; /*!< Number of frequecy channels. */
  unsigned char nbits; /*!< Bits per time sample. */
  float fch1; /*!< Frequency of top channel (MHz) */
  float foff; /*!< Channel bandwidth (MHz) */
  float tsamp; /*!< Sampling time (seconds) */

  /*!
    \brief Instantiate a new Filterbank object with metadata.

    Instantiate a new Filterbank object from an existing data
    pointer and metadata.

    \param data_ptr A pointer to a memory location containing filterbank data.
    \param nsamps The number of time samples in the data.
    \param nchans The number of frequency channels in that data.
    \param nbins The size of a single data point in bits.
    \param fch1 The centre frequency of the first data channel.
    \param foff The bandwidth of a frequency channel.
    \param tsamp The sampling time of the data.
  */
  Filterbank(unsigned char* data_ptr, unsigned int nsamps,
	     unsigned int nchans, unsigned char nbits,
	     float fch1, float foff, float tsamp)
    :data(data_ptr),nsamps(nsamps),nchans(nchans),
     nbits(nbits),fch1(fch1),foff(foff),tsamp(tsamp){}

  /*!
    \brief Instantiate a new default Filterbank object.

    Create a new Filterbank object with the data pointer and
    all metadata set to zero.
  */
  Filterbank(void)
    :data(0),nsamps(0),nchans(0),
     nbits(0),fch1(0.0),foff(0.0),tsamp(0.0){}

public:

  /*!
    \brief Get the currently set sampling time.

    \return The currently set sampling time.
  */
  virtual float get_tsamp(void){return tsamp;}

  /*!
    \brief Set the sampling time.

    \param tsamp The sampling time of the data (in seconds).
  */
  virtual void set_tsamp(float tsamp){this->tsamp = tsamp;}

  /*!
    \brief Get the currently set channel bandwidth.

    \return The channel bandwidth (in MHz).
  */
  virtual float get_foff(void){return foff;}

  /*!
    \brief Set the channel bandwidth.

    \param foff The channel bandwidth (in MHz).
  */
  virtual void set_foff(float foff){this->foff = foff;}

  /*!
  \brief Get the frequency of the top channel.

  \return The frequency of channel 0 (in MHz)
  */
  virtual float get_fch1(void){return fch1;}

  /*!
    \brief Set the frequency of the top channel.

    \param fch1 The frequency of channel 0 (in MHz).
  */
  virtual void set_fch1(float fch1){this->fch1 = fch1;}

  /*!
    \brief Get the number of frequency channels.

    \return The number of frequency channels.
  */
  virtual float get_nchans(void){return nchans;}

  /*!
    \brief Set the number of frequency channels.

    \param nchans The number of frequency channels in the data.
  */
  virtual void set_nchans(unsigned int nchans){this->nchans = nchans;}

  /*!
    \brief Get the number of time samples in the data.

    \return The number of time samples.
  */
  virtual unsigned int get_nsamps(void){return nsamps;}

  /*!
    \brief Set the number of time samples in data.

    \param nsamps The number of time samples.
  */
  virtual void set_nsamps(unsigned int nsamps){this->nsamps = nsamps;}

  /*!
    \brief Get the number of bits per sample.

    \return The number of bits per sample.
  */
  virtual float get_nbits(void){return nbits;}

  /*!
    \brief Set the number of bits per sample.

    \param nbits The number of bits per sample.
  */
  virtual void set_nbits(unsigned char nbits){this->nbits = nbits;}

  /*!
    \brief Get the pointer to the filterbank data.

    \return The pointer to the filterbank data.
  */
  virtual unsigned char* get_data(void){return this->data;}

  /*!
    \brief Set the filterbank data pointer.

    \param data A pointer to a block of filterbank data.
  */
  virtual void set_data(unsigned char *data){this->data = data;}

  virtual std::size_t load_gulp(std::size_t start_sample, std::size_t nsamples)
  {
    /* NOT IMPLEMENTED */
    return 0;
  }

  /*!
  \brief Get the centre frequency of the data block.

  \return The centre frequency of the filterbank data.
  */
  virtual float get_cfreq(void)
  {
    if (foff < 0)
      return fch1+foff*nchans/2;
    else
      return fch1-foff*nchans/2;
  }
};


/*!
  \brief A class for handling Sigproc format filterbanks.

  A subclass of the Filterbank class for handling filterbank
  in Sigproc style/format from file. Filterbank memory buffer
  is allocated in constructor and deallocated in destructor.
*/
class SigprocFilterbank: public Filterbank {
public:
  /*!
    \brief Create a new SigprocFilterbank object from a file.

    Constructor opens a filterbank file reads the header and then
    reads all of the data from the filterbank file into CPU RAM.
    Metadata is set from the filterbank header values.

    \param filename Path to a valid sigproc filterbank file.
  */
  SigprocFilterbank(std::string filename)
  {
    // Open filterbank
    this->_filestream.open(filename.c_str(), std::ifstream::in | std::ifstream::binary);
    ErrorChecker::check_file_error(this->_filestream, filename);

    // Read the header
    read_header(this->_filestream, this->_header);

    //removing the last 1 second of data that could have different statistics (EB: eh?)
    this->_header.nsamples -= (std::size_t)(1.0f / this->_header.tsamp);
    this->nsamps = this->_header.nsamples;
    this->nchans = this->_header.nchans;
    this->tsamp = this->_header.tsamp;
    this->nbits = this->_header.nbits;
    this->fch1 = this->_header.fch1;
    this->foff  = this->_header.foff;
    set_data(_data.data());
  }

  /*
   * Load a gulp and return the actual number of samples loaded
   */
  std::size_t load_gulp(std::size_t start_sample, std::size_t nsamples)
  {
    std::cout << "Loading filterbank gulp,"
              << " start = " << start_sample
              << " nsamples = " << nsamples
              << std::endl;
    std::size_t size = static_cast<std::size_t>(nsamples * _header.nbits * _header.nchans / 8);
    _data.resize(size);
    std::size_t byte_offset = start_sample * _header.nbits * _header.nchans / 8;
    _filestream.seekg(_header.size + byte_offset, std::ios::beg);
    _filestream.read(reinterpret_cast<char*>(_data.data()), size);
    std::size_t bytes_read = _filestream.gcount();
    _data.resize(bytes_read);
    set_data(_data.data());
    std::size_t nsamps_read = bytes_read / ( _header.nbits * _header.nchans / 8);
    std::cout << "Loaded " << nsamps_read << " samples" << std::endl;
    return nsamps_read;
  }

  /*!
    \brief Deconstruct a SigprocFilterbank object.

    The deconstructor cleans up memory allocated when
    reading data from file.
  */
  ~SigprocFilterbank()
  {
    this->_filestream.close();
  }

private:
  SigprocHeader _header;
  std::ifstream _filestream;
  std::vector<unsigned char> _data;

};
