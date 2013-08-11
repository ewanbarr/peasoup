#pragma once
#include <iostream>
#include <vector>
#include "stdio.h"

struct Candidate {
public:
  float dm;
  int dm_idx;
  float acc;
  int nh;
  float snr;
  float freq;
  
  Candidate(float dm, int dm_idx, float acc, int nh, float snr, float freq)
    :dm(dm),dm_idx(dm_idx),acc(acc),nh(nh),snr(snr),freq(freq){}
  
  void print(FILE* fo=stdout){
    fprintf(fo,"%.9f\t%.2f\t%.2f\t%d\t%.1f\n",freq,dm,acc,nh,snr);
  }

};

class CandidateCollection {
public:
  std::vector<Candidate> cands;

  CandidateCollection(){}
  
  void append(CandidateCollection& other)
  {
    cands.insert(cands.end(),other.cands.begin(),other.cands.end());
  }

  void append(std::vector<Candidate> other)
  {
    cands.insert(cands.end(),other.begin(),other.end());
  }

  void reset(void){
    cands.clear();
  }

  void print(FILE* fo=stdout){
    for (int ii=0;ii<cands.size();ii++)
      cands[ii].print(fo);
  }
};


class SpectrumCandidates: public CandidateCollection {
public:
  float dm;
  int dm_idx;
  float acc;

  SpectrumCandidates(float dm, int dm_idx, float acc)
    :dm(dm),dm_idx(dm_idx),acc(acc){}
  
  void append(float* snrs, float* freqs, int nh, int size){
    cands.reserve(size+cands.size());
    for (int ii=0;ii<size;ii++)
      cands.push_back(Candidate(dm,dm_idx,acc,nh,snrs[ii],freqs[ii]));
  }
};


