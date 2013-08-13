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
  float folded_snr;
  float opt_period;
  
  Candidate(float dm, int dm_idx, float acc, int nh, float snr, float freq)
    :dm(dm),dm_idx(dm_idx),acc(acc),nh(nh),
     snr(snr),folded_snr(0.0),freq(freq),
     opt_period(0.0){}
  
  Candidate(float dm, int dm_idx, float acc, int nh, float snr, float folded_snr, float freq)
    :dm(dm),dm_idx(dm_idx),acc(acc),nh(nh),snr(snr),
     folded_snr(folded_snr),freq(freq),opt_period(0.0){}

  void print(FILE* fo=stdout){
    fprintf(fo,"%.9f\t%.9f\t%.9f\t%.2f\t%.2f\t%d\t%.1f\t%.1f\n",1.0/freq,opt_period,freq,dm,acc,nh,snr,folded_snr);
  }

};

class CandidateCollection {
public:
  std::vector<Candidate> cands;

  CandidateCollection(){}
  
  void append(CandidateCollection& other){
    cands.insert(cands.end(),other.cands.begin(),other.cands.end());
  }
  
  void append(std::vector<Candidate> other){
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

class Event: public Candidate {
public:
  std::vector<Candidate>* cands; //for memory efficiency this is stored as a ptr.
  
  Event(Candidate fundamental)
    :Candidate(fundamental){
    cands = new std::vector<Candidate>;
    cands->reserve(100);
  }
  
  void append(Candidate& cand){
    cands->push_back(cand);
  }
  
};
