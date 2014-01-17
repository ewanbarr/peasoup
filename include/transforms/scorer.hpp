#pragma once
#include "stdio.h"
#include "data_types/candidates.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

class CandidateScorer {
private:
  float tsamp;
  float cfreq;
  float foff;
  float tdm_chan_partial;
  float tdm_band_partial;
  int max_dm_gap;
  
  inline float tdm_chan(float ddm){
    return ddm*tdm_chan_partial;
  }

  inline float tdm_band(float ddm){
    return ddm*tdm_band_partial;
  }
  
  inline bool has_physical_period(Candidate& cand){
    return 1.0/cand.freq>tdm_chan(cand.dm);
  }

  inline bool has_adjacency(Candidate& cand){
    int idx = cand.dm_idx;
    bool adjacent = false; 
    bool unique = true;
    for (int ii=0;ii<cand.assoc.size();ii++){
      if (cand.assoc[ii].dm_idx!=idx)
	unique = false;
      if (cand.assoc[ii].dm_idx==idx+1 || cand.assoc[ii].dm_idx==idx-1){
	adjacent = true;
	break;
      }
    }
    if (adjacent || unique)
      return true;
    else
      return false;
  }
  
  inline void delta_dm_ratio(Candidate& cand){
    int inside_count = 1;
    int total_count = 1;
    float inside_snr = cand.snr;
    float total_snr = cand.snr;
    float ddm = 1.0/(cand.freq*tdm_band_partial);
    for (int ii=0;ii<cand.assoc.size();ii++){
      total_count++;
      total_snr+=cand.assoc[ii].snr;
      if (fabs(cand.dm-cand.assoc[ii].dm) <= ddm){
	inside_count++;
	inside_snr+=cand.assoc[ii].snr;
      }
    }
    float count_ratio = (float) inside_count/total_count;
    float snr_ratio = (float) inside_snr/total_snr;
    cand.ddm_count_ratio = count_ratio;
    cand.ddm_snr_ratio = snr_ratio;
  }
    
public:  
  CandidateScorer(float tsamp, float cfreq, float foff, float bw)
    :tsamp(tsamp),cfreq(cfreq),foff(foff)
  {
    float ftop = cfreq+bw/2.0;
    float fbottom = cfreq-bw/2.0;
    tdm_chan_partial = 8300.0 * foff / std::pow(cfreq,3.0);
    tdm_band_partial = 4150.0*(1.0/std::pow(fbottom,2) - 1.0/std::pow(ftop,2));
  }

  void score(Candidate& cand){
    cand.is_physical = has_physical_period(cand);
    cand.is_adjacent = has_adjacency(cand);
    delta_dm_ratio(cand);
  }

  void score_all(std::vector<Candidate>& cands){
    for (int ii=0;ii<cands.size();ii++)
      score(cands[ii]);
  }
};
