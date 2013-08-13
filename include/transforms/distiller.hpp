#pragma once
#include "stdio.h"
#include "data_types/candidates.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

#define SPEED_OF_LIGHT 299792458.0

struct snr_less_than {
  bool operator()(const Candidate& x, const Candidate& y){
    return (x.snr>y.snr);
  }
};

class BaseDistiller {
protected:
  std::vector<bool> unique;
  int size;
  virtual void condition(std::vector<Candidate>& cands, int idx){}

public:
  std::vector<Candidate> distill(std::vector<Candidate>& cands)
  {
    size = cands.size();
    unique.resize(size);
    std::fill(unique.begin(),unique.end(),true);
    std::sort(cands.begin(),cands.end(),snr_less_than()); //Sort by snr !IMPORTANT       
    int ii;
    int idx;
    int start=0;
    int count=0;
    while (true) {
      idx = -1; //Sentinel value                                                    
      for(ii=start;ii<size;ii++){
        if (unique[ii]){
          start = ii+1;
          idx = ii;
          break;
        }
      }
      if (idx==-1)
        break;
      count++;
      condition(cands,idx);
    }
    std::vector<Candidate> new_cands;
    new_cands.reserve(count);
    for (ii=0;ii<size;ii++){
      if (unique[ii])
        new_cands.push_back(cands[ii]);
    }
    return new_cands;
  }
};

class HarmonicDistiller: public BaseDistiller {
private:
  float tolerance;
  float max_harm;

  void condition(std::vector<Candidate>& cands, int idx)
  {
    int ii,jj,kk;
    float ratio,freq;
    int nh;
    float upper_tol = 1+tolerance;
    float lower_tol = 1-tolerance;
    float fundi_freq = cands[idx].freq;
    for (ii=idx+1;ii<size;ii++){
      freq = cands[ii].freq;
      nh = cands[ii].nh;
      for (jj=1;jj<=max_harm;jj++){
        for (kk=1;kk<=pow(2.0,nh);kk++){
          ratio = kk*freq/(jj*fundi_freq);
          if (ratio>(lower_tol)&&ratio<(upper_tol)){
            unique[ii]=false;
          }
	}
      }
    }
  }
  
public:
  HarmonicDistiller(float tol, float max_harm)
    :tolerance(tol),max_harm(max_harm){}
};

//Remove other candidates with lower S/N and equal or lower harmonic number
//Use a user defined period tolerance, but calculate the delta f for the 
//delta acc between fundamental and test signal.

class AccelerationDistiller: public BaseDistiller {
private:
  float tobs;
  float tobs_over_c;
  float tolerance;
  
  float correct_for_acceleration(float freq, float delta_acc){
    return freq+delta_acc*freq*tobs_over_c;
  }

  void condition(std::vector<Candidate>& cands,int idx)
  {
    int ii,jj,kk;
    float ratio,freq;
    float fundi_freq = cands[idx].freq;
    float fundi_acc = cands[idx].acc;
    float acc_freq;
    float delta_acc;
    float edge = fundi_freq*tolerance;

    for (ii=idx+1;ii<size;ii++){
      /*
      if (cands[ii].nh > cands[idx].nh){
	continue;
      }
      */
      delta_acc = fundi_acc-cands[ii].acc;
      acc_freq = correct_for_acceleration(fundi_freq,delta_acc);
      
      if (acc_freq>fundi_freq){
	if (cands[ii].freq>fundi_freq-edge && cands[ii].freq<acc_freq+edge){
	  unique[ii]=false;
	}
      } else {
	if (cands[ii].freq<fundi_freq+edge && cands[ii].freq>acc_freq-edge){
	  unique[ii]=false;
	}
      }
    }
  }
  
public:
  AccelerationDistiller(float tobs, float tolerance)
    :tobs(tobs),tolerance(tolerance){
    tobs_over_c = tobs/SPEED_OF_LIGHT;
  }
};
//NOTE: +ve acceleration is away from observer


class DMDistiller: public BaseDistiller {
private:
  float tolerance;
  float ratio;

  void condition(std::vector<Candidate>& cands,int idx)
  {
    int ii;
    float fundi_freq = cands[idx].freq;
    float upper_tol = 1+tolerance;
    float lower_tol = 1-tolerance;
    for (ii=idx+1;ii<size;ii++){
      /*
      if (cands[ii].nh > cands[idx].nh){
        continue;
      }
      */
      ratio = cands[ii].freq/fundi_freq;
      if (ratio>(lower_tol)&&ratio<(upper_tol)){
	unique[ii]=false;
      }
    }
  }
  
public:
  DMDistiller(float tolerance)
    :tolerance(tolerance){}
};

