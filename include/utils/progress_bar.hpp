#pragma once
#include <unistd.h>
#include "pthread.h"
#include "stdio.h"
#include <utils/stopwatch.hpp>

void *progress_printer(void* progress_){

  float* progress = (float*) progress_;
  float val;
  float prev = 0.0;
  int eta;
  Stopwatch timer;
  float current_time;
  timer.start();
  printf("Estimated time to completion: --- s  (0.0%% completed)        \r");
  fflush(stdout);
  while (true){
    usleep(100000);
    val = *progress;

    if (val<0){
      printf("\nComplete (execution time %.2f s)\n",timer.getTime());
      fflush(stdout);
      pthread_exit(0);
      timer.stop();
      break;
    }
    
    if (val==prev){
      continue;
    } 
    
    current_time = timer.getTime();
    eta = current_time/(val)-current_time;
    fflush(stdout);
    printf("Estimated time to completion: %d s  (%.2f%% completed)        \r",
	   eta,val*100.0);
    fflush(stdout);
    usleep(100000);
    prev = val;
  }
  return NULL;
}

class ProgressBar {
private:
  float* progress;
  pthread_t thread;

public:
  ProgressBar(){
    progress = new float(0.0);
  }

  void start(void){
    pthread_create(&thread, NULL, progress_printer,(void*) progress);
  }

  void stop(void){
    *progress = -1;
    pthread_join(thread,NULL);
  }

  void set_progress(float fraction){
    *progress = fraction;
  }

  ~ProgressBar(){
    delete progress;
  }

};
