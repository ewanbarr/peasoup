//This is place to play with ideas for folding multiple
//candidates from one filterbank file
//WORK IN PROGRESS

#define WARP_SIZE 32

__global__
void fold_subintegration_kernel ()
{
  
  extern __shared__ float buffer [];
  float * acc = buffer;
  int * count = buffer + acc_size;
  
  // Read block of 32 frequency channels
  // Do not wish to re-read any data from global memory
  // May have to sacrifice this to acheive maximum occupancy
  // Otherwise, the corner turn is required to facilitate 
  // coalesced reads.

}
