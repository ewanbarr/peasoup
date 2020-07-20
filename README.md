peasoup_32
==========

C++/CUDA GPU pulsar searching library 

The Master branch has been incorporated as the default source code in the original GitHub repository  (https://github.com/ewanbarr/peasoup/)

**peasoup** has been modified for handling 32 bit dedispersed timeseries. This change was necessary to ensure that the scaling done by the dedisp library does not lead to severe compression of the dynamic range of the timeseries. This change has already demonstrated a siginificant change in the detection SNR while doing the Fourier search after resampling


There are two other branches in this repository which has been setup to adaptively handle TRAPUM(www.trapum.org) specific needs

1. **user_dm** : This branch takes in a user defined input of a list of DM trials to search on. The default code calculates the DMs based on a tolerance limit and a DM range defined by the input. This feature has been added to add more flexibility for deciding DM values and step sizes. 

2. **jerk_searcher**: This branch has a modified source code to also do a time domain resampling in the jerk phase space too before searching. This is currently work in progress. 

Modified for handling 32 bit dedispersed timeseries. This takes in either a dm range with tolerance or a dm_list file
=======
