peasoup_32
==========


C++/CUDA GPU pulsar searching library 

The Master branch has been incorporated as the default source code in the original GitHub repository  (https://github.com/ewanbarr/peasoup/)

**peasoup** has been modified for handling 32 bit dedispersed timeseries. This change was necessary to ensure that the scaling done by the dedisp library does not lead to severe compression of the dynamic range of the timeseries. This change has already demonstrated a siginificant change in the detection SNR while doing the Fourier search after resampling. **peasoup_32** can now also take in an input DM list file from the user or stick to the DM range and tolerance options.




