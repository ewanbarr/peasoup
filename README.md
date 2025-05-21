# Peasoup

**Peasoup** is a fast, GPU-accelerated acceleration search pipeline for discovering compact binary pulsars in high time resolution radio observations. It operates on filterbank data and outputs an XML file of candidate detections. Peasoup is optimized for large-scale, high-throughput pulsar searches on modern GPU architectures. It uses the library [`dedisp`](https://github.com/vishnubk/dedisp) originally written by [`Ben Barsdell`](https://github.com/benbarsdell)  to perform dedispersion.


> ⚠️ Peasoup does **not** fold candidates. Use tools like [`psrfold_fil`](https://github.com/ypmen/PulsarX) (from PulsarX -> recommended), [`prepfold`](https://github.com/scottransom/presto) (from PRESTO), or [`dspsr`](https://dspsr.sourceforge.net/current/), to fold using candidate outputs.


## New Features in the Latest Version

- **CUDA 12.6 Compatibility**  
  Peasoup now builds and runs successfully with CUDA versions up to **12.6**.  
  > Previous versions of `dedisp` failed to compile with CUDA > 12 due to the deprecation of legacy texture memory code.


- **Segmented Acceleration Search Support**  
  Peasoup now supports **segmented acceleration searches**, allowing you to process specific chunks of your filterbank file. You can control the search segment using:

  - `--start_sample`: Index of the first sample (inclusive) where the segment begins.
  - `--fft_size`: Number of samples to use in the FFT. This defines the length of the filterbank file to be analyzed. If the specified range exceeds the end of the filterbank, Peasoup will **zero-pad** the remaining samples after baseline subtraction.
  - `--nsamples` *(Optional)*: Exact number of samples to read from the filterbank file. 



   **Difference between `--fft_size` and `--nsamples`**:
  
  - `--fft_size` determines the **search transform size**, i.e., how many samples Peasoup will use *regardless of what's available*—it pads with zeros if needed.
   - `--nsamples` enforces a **strict upper bound** on how many real samples. If `--fft_size` > `--nsamples`, then the software will pad zeros after reading `--nsamples` from the filterbank file.
  
  *Recommended usage:* Use `--start_sample` and `--fft_size` for most search scenarios. Use `--nsamples` only when you want to tightly control the **actual** number of samples read from the filterbank file.

  ### Example: Illustrating the difference.

  - Using `--start_sample=0.25*total`, `--fft_size=0.5*total`:  
    → Peasoup reads 25% to 75% of the filterbank file time samples (Applicable to most scenarios).
  
  - Using `--start_sample=0.25*total`, `--nsamples=0.25*total`, `--fft_size=0.5*total`:  
    → Peasoup reads only 25% to 50% of the filterbank file time samples, and pads an additional `0.25*total` of zeros.

- **Coherent DM Correction (`--cdm`)**  
  If your filterbank file has been **coherently dedispersed to a non-zero DM**, you can now inform Peasoup using the `--cdm` flag.  
  - This modifies the acceleration plan, making the search step size finer near the coherent DM value and improving sensitivity. 



## Installation

### Recommended: Docker (or Singularity for HPC)

Pull from DockerHub:

```bash
docker pull vishnubk/peasoup
```
Singularity/Apptainer
```bash
apptainer pull docker://vishnubk/peasoup
```

Build from Source (Advanced)

After cloning the repo, edit `Makefile.inc` to point to the location where you have CUDA installated. Current version supports from `sm_60` to `sm_90`. If your GPU's compute capability is not covered in this range, add the appropriate `-gencode` flags to the `GPU_ARCH_FLAG` variable to both `dedisp` and `peasoup` to ensure compatibility.

```bash
git clone https://github.com/vishnubk/dedisp.git
cd dedisp
make 
make install

cd ..
git clone https://github.com/vishnubk/peasoup.git
cd peasoup
make
make install
```

Basic Usage

```bash
peasoup -i data.fil \
        --fft_size 67108864 \
        --limit 100000 \
        -m 7.0 \
        -o output_dir \
        -t 1 \
        --acc_start -50 \
        --acc_end 50 \
        --dm_file my_dm_trials.txt \
        --ram_limit_gb 180.0 \
        -n 4

```

> ⚠️ IMPORTANT: Always specify `--fft_size` explicitly. If omitted, Peasoup defaults to the nearest lower power-of-two FFT size. While `cuFFT` supports efficient FFTs for sizes composed of 2, 3, 5, or 7 as prime factors, performance can vary across GPUs. For large-scale surveys, we recommend benchmarking different FFT sizes and explicitly setting `--fft_size` to avoid issues from observations that fall slightly short of the next optimal size. Peasoup still supports specifying a DM range using `--dm_start` and `--dm_end`, allowing `dedisp` to generate internal trial steps. However, we strongly recommend using the `--dm_file` flag to provide full control over dispersion trials. Create a file where each line contains a DM value, and pass it to Peasoup using `--dm_file`. You can generate this file using `DDplan.py` from the PRESTO suite to ensure optimal coverage.

Folding Candidates

Using PulsarX

Use the value of `--segment_pepoch` from the Peasoup output XML as `--pepoch` in either `psrfold_tim`, `psrfold_fil` or `psrfold_fil2`:

```bash

psrfold_fil -v -t 12 --candfile pulsarx.candfile \
    -n 64 -b 128 --template meerkat_fold.template \
    -f data.fil --pepoch ${segment_pepoch} -o results
```

Using `prepfold` (PRESTO)

Peasoup reports spin period referring to the middle epoch of your `--fft_size` whereas `prepfold` needs the period at the beginning of your observation. To fold peasoup search candidates with `prepfold`, you need to do three steps:  

A) Convert acceleration to Pdot:

```python

def a_to_pdot(P_s, acc_ms2):
    c = 2.99792458e8
    return P_s * acc_ms2 / c
```
B) Shift period to the start of the observation:

```python
def period_correction_for_prepfold(p0, pdot, tsamp, fft_size):
    return p0 - pdot * fft_size * tsamp / 2
```

C) Add the `--topo` flag while folding with `prepfold`. This is because `Peasoup` does not barycenter the timeseries and therefore your non-zero acceleration search detections are only valid in topocentric units.

```python

prepfold -topo -noxwin -p ${corrected_period} -pd ${pdot} data.fil

```

Using DSPSR 

Create a phase predictor file and pass that to `-P` flag in `dspsr`. Use the value of `--segment_pepoch` from the Peasoup output XML as `EPOCH` inside your phase predictor file.
 

```bash
SOURCE: J1546-5431
EPOCH: 55739.5399653 #This should be same as --segment_epoch from peasoup
PERIOD: 1.466892342 s
DM: 316.2835
ACC: 1.25571819897 (m/s/s)
RA: 15:46:48.00
DEC: -54:31:00.216
```

Acknowledgements

Peasoup is almost entirely written by [`Ewan Barr`](https://github.com/ewanbarr) (MPIfR). With minor contributions over the years from different people, including Vivek, Vishnu, Prajwal, Yunpeng and Jiri. It has already been used to discover > 200 pulsars. There may be a paper on this someday, but until then, if you use **Peasoup** in a publication, please cite the repository.



