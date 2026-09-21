Last update: 22.05.25
Contact: [Alexandre Göttel](mailto:alexandresebastien.goettel@ligo.org)

---
# Running LPSD, from strain to candidate list and upper limits
The goal of the LPSD pipeline is to search for scalar dark matter while maximizing SNR in every single frequency bin, by adjusting the integration time in every bin to be exactly the corresponding DM coherence time. Details of the underlying physics, as well as a detailed description of the pipeline, can be found [here](https://doi.org/10.1103/PhysRevLett.133.101001) and [here](https://www.nature.com/articles/s41598-025-33428-2). This document is meant to show how to run the pipeline by guiding the reader through an example run through it.

## 0. Installation
To run the scripts mentioned below, you need two things
- A (e.g. conda) environment in which to run the python code
- A compiled LPSD C executable

To get a working environment, one can simply clone an igwn environment on CIT, then add the findiff package. This is used to estimate the upper limit uncertainty in the maximum likelihood estimation. To create an environment called *scalardarkmatter*:
```bash
conda create --clone igwn --name scalardarkmatter
conda activate scalardarkmatter
conda install conda-forge::findiff
```

Compiling the LPSD C code is also relatively easy as a script (LPSD/make_install.sh) is provided to perform the installation through cmake. Running the script in bash will create a build and install directory in the current working directory, and create an executable `install/lpsd-exec`.
Before you begin, ensure you have the following dependencies installed on your system:
- [gcc and g++](https://gcc.gnu.org/) with C++14 support (version 11.4.0 or higher)
- [CMake](https://cmake.org/) (version 3.4.1 or higher)
- [HDF5](https://www.hdfgroup.org/solutions/hdf5/) (version 1.14.4.3 or higher)
- [GSL](https://www.gnu.org/software/gsl/) (GNU Scientific Library 2.8 or higher)

you can install the dependencies using `apt-get`:

```bash
sudo apt-get update
sudo apt-get install cmake libhdf5-dev libgsl-dev
```
Or, if you don't have access to `sudo apt-get` (e.g. on CIT), you can build them from source.

## 1. Running the core LPSD algorithm
With input time-domain data "on hand" one can immediately start running LPSD. This can be done manually / locally, see (examples/run_lpsd.sh), but it is recommended to use (scripts/lvk/create_lpsd_runfile.py).

This is run with arguments:
- `input-file`: The path to the time-domain strain
- `output-file`: Full output file path
- `run-file`: Full path to bash script that will be run by condor, it is a good idea to create a dedicated run directory
- `fft-file`: Path to where the iterim FFT products will be saved
- `channel`: This should be "{ifo}:GDS-CALIB_STRAIN_CLEAN_GATED_G02", but will depend on where you got the strain data
- `fsample`: This has to be the exact sampling rate used in the input-file, in Hz.

In order to run using condor, you have set the flag:
- `--use-condor`

and give the arguments:
- `accounting-group`: appropriate LDG accounting tag
- `accounting-group-user`: (you!)
- `python-executable`: path to the python executable of your conda environment, to make sure it is properly used.

If using condor, this will create a dag submit file, which can be started using `condor_submit_dag`. Otherwise, this will simply create a bash script to run LPSD with the desired frequency resolution on the input file.

More optional variables can be set, use `--help`.

## 2. Fitting the background
The next step in the analysis is to fit the PSD spectra. This is done by running python, using (scripts/lvk/fit_background.py) with the arguments:
- `data-path`: path to the directory containing the PSDs. If running on a single file, path to a psd file.
- `data-prefix`: if `data-path`is a dir, the script will use all the files in that dir that start with `data-prefix`.
- `output-path`: full path to the output (.json) file.

The default values for all other arguments should work well. The most notable are:
- `--verbose`: flag to make plots of every fit.
- `plot-path`: path to directory to save the plots. Required if `--verbose` was given.

## 3. Maximising the likelihood
With PSD data and a background model, we can start the last computationally intensive step in this pipeline: maximising a likelihood function in each bin. The products of this step will be used to find candidates, and to set upper limits. Meant to be used on CIT, use (scripts/lvk/maximise_likelihood.py) with the arguments:
- `data-path` and data-prefix, same as with fit_background.py above.
- `rundir`: path to a directory in which to save the job's run files.
- `outdir`: path to a directory in which to save the job's output.
- `prefix`: prefix for the job's files. Using different prefix allows to re-use the same run and out directories for different jobs.
- `bkg-info-path`: path to the output .json file from the previous step.
- `tf-path`: path to the directory with the transfer functions.

Because this will create dag submit files for condor (on CIT), you must also give:
and give the arguments:
- `accounting-group`: appropriate LDG accounting tag
- `accounting-group-user`: (you!)
- `python-executable`: path to the python executable of your conda environment, to make sure it is properly used.

## 4.1 Creating a candidate list
This can take a few minutes because of maximum likelihood maximisation to compare results in both individual detectors. Use [scripts/get_candidates](https://git.ligo.org/alexandresebastien.goettel/Scalar-Dark-Matter-LPSD/-/blob/O4a/scripts/get_candidates.py) with:
- `max-lkl-path`: path to the .npy output file from step 3.
- `bkg-path`: path to the .json output file from step 2.
- `data-path`: path to the directory containing the lpsd output files from step 1.
- `data-prefix`: prefix to help find the lpsd files (prefix*).
- `tf-path`: path to the directory containing the transfer functions.
- `out-folder`: where to store the candidate list (.csv) and candidate plots.

## 4.2 Calculating upper limits
This needs about 1-2 minutes on most cpus. Use (scripts/calc_upper_limits.py) with:
- `max-lkl-path`: path to the .npy output file from step 3.
- `output-file`: path to the .npy output file for this step.

The default arguments here will ensure the upper limits are created for a 95% C.L.
