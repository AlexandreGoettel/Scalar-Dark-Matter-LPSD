#!/bin/bash

# Declare data file, must be hdf5 file
filename=
dataset=

# Declare number of seconds data in file
TSlength=

# Declare start, end, and sampling frequencies
f_min=
f_max=
f_sample=

# Declare 'Jdes': The total number of required frequency bins
Nfreqs_full=

# n for maximum array size 2^n
n_memory=26
constQ_eps=1e-5

# Declare the output file(s)
outfilename=
fft_file=

$path_to_lpsd_exec \
	-A 2 \
	-b 0 \
	-e ${TSlength} \
	-f ${f_sample} \
	-h 2 \
	-i ${filename}\
	-D ${dataset}\
	-l 20 \
	-n ${$Nfreqs_full} \
	-o ${outfilename} \
	-r 0 \
	-s ${f_min} \
	-t ${f_max} \
	-T \
	-w -2 \
	-p 238.13 \
	-x 1 \
	-N 0 \
	-J ${Nfreqs_full} \
	-M ${n_memory} \
	-R ${constQ_eps} \
	-O ${fft_file}
