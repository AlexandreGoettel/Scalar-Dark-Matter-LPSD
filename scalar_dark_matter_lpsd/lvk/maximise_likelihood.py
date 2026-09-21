"""Schedule job creation and write submit files for DM hunting."""
import os
import argparse
import glob
import numpy as np
# Project imports
from ..LPSDIO import LPSDOutput


BASE_PATH = os.path.split(os.path.abspath(__file__))[0]


def parse_cmdl_args():
    """Parse cmdl args to pass to main."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add arguments
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to lpsd output file or folder of files.")
    parser.add_argument("--data-prefix", type=str, default=None,
                        help="Prefix of data files.")
    parser.add_argument("--rundir", type=str, required=True,
                        help="Where to store all run files.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Where to store the run output.")
    parser.add_argument("--prefix", type=str, required=True,
                        help="Run file prefix in outdir (for condor).")
    parser.add_argument("--bkg-info-path", type=str, required=False,
                        help="Output data from fit_background.py.")
    parser.add_argument("--peak-shape-path", type=str, default=None,
                        help="Path to the peak_shape_data.npz file. Ignore to use single-bin.")
    parser.add_argument("--python-executable", type=str,
                        default="/home/alexandresebastien.goettel/.conda/envs/scalardarkmatter/bin/python",
                        help="Path to python executable from desired env.")
    parser.add_argument("--tf-path", type=str, required=True,
                        help="Path to directory holding transfer functions.")
    parser.add_argument("--min-log10mu", type=int, default=-40,
                        help="Smallest mu to test in max_lkl search.")

    parser.add_argument("--ana-fmin", type=float, default=10,
                        help="Minimum frequency (Hz)")
    parser.add_argument("--ana-fmax", type=float, default=5000,
                        help="Maximum frequency (Hz)")
    parser.add_argument("--freqs-per-job", type=int, default=35000,
                        help="Frequencies to analyse per job")
    parser.add_argument("--n-processes", type=int, default=4,
                        help="Number of MPI processes per job")
    parser.add_argument("--accounting-group-user", type=str, required=True)
    parser.add_argument("--accounting-group", type=str, default="ligo.dev.o4.cw.darkmatter.lpsd",
                        help="LIGO Accounting group")
    parser.add_argument("--request-memory-combine", type=int, default=1,
                        help="RequestMemory for combine job in GB.")
    parser.add_argument("--request-disk-GB", type=int, default=2,
                        help="Disk request for max_lkl jobs in GB.")
    return vars(parser.parse_args())


def write_submit_wrapper(python_executable, script_path):
    """Write the condor submit executable - wrapping around the python script."""
    _str = f"""#!/bin/bash

# The iteration number is passed by HTCondor as the process number
ITERATION=$(($1 + $2))

# Call the python script
{python_executable} {script_path} --iteration $ITERATION"""
    names = ["ana-fmin", "ana-fmax", "data-path", "data-prefix", "bkg-info-path",
             "peak-shape-file", "output-path", "n-processes", "n-frequencies", "tf-path", "min-log10mu"]
    for i, name in enumerate(names):
        _str += f" --{name} ${{{i+3}}}"
    return _str + "\n"

def write_combine_wrapper(python_executable, script_path):
    """Write the condor combine executable - wrapping around the python script."""
    _str = f"""#!/bin/bash

# Call the python script
{python_executable} {script_path} --results-dir $1 --prefix $2 --output-file $3
"""
    return _str


def get_condor_submit(executable, args, accounting_group, accounting_group_user, request_memory,
                      request_cpus, request_disk, out_path, queue):
    """Write a generic condor submit file to string."""
    _str = f"""Universe = vanilla
Executable = {executable}
Arguments = {args}
accounting_group = {accounting_group}
accounting_group_user = {accounting_group_user}
request_cpus = {request_cpus}
request_disk = {request_disk} GB
request_memory = {request_memory} GB

Output = {out_path}.out
Error = {out_path}.err
Log = {out_path}.log

Queue {queue}
"""
    return _str


def write_submit_file(N_start, N_end, outdir, prefix, peak_shape_path, n_freqs, ana_fmin,
                      ana_fmax, json_path, tf_path, data_path, data_prefix, min_log10mu,
                      request_cpus, **kwargs):
    """Write the condor submit file for DM hunting jobs."""
    out_path = os.path.join(outdir, f"{prefix}_$(Process)")
    args = " ".join([f"{x}" for x in [ana_fmin, ana_fmax, data_path, data_prefix, json_path,
                                      peak_shape_path, os.path.join(outdir, prefix), request_cpus,
                                      n_freqs, tf_path, min_log10mu]])
    return get_condor_submit(
        args=f"$(Process) {N_start} {args}",
        out_path=out_path,
        queue=N_end + 1 - N_start,
        request_cpus=request_cpus,
        **kwargs
    )


def write_combine_submit(results_dir, prefix, **kwargs):
    """Write the submit file for the combine job."""
    kwargs["out_path"] = os.path.join(results_dir, "combine")
    output_file = os.path.join(results_dir, f"combined_{prefix}.npy")
    return get_condor_submit(
        args = f"{results_dir} {prefix} {output_file}",
        **kwargs
    )



def writefile(_str, filename, permissions=0o664):
    """Write the contents of _str to filename."""
    with open(filename, "w") as _f:
        _f.write(_str)
    os.chmod(filename, permissions)


def main(rundir=None, outdir=None, prefix=None, ana_fmin=10, ana_fmax=5000, freqs_per_job=35000,
         data_path=None, bkg_info_path=None, peak_shape_path=None, data_prefix="", n_processes=4,
         accounting_group="", accounting_group_user="", python_executable=None, min_log10mu=-40,
         tf_path=None, request_memory_combine=1, request_disk_GB=4, **_):
    """Organise argument creation and job submission."""
    if os.path.isdir(data_path):
        if data_prefix == "" or data_prefix is None:
            raise ValueError("If data_path is a directory, data_prefix must be given.")
        path = glob.glob(os.path.join(data_path, f"{data_prefix}*"))[0]
    else:
        path = data_path
    data = LPSDOutput(path)

    iter_start, iter_end = None, None
    for i in range(len(data)//freqs_per_job + 1):
        if iter_start is None and data.freq[(i+1)*freqs_per_job] >= ana_fmin:
            iter_start = i
        if (i + 1) * freqs_per_job >= len(data) or data.freq[(i+1)*freqs_per_job] >= ana_fmax:
            iter_end = i
            break
    else:
        raise ValueError

    # Write submit files, coord with executable
    isolated_prefix = os.path.split(prefix)[-1]
    run_prefix = os.path.abspath(os.path.join(rundir, isolated_prefix))
    path_to_wrapper = f"{run_prefix}_wrapper.sh"
    path_to_executable = os.path.join(BASE_PATH, "lvk", "max_lkl_executable.py")
    path_to_submitfile = f"{run_prefix}.submit"
    path_to_combine_wrapper = f"{run_prefix}_combine_wrapper.sh"
    path_to_combine_exe = os.path.join(BASE_PATH, "lvk", "combine_maxlkl_output.py")
    path_to_combine_submit = f"{run_prefix}_combine.submit"
    path_to_dag = f"{run_prefix}_dag.submit"

    # Main jobs
    writefile(
        write_submit_wrapper(python_executable, path_to_executable), path_to_wrapper,
        permissions=0o775
    )
    writefile(
        write_submit_file(
            iter_start, iter_end, outdir, prefix, peak_shape_path, freqs_per_job, ana_fmin,
            ana_fmax, bkg_info_path, tf_path, data_path, data_prefix, min_log10mu, n_processes,
            executable=path_to_wrapper, accounting_group=accounting_group,
            accounting_group_user=accounting_group_user, request_memory=4,
            request_disk=request_disk_GB
        ),
        path_to_submitfile
    )

    # Combine job
    writefile(
        write_combine_wrapper(python_executable, path_to_combine_exe), path_to_combine_wrapper,
        permissions=0o775
    )
    writefile(
        write_combine_submit(
            outdir, prefix, executable=path_to_combine_wrapper, accounting_group=accounting_group,
            accounting_group_user=accounting_group_user, request_memory=request_memory_combine,
            request_cpus=1, queue=1, request_disk=1
        ),
        path_to_combine_submit
    )

    # Write DAG
    dag_str = f"""JOB main {path_to_submitfile}
JOB combine {path_to_combine_submit}
PARENT main CHILD combine
"""
    writefile(dag_str, path_to_dag)

    print(f"Wrote dag submit file to {path_to_dag}.")


if __name__ == '__main__':
    main(**parse_cmdl_args())
