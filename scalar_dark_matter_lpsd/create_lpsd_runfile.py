"""Create config files to run LPSD on condor or locally."""
import os
from string import Template
import argparse
import numpy as np
import h5py
# Project imports
from utils import LPSDVars


BASE_PATH = os.path.dirname(__file__)


def parse_args():
    """Gather dict of cmdl-args parser output."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    LPSD_args = parser.add_argument_group("LPSD args")
    LPSD_args.add_argument("--input-file", type=str, required=True,
                           help="Path to hdf5 input file.")
    LPSD_args.add_argument("--output-file", type=str, default="output.lpsd",
                           help="Path to output file.")
    LPSD_args.add_argument("--run-file", type=str, default="run_lpsd.sh",
                           help="Path to run file.")
    LPSD_args.add_argument("--path-to-lpsd-exec", type=str, default="lpsd-exec",
                           help="Path to LPSD executable.")
    LPSD_args.add_argument("--fft-file", type=str, default="fft.h5",
                           help="Path to FFT results.")
    LPSD_args.add_argument("--channel", type=str, default=None,
                           help="Dataset within the hdf5 file. Defaults to" +
                           " X1:GDS-CALIB_STRAIN_CLEAN where X is the input file's first char.")
    LPSD_args.add_argument("--data-length", type=int, default=None,
                           help="Length of the data in s. Defaults to entire file.")
    LPSD_args.add_argument("--fmin", type=float, default=10,
                           help="Minimum frequency in Hz. (default: 10)")
    LPSD_args.add_argument("--fmax", type=float, default=None,
                           help="Maximum frequency in Hz, defaults to fs/2.")
    LPSD_args.add_argument("--fsample", type=float, default=16384,
                           help="Sampling frequency in Hz. (default: 16384)")
    LPSD_args.add_argument("--resolution", type=float, default=1e-6,
                           help="Relative bin width for LPSD. (default: 1e-6)")
    LPSD_args.add_argument("--Jdes", type=int, default=None,
                           help="Set to manually override the number of required frequency bins.")
    LPSD_args.add_argument("--block-epsilon", type=float, default=10,
                           help="epsilon factor for block approximation in percent. (default: 10)")
    LPSD_args.add_argument("--constQ-epsilon", type=float, default=1e-15,
                           help="ConstQ relative threshold.")
    LPSD_args.add_argument("--max-memory-power", type=int, default=26,
                           help="2^n is maximum array length for memory management. (default 26)")
    LPSD_args.add_argument("--n-batches", type=int, default=1,
                           help="if > 1 then launch jobs in parallel fashion.")
    LPSD_args.add_argument("--method", type=int, choices=[1, 2, 3, 5], default=5,
                           help="1 for block, 2 for constQ, 3 for constQ FFT-only, 5 for 3+2.")

    condor_parser = parser.add_argument_group("Condor args")
    condor_parser.add_argument("--use-condor", action="store_true")
    condor_parser.add_argument("--request-cpus", type=int, default=1)
    condor_parser.add_argument("--request-disk-GB", type=int, default=16)
    condor_parser.add_argument("--request-memory-GB", type=int, default=16)
    condor_parser.add_argument("--accounting-group", type=str,
                               default="ligo.prod.o4.cw.darkmatter.lpsd")
    condor_parser.add_argument("--accounting-group-user", type=str, required=True)
    condor_parser.add_argument("--submit-file", type=str, default="lpsd.submit",
                               help="Path to submit file to write.")
    condor_parser.add_argument("--outdir", type=str, default=".",
                               help="Where to store condor products.")
    condor_parser.add_argument("--python-executable", type=str,
                               default="/home/alexandresebastien.goettel/.conda/envs/scalardarkmatter/bin/python",
                               help="Path to python executable.")
    return vars(parser.parse_args())


def write_template(args, infile, ofile, permissions=0o664, **kwargs):
    """Write the args-substituted contents of infile to outfile."""
    infile = os.path.join(BASE_PATH, "templates", infile)
    with open(os.path.join(os.path.split("__file__")[0],
                           infile), "r") as _f:
        contents = _f.read()

    print(f"Writing to '{ofile}'..")
    with open(ofile, "w") as _f:
        _args = args.copy()
        _args.update(kwargs)
        _f.write(Template(contents).safe_substitute(_args))
    os.chmod(ofile, permissions)


def exponential_separator(Jdes, n_batches, theta=2e-6):
    """Generate exponentially-distanced iteration start points for given Jdes, n_batches."""
    # Recursive answer
    a, b = np.zeros(n_batches - 1), np.zeros(n_batches - 1)
    a[0], b[0] = 0.5, 0.5
    for i in range(1, n_batches - 1):
        a[i], b[i] = 1 / (2 - a[i - 1]), b[i - 1] / (2 - a[i - 1])

    # Now reverse solve
    N = np.zeros(n_batches)
    N[-1] = Jdes

    for i in range(n_batches - 2, -1, -1):
        if theta*Jdes <= 700:
            N[i] = np.log(a[i]*np.exp(theta*N[i+1]) + b[i]) / theta
        else:
            N[i] = N[i+1] + np.log(a[i]) / theta

    return N


def main(args):
    """
    Generates a LPSD run file and, optionally, condor submit file.

    Parameters:
    - args (dict)

    Raises:
    - ValueError: If the data length is insufficient for the provided 'fmin' and 'resolution'.
    """
    # Make sure default cases are treated correctly
    if args["channel"] is None:
        args["channel"] = f"{args['input_file'][0]}1:GDS-CALIB_STRAIN_CLEAN"

    if args["fmax"] is None:
        args["fmax"] = args["fsample"] / 2

    if args["Jdes"] is None:
        lpsd_args = LPSDVars(*map(lambda x: args[x],
                                  ["fmin", "fmax", "resolution", "fsample", "block_epsilon"]))
        args["Jdes"] = lpsd_args.Jdes

    if args["data_length"] is None:
        with h5py.File(args["input_file"], "r") as _f:
            args["data_length"] = len(_f[args["channel"]]) // args["fsample"]

    # Check if the settings are viable
    if args["data_length"] < 1 / (args["fmin"] * args["resolution"]):
        _fmin = int(np.ceil(1. / (args["data_length"] * args["resolution"])))
        raise ValueError(f"Insufficient data length\nUnder current settings, try fmin >= {_fmin}")

    if args["n_batches"] == 1:
        args["index_string"] = f"(0 {args['Jdes']})"
    else:
        _str = " ".join(map(lambda x: str(int(x)),
                            exponential_separator(args["Jdes"], args["n_batches"])))
        args["index_string"] = f"(0 {_str})"

    # Change relative paths to absolute
    for path_arg in ["run_file", "input_file", "output_file", "fft_file", "path_to_lpsd_exec",
                     "submit_file", "outdir"]:
        args[path_arg] = os.path.abspath(args[path_arg])

    prefix, _ = os.path.splitext(args["run_file"])
    constQ_runfile = f"{prefix}_constQ.sh"
    combine_submitfile = f"{prefix}_combine.submit"
    combine_wrapper = f"{prefix}_combine.sh"
    if args["method"] == 5:
        assert args["use_condor"]
        # Run both method 3 and 2 using a DAG
        fft_runfile = f"{prefix}_FFT.sh"

        prefix, _ = os.path.splitext(args["submit_file"])
        dagfile = f"{prefix}_dag.submit"
        fft_submitfile = f"{prefix}_FFT.submit"

        write_template(args, "lpsd_config.template", fft_runfile, permissions=0o775,
                       method=3)

        write_template(args, "lpsd_config.template", constQ_runfile, permissions=0o775,
                       method=2)

        prefix = os.path.splitext(os.path.split(args["output_file"])[-1])[0]
        write_template(args, "lpsd_submitfile.template", args["submit_file"],
                       prefix=prefix,
                       run_file=constQ_runfile)

        write_template(args, "lpsd_submitfile.template", fft_submitfile,
                       n_batches=1,
                       run_file=fft_runfile,
                       prefix=prefix + "_FFT")

        write_template(args, "combine_submitfile.template", combine_submitfile,
                       outfile=args["output_file"],
                       run_file=combine_wrapper,
                       outdir=os.path.dirname(args["output_file"]),
                       prefix=prefix)

        write_template(args, "combine_wrapper.template", combine_wrapper, permissions=0o775,
                       script_path=os.path.join(BASE_PATH, "combine_lpsd_output.py"),
                       prefix=prefix)

        # Write the DAG file
        with open(dagfile, "w") as f:
            f.write(f"JOB FFT {fft_submitfile}\nJOB LPSD {args['submit_file']}\n"
                    f"JOB combine {combine_submitfile}\nPARENT FFT CHILD LPSD\n"
                    "PARENT LPSD CHILD combine\n")
        print(f"Wrote DAG file to '{dagfile}'")
    else:
        # Write a run file and corresponding condor submit file
        prefix = os.path.splitext(os.path.split(args["output_file"])[-1])[0]
        write_template(args, "lpsd_config.template", args["run_file"], permissions=0o775)
        if not args["use_condor"]:
            return
        write_template(args, "lpsd_submitfile.template", args["submit_file"],
                       prefix=prefix)
        # Combine job
        write_template(args, "combine_submitfile.template", combine_submitfile,
                       outfile=args["output_file"],
                       run_file=combine_wrapper,
                       outdir=os.path.dirname(args["output_file"]),
                       prefix=prefix)
        write_template(args, "combine_wrapper.template", combine_wrapper, permissions=0o775,
                       script_path=os.path.join(BASE_PATH, "combine_lpsd_output.py"),
                       prefix=prefix)
        # DAG
        prefix, _ = os.path.splitext(args["submit_file"])
        dagfile = f"{prefix}_dag.submit"
        with open(dagfile, "w") as f:
            f.write(f"JOB LPSD {args['submit_file']}\n"
                    f"JOB combine {combine_submitfile}\n"
                    "PARENT LPSD CHILD combine\n")
        print(f"Wrote DAG file to '{dagfile}'")


if __name__ == '__main__':
    main(parse_args())
