"""Combine all maxlkl output jobs into a single .hdf5 output file."""
import os
import glob
import argparse
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt


def parse_args():
    """Parse cmdl-args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Path to LPSD output folder.")
    parser.add_argument("--prefix", type=str, default="output",
                        help="Prefix for the naming of the LPSD output files.")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Path to combined output file.")

    return vars(parser.parse_args())


def main(results_dir=None, prefix="", output_file=None, verbose=False):
    input_files = glob.glob(os.path.join(results_dir, f"{prefix}_*_result.npy"))

    data = None
    for outfile in tqdm(input_files, desc="Read & combine data"):
        _data = np.load(outfile)
        if data is None:
            data = _data
        else:
            data = np.concatenate([data, _data])

    # Sort by frequency axis
    data = data[np.argsort(data[:, 0])]
    np.save(output_file, data)

    if verbose:
        # Validation plots
        # "zero_lkl", "max_lkl", "mu", "sigma"
        plt.figure()
        ax = plt.subplot(111)
        ax.set_xscale("log")
        ax.plot(data[:, 0], -data[:, 1], label="-zero_lkl")
        ax.plot(data[:, 0], -data[:, 2], label="-max_lkl")
        ax.legend(loc="best")
        plt.savefig("lkl.pdf")

        plt.figure()
        ax.plot(data[:, 0], data[:, 2] - data[:, 1])
        ax.set_title("max_lkl - zero_lkl")
        plt.savefig("diff.pdf")

        plt.figure()
        ax = plt.subplot(111)
        ax.set_xscale("log")
        ax.plot(data[:, 0], data[:, 3], color="C0")
        plt.savefig("mu.pdf")


if __name__ == '__main__':
    main(**parse_args())
