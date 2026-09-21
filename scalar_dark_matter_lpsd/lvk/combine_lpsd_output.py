"""Combine all lpsd output jobs into a single .lpsd output file."""
import os
import argparse
import glob
from tqdm import tqdm
import numpy as np


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


def main(results_dir=None, prefix="output", output_file=None):
    """Loop over all files starting with prefix in results_dir, combine to output_file."""
    input_files = glob.glob(os.path.join(results_dir, f"{prefix}.lpsd.*"))

    header, file_data = None, None
    for _file in tqdm(input_files, desc="Read & combine data"):
        if header is None:
            with open(_file, "r") as _f:
                header = "".join([line for line in _f.readlines() if line.startswith("#")])

        # Cross-checks
        size = -1
        with open(_file, "r") as _f:
            for line in _f.readlines():
                if line.startswith("# Size"):
                    size = int("\t".join(line.split(" ")).split("\t")[2])
                    break
            else:
                raise IOError(f"No size value found in {_file}")

        _file_data = np.loadtxt(_file, comments="#")
        if len(_file_data) == 0:
            tqdm.write(f"No data in {_file}..")
        elif abs(len(_file_data) - size) > 1:
            tqdm.write(f"Inconsistent nbatch={size} and {len(_file_data)} lines in '{_file}'..")

        if file_data is not None:
            file_data = np.vstack((file_data, _file_data))
        else:
            file_data = _file_data

    # Sort by frequency
    idx = np.argsort(file_data[:, 0])
    file_data = file_data[idx, :4]

    print(f"Writing {file_data.shape[0]} lines to {output_file}..")
    np.savetxt(output_file, file_data, comments="",
               delimiter="\t", header=header, fmt="%.10e\t%.10e\t%i\t%i")


if __name__ == '__main__':
    main(**parse_args())
