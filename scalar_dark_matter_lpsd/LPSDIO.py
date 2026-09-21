"""Communicate to and from disk for LPSD-relevant variables and results.

Copyright (C) 2026 Alexandre Goettel

License: GPLv3 or later <https://www.gnu.org/licenses/gpl-3.0.html>
Contact: Alexandre Goettel <alexandre.goettel@nottingham.ac.uk>
"""
import os
import glob
import csv
import json
from io import StringIO
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
# Project imports
from .utils import LPSDVars


def get_A_star(tf_path):
    """Return dict of interpolant to the transfer functions for each ifo."""
    f_A_star = {}
    for ifo, nice_ifo in zip(["H1", "L1"], ["LHO", "LLO"]):
        transfer_function = pd.read_csv(
            os.path.join(tf_path, f"Amp_Cal_{nice_ifo}.txt"),
            delimiter="\t",
            header=0,
            names=("Frequency", "Amplitude"),
            index_col=False,
        )
        f_A_star[ifo] = interp1d(transfer_function["Frequency"], transfer_function["Amplitude"])
    return f_A_star


def sep_label(filepath):
    """Extract timestamp and ifo from LPSD output filepath."""
    t0, t1, ifo = os.path.splitext(os.path.split(filepath)[-1])[0].split("_")[-3:]
    return t0, t1, ifo


class LPSDJSONIO:
    """Read/Write from JSON files."""

    def __init__(self, filename):
        self.filename = filename
        self.data = None
        self.update_data()

    def update_data(self):
        """Read data contents from JSON file."""
        if os.path.exists(self.filename):
            with open(self.filename, "r") as _file:
                self.data = json.load(_file)
        else:
            try:
                assert isinstance(self.filename, str)
                with open(self.filename, "w") as _f:
                    pass
                os.remove(self.filename)
            except (OSError, IOError, AssertionError):
                raise IOError(f"'{self.filename} is not a valid filename!")
            self.data = {}

    def update_file(self, name, df, orient="records"):
        """Write data to the JSON file, give it reference "name"."""
        self.data[self.get_label(name)] = df.to_json(orient=orient)
        with open(self.filename, "w") as _file:
            json.dump(self.data, _file)
        self.update_data()

    def get_df(self, name):
        """Get a dataframe labelled "name" from the JSON file."""
        if not os.path.exists(self.filename):
            raise IOError(f"File '{self.filename}' does not exist..")

        json_df = self.data.get(name)
        if json_df is None:
            raise IOError(f"'{name} is not in '{self.filename}..")

        # Convert the JSON object back to DataFrame
        return pd.read_json(StringIO(json_df))

    @staticmethod
    def get_label(filepath=None, t0=None, t1=None, ifo=None, prefix="bkginfo"):
        """Get simple df label from filepath."""
        if filepath is None:
            assert all([x is not None for x in [t0, t1, ifo]])
            return f"{prefix}_{t0}_{t1}_{ifo}"
        else:
            main, _ = os.path.splitext(os.path.split(filepath)[-1])
            body = "_".join(main.split("_")[-3:])
            return f"{prefix}_{body}"


class LPSDDataGroup:
    """Gather & generalise several LPSD output files."""

    def __init__(self, data_path, data_prefix="", buffer_path=None):
        """Read data found in data_path to numpy format."""
        self.data_path = data_path
        self.data_prefix = data_prefix
        if buffer_path is None:
            self.buffer_path = self.create_buffer_path(data_path)
        else:
            self.buffer_path = buffer_path

        if os.path.exists(self.buffer_path):
            files = []
        else:
            if os.path.isdir(data_path):
                files = list(glob.glob(os.path.join(data_path, f"{data_prefix}*")))
            elif os.path.exists(data_path):
                files = [data_path]
            else:
                raise IOError(f"Invalid path: '{data_path}'.")

        self.freq, self.logPSD, self.metadata = self.read_all_data(files)

    def __len__(self):
        return len(self.freq)

    def create_buffer_path(self, data_path, suffix=".h5"):
        """Create path based on data_path to store interim HDF data for fast retrieval."""
        loc, body = os.path.split(data_path)
        (main, _) = os.path.splitext(body)
        return os.path.join(loc, f"buffer_{main}{suffix}")

    def read_all_data(self, files):
        """Read from all files in 'files' and combine."""
        files = sorted(files)  # For reproducibility

        if os.path.exists(self.buffer_path):
            print("[INFO] Buffer exists, ignoring raw data files..")
            with h5py.File(self.buffer_path, "r") as buffer:
                freq = np.array(buffer["freq"])
                logPSD = np.array(buffer["logPSD"])
                metadata = dict(buffer["logPSD"].attrs)
            metadata = self.convert_metadata_Nones(metadata)
            return freq, logPSD, metadata

        # Make sure that all files are compatible
        data = []
        for f in tqdm(files):
            data.append(LPSDOutput(f))
        _len = len(data[0])

        # Get metadata
        metadata = {"t0": [], "t1": [], "ifo": []}
        for x in data[0].kwargs.keys():
            metadata[x] = []
        for i, filename in enumerate(files):
            t0, t1, ifo = sep_label(filename)
            metadata["t0"].append(int(t0))
            metadata["t1"].append(int(t1))
            metadata["ifo"].append(ifo)
            for k, v in data[i].kwargs.items():
                metadata[k].append(v)

        # Combine to numpy array
        freq = data[0].freq
        output = np.zeros((len(data), _len), dtype=np.float64)
        for i, row in enumerate(data):
            output[i, :] = row.logPSD
        return freq, output, metadata

    def save_data(self, overwrite=False):
        """Save gathered data to HDF format."""
        if not overwrite and os.path.exists(self.buffer_path):
            return
        print(f"[INFO] Saving buffer to {self.buffer_path}..")
        self.metadata = self.convert_metadata_Nones(self.metadata)
        with h5py.File(self.buffer_path, "w") as outfile:
            outfile.create_dataset("freq", data=self.freq)
            outfile.create_dataset("logPSD", data=self.logPSD)
            outfile["logPSD"].attrs.update(self.metadata)

    def convert_metadata_Nones(self, metadata):
        """Convert -1 to None or viceversa."""
        for k, v in metadata.items():
            metadata[k] = [None if x == -1 else (-1 if x is None else x) for x in v]
        return metadata


class LPSDData:
    """Hold & manage LPSD output data."""

    def __init__(self, logPSD, freq=None, **kwargs):
        assert all([flag in kwargs for flag in ["fmin", "fmax", "fs", "Jdes"]])
        self.kwargs = kwargs
        self.logPSD = logPSD
        if freq is None:
            self.freq = self.freq_from_kwargs()
        # Get full LPSD vars
        if "resolution" not in kwargs:
            kwargs["resolution"] = np.exp((np.log(kwargs["fmax"]) - np.log(kwargs["fmin"])) /\
                (kwargs["Jdes"] - 1)) - 1.
        if "epsilon" not in kwargs:
            kwargs["epsilon"] = None
        self.vars = LPSDVars(*map(lambda x: kwargs[x],
                                  ["fmin", "fmax", "resolution", "fs", "epsilon"]))

    def __len__(self):
        return len(self.logPSD)

    def freq_from_kwargs(self):
        """Derive frequency from kwargs, can be more precise than reading from file."""
        return np.logspace(np.log10(self.kwargs["fmin"]),
                           np.log10(self.kwargs["fmax"]),
                           int(self.kwargs["Jdes"])
                           )

    def save_to_hdf5(self, filename, dset="logPSD", dset_freq="frequency"):
        """Save (log)PSD data to HDF5."""
        tqdm.write(f"Writing data to: '{filename}'..")
        with h5py.File(filename, "w") as _file:
            _file.create_dataset(dset_freq, data=self.freq)
            _file.create_dataset(dset, data=self.logPSD)
            _file[dset].attrs.update(self.kwargs)


class LPSDOutput(LPSDData):
    """Extend LPSDData with functionality to read from an output file."""

    def __init__(self, filename, delimiter="\t"):
        self.filename = filename
        is_hdf5 = filename.endswith(".h5") or filename.endswith(".hdf5")
        self.kwargs = self.get_lpsd_kwargs_hdf5() if is_hdf5 else self.get_lpsd_kwargs()
        # Check that file is valid
        assert self.kwargs
        if "resolution" not in self.kwargs:
            self.kwargs["resolution"] = np.exp((np.log(self.kwargs["fmax"]) - np.log(self.kwargs["fmin"])) /\
                (self.kwargs["Jdes"] - 1)) - 1.

        # Get logPSD data from file
        if is_hdf5:
            self.freq, self.logPSD = self.read_hdf5()
        else:
            raw_freq, psd = self.read(raw_freq=True, delimiter=delimiter)
            # Protection against (old) LPSD bug
            self.logPSD = np.log(psd[:-1]) if psd[-1] == 0 else np.log(psd)
            Jdes = int(self.kwargs["Jdes"])
            if len(self.logPSD) > Jdes:
                raise ValueError("The input file '{filename}' contains more lines than Jdes!")
            else:
                self.freq = self.freq_from_kwargs()
            if len(self.logPSD) < Jdes:
                # Buffer missing frequencies with NaNs
                tolerance = self.kwargs["resolution"] * 0.1
                filled_raw_freq = np.full_like(self.freq, np.nan)
                filled_logPSD = np.full_like(self.freq, np.nan)
                j = 0
                for i, val in enumerate(self.freq):
                    if np.abs(val - raw_freq[j]) <= val * tolerance:
                        filled_raw_freq[i] = val
                        filled_logPSD[i] = self.logPSD[j]
                        j += 1
                        if j == len(raw_freq):
                            break
                self.freq = filled_raw_freq
                self.logPSD = filled_logPSD

        super().__init__(self.logPSD, freq=self.freq, **self.kwargs)

    def get_lpsd_kwargs_hdf5(self, dset="logPSD"):
        """Extract LPSD parameters from a lpsd .h5 file."""
        with h5py.File(self.filename, "r") as _f:
            return dict(_f[dset].attrs)

    def get_lpsd_kwargs(self):
        """Extract LPSD parameters from an lpsd output file."""
        flags = {
            "fmax": "t",
            "fmin": "s",
            "fs": "f",
            "Jdes": "J",
            "batch": "n",
            "epsilon": "E"
        }
        with open(self.filename, "r", encoding="utf-8") as _f:
            for line in _f:
                if "Command line" in line:
                    line_flags = line.split()
                    values = [(float(line_flags[line_flags.index(f"-{flag}") + 1])
                            if f"-{flag}" in line_flags
                            else None)
                            for flag in flags.values()]
                    return dict(zip(flags.keys(), values))
        raise IOError("Invalid output file.")

    def read_hdf5(self, freq_dset="frequency", psd_dset="logPSD"):
        """Read freq_dset and psd_dset from datafile."""
        with h5py.File(self.filename, "r") as _f:
            x, y = _f[freq_dset][()], _f[psd_dset][()]
        return np.array(x), np.array(y)

    def read(self, dtype=np.float64, delimiter="\t", raw_freq=False, idx=1):
        """
        Read an output file from LPSD.

        return: frequency & PSD arrays.
        """
        x, y = [], []
        with open(self.filename, "r") as _file:
            data = csv.reader(_file, delimiter=delimiter)
            for row in tqdm(data, total=int(self.kwargs["Jdes"]), desc="Reading LPSD", leave=False):
                try:
                    if raw_freq:
                        x += [float(row[0])]
                    y += [float(row[idx])]
                except (ValueError, IndexError):
                    continue
        return np.array(x, dtype=dtype), np.array(y, dtype=dtype)


class MaxLKLIO:
    """Buffer data between max_lkl_executable and downstream uses."""

    def __init__(self, data=None, infile=None):
        self.data = None
        if data is not None:
            self.data = data
            assert self.is_compatible(self.data)
        elif infile is not None:
            self.load(infile)
        else:
            raise ValueError("'data' or 'infile' must be provided as kwargs.")

    def __len__(self):
        return self.data.shape[0]

    def __getattr__(self, name):
        property_map = {"frequency": 0, "zero_lkl": 1, "max_lkl": 2, "mu": 3, "sigma": 4}
        if name in property_map:
            return self.data[:, property_map[name]]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def is_compatible(self, data):
        """Make sure the file is compatible."""
        return len(data.shape) == 2 and data.shape[1] == 5

    def save(self, outfile):
        """Save as .npy."""
        np.save(outfile, self.data)

    def load(self, infile):
        """Load from .npy."""
        self.data = np.load(infile)
        assert self.is_compatible(self.data)
