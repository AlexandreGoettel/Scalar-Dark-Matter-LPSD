"""Implement classes for input and otuput."""
import os
from string import Template
import numpy as np
import utils


class Reader:
    """Read from LPSD and sensitivity/ output files."""
    pass


def write_lpsd_sh(output_file=None, length=3600, sampling_frequency=16384,
                  sh_filename=None, path_to_lpsd_exec=None,
                  fmin=10, fmax=8192, resolution=1e-6, **_):
    """Save relevant parameters and write .sh file to run LPSD."""
    # Make sure relevant vars have been parsed
    assert path_to_lpsd_exec != "" and output_file != "" and sh_filename != ""

    # Take care of abs. vs rel. paths.
    if not output_file.startswith("/"):
        output_file = os.path.join(os.getcwd(), output_file)

    # What fmin can we support with data length?
    fmin_from_length = int(.5 + utils.get_fmin(length, resolution))
    if fmin_from_length > fmax:
        raise ValueError("Cannot find valid configuration with this resolution and segment length!")

    if fmin_from_length > fmin:
        print(f"Warning: using fmin={fmin_from_length}Hz instead of supplied {fmin}Hz" +
                " because data length is insufficient.")
        fmin = fmin_from_length

    number_of_frequencies = utils.get_Jdes(fmin, fmax, resolution)
    # Write .sh file
    file_contents = {
            "filename": output_file,
            "TSlength": length,
            "fmin": fmin,
            "fmax": fmax,
            "Jdes": number_of_frequencies,
            "fsamp": sampling_frequency,
            "output_filename": f"auto_{length}s_{np.log10(resolution):.1f}res.lpsd",
            "LPSD_PATH": path_to_lpsd_exec
        }
    PATH_prefix = os.path.split(os.path.abspath(__file__))[0]
    with open(os.path.join(PATH_prefix, "lpsd_template.sh"), "r") as _f:
        _src = _f.read()

    with open(sh_filename, "w") as _f:
        _f.write(Template(_src).safe_substitute(file_contents))
