"""Run q0 calculation - condor version."""
import os
import argparse
from multiprocessing import Pool
from tqdm import tqdm
from scipy.optimize import minimize
from scipy import constants
import numpy as np
# Project imports
from . import stats
from ..LPSDIO import LPSDDataGroup, LPSDJSONIO, get_A_star, MaxLKLIO
from .fit_background import bkg_model


def parse_args():
    """Define cmdl. arg. parser and return vars."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to lpsd output file or folder of files.")
    parser.add_argument("--data-prefix", type=str, default="",
                        help="If data-path is a dir, only consider files starting with prefix.")
    parser.add_argument("--peak-shape-file", type=str, default=None,
                        help="Path to the peak shape numpy file. Ignore to use single-bin.")
    parser.add_argument("--bkg-info-path", type=str, required=True,
                        help="Output data from fit_background.py.")
    parser.add_argument("--tf-path", type=str, required=True,
                        help="Path to directory holding transfer functions.")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to file (ending in .h5 or .hdf5) to store the results.")
    parser.add_argument("--iteration", type=int, default=0,
                        help="Condor iteration number")
    parser.add_argument("--n-frequencies", type=int, default=-1,
                        help="If positive, number of frequencies to run over (else all freqs go).")
    parser.add_argument("--ana-fmin", type=float, default=10,
                        help="Minimum analysis frequency.")
    parser.add_argument("--ana-fmax", type=float, default=5000,
                        help="Maximum analysis frequency.")
    parser.add_argument("--max-chi-sqr", type=float, default=10.,
                        help="Maximum chi^2/dof value for skew-norm fits.")
    parser.add_argument("--n-processes", type=int, default=1,
                        help="If > 1, use multiprocessing.")
    parser.add_argument("--min-log10mu", type=int, default=-40,
                        help="Min of the log10mu values to investigate.")
    parser.add_argument("--max-log10mu", type=int, default=-32,
                        help="Max of the log10mu values to investigate.")

    return vars(parser.parse_args())


def find_leading_trailing_invalid(arr):
    """Find the indices of the leading and trailing groups of invalid numbers in an array."""
    # Create a boolean mask where True indicates either NaN or Inf
    mask = np.isnan(arr) | np.isinf(arr)
    if not np.sum(~mask):
        return 0, 0

    # Find the last index of the leading group of invalid numbers
    if mask[0]:
        for i, val in enumerate(mask):
            if not val:
                first_idx = i - 1
                break
    else:
        first_idx = 0

    # Find the first index of the trailing group of invalid numbers
    if mask[-1]:
        for i, val in enumerate(mask[::-1]):
            if not val:
                last_idx = len(arr) - 1 - (i - 1)
                break
    else:
        last_idx = len(arr) - 1

    return first_idx + 1, last_idx - 1


def get_valid_mu_ranges(Y, bkg, peak_norm, model_args,
                        start=-40, end=-30, n=100):
    """Investigate where mu is valid for an easier later analysis."""
    mu = np.logspace(start, end, n)
    output = []
    def get_mu_ranges(test_mus):
        valid_mu_ranges = np.zeros((len(Y), 2))
        for i, logPSD in enumerate(Y):
            log_term = np.exp(bkg[i]) + peak_norm[i]*test_mus[:, None]
            log_term[log_term <= 0] = np.nan  # Protect against NaNs
            residuals = logPSD - np.log(log_term)

            lkl_values = stats.pdf_skewnorm(residuals, 1, *model_args[i])
            lkl_is_pos = lkl_values > 0
            log_lkl_values = np.full_like(lkl_values, -np.inf)
            log_lkl_values[lkl_is_pos] = np.log(lkl_values[lkl_is_pos])

            log_lkl_values = np.sum(log_lkl_values, axis=1)
            start, end = find_leading_trailing_invalid(log_lkl_values)

            if not np.isnan(log_lkl_values[0]) and not np.isinf(log_lkl_values[0]) and start:
                start -= 1
            if np.sum(np.isnan(log_lkl_values[start:end])) or np.sum(np.isinf(log_lkl_values[start:end])):
                tqdm.write("[ERROR] RANGE HAS NANs/INFs")
                tqdm.write(f"{i}, {log_lkl_values[start:end]}")
            assert not np.sum(np.isnan(log_lkl_values[start:end])) and not np.sum(np.isinf(log_lkl_values[start:end]))

            valid_mu_ranges[i, :] = test_mus[start], test_mus[end]

        return valid_mu_ranges

    # Positive values
    valid_mu_ranges_pos = get_mu_ranges(mu)
    valid_mu_range_pos = np.max(valid_mu_ranges_pos[:, 0]), np.min(valid_mu_ranges_pos[:, 1])
    if valid_mu_range_pos[1] - valid_mu_range_pos[0] > 0:
        output.append(valid_mu_range_pos)
    else:
        output.append(None)

    # Negative values
    valid_mu_ranges_neg = get_mu_ranges(-mu[::-1])
    valid_mu_range_neg = np.max(valid_mu_ranges_neg[:, 0]), np.min(valid_mu_ranges_neg[:, 1])
    if valid_mu_range_neg[1] - valid_mu_range_neg[0] > 0:
        output.append(valid_mu_range_neg)
    else:
        output.append(None)

    return output


def calc_max_lkl(Y, bkg, model_args, peak_norm, peak_shape,
                 min_log10mu=-45, max_log10mu=-32, n_seed_lkl=1000):
    """Calculate maximum likelihood information."""
    # Question: do I calculate sigma here as well? Might as well no?
    def log_lkl_shape(params):
        return stats.log_likelihood(params[0], Y, bkg, peak_norm, peak_shape, model_args)

    # Get initial guess for mu, based on lkl
    valid_mu_ranges = get_valid_mu_ranges(Y, bkg, peak_norm, model_args,
                                          start=min_log10mu, end=max_log10mu, n=n_seed_lkl)
    if valid_mu_ranges == [None, None]:
        tqdm.write("RANGE IS NONE")
        return np.nan, np.nan, np.nan, np.nan
    max_test_lkl, initial_guess = np.inf, 0
    for mu_range in valid_mu_ranges:
        if mu_range is None:
            continue
        test_mus = np.sign(mu_range[0])*np.logspace(np.log10(abs(min(mu_range))),
                                                    np.log10(abs(max(mu_range))),
                                                    n_seed_lkl)
        test_lkl = np.array([-log_lkl_shape([mu]) for mu in test_mus])
        mask = np.isnan(test_lkl) | np.isinf(test_lkl)
        assert not np.sum(mask)  # FIXME
        _test_lkl = min(test_lkl[~mask])
        if _test_lkl < max_test_lkl:
            initial_guess = test_mus[np.argmin(test_lkl[~mask])]
            max_test_lkl = _test_lkl

    # Calculate max lkl
    bounds = [(None, 0)] if initial_guess < 0 else [(0, None)]
    popt_peak_shape = minimize(
        lambda x: -log_lkl_shape(x),
        [initial_guess],
        bounds=bounds,
        method="Nelder-Mead",
        tol=1e-10)
    # Get mu, max lkl, and zero lkl
    zero_lkl = log_lkl_shape([0])
    max_lkl = -popt_peak_shape.fun
    mu = popt_peak_shape.x[0]

    # Calculate uncertainty on mu
    try:
        if mu == 0:  # This either means railing, where Fisher is invalid, or failed optimisation
            raise ValueError("mu railing to 0")
        elif mu > 0:
            if valid_mu_ranges[0] is None:  # Just in case, should not happen
                raise ValueError
            max_dist = min(mu - valid_mu_ranges[0][0], valid_mu_ranges[0][1] - mu)
        else:
            if valid_mu_ranges[1] is None:  # Just in case, should not happen
                raise ValueError
            max_dist = min(mu - valid_mu_ranges[1][0], valid_mu_ranges[1][1] - mu)
    except ValueError as err:
        tqdm.write(f"{err}")
        return np.nan, np.nan, np.nan, np.nan

    # Derive uncertainty from Fisher matrix
    try:
        sigma = stats.sigma_at_point(lambda x: log_lkl_shape([x]),
                                     mu,
                                     initial_dx=min(max_dist/2., abs(mu)),
                                     tolerance=1e-4)
        if np.isnan(sigma):
            raise AssertionError("Sigma is NaN")
    except (ValueError, AssertionError) as err:
        tqdm.write(f"[ERROR] Sigma calculation: {err}")
        return zero_lkl, max_lkl, mu, np.nan
    return zero_lkl, max_lkl, mu, sigma


def process_q0(args):
    """Wrap calculate_q0 for multiprocessing."""
    Y, bkg, model_args, peak_norm, peak_shape, fmin, kwargs = args
    try:
        zero_lkl, max_lkl, mu, sigma = calc_max_lkl(
            Y, bkg, model_args, peak_norm, peak_shape, **kwargs)
    except AssertionError as err:
        tqdm.write(f"ASSERTION ERROR {err}")
        zero_lkl, max_lkl, mu, sigma = np.nan, np.nan, np.nan, np.nan
    return fmin, zero_lkl, max_lkl, mu, sigma


def make_args(datagroup, bkg_data, f_A_star, iteration, n_frequencies, ana_fmin, ana_fmax,
              peak_shape, max_chi_sqr, min_log10mu, max_log10mu):
    """Generator to prep. parallel q0 calculation."""
    idx_end = np.where(datagroup.freq >= ana_fmax)[0]
    if len(idx_end) == 0:  # If ana_fmax is outside of the datagroup
        idx_end = len(datagroup)
    else:
        idx_end = idx_end[0]
    if n_frequencies < 0:
        idx_start = np.where(datagroup.freq >= ana_fmin)[0][0]
    else:
        idx_start = iteration * n_frequencies
        assert idx_start < idx_end
        idx_end = (iteration + 1) * n_frequencies
    frequencies = datagroup.freq[idx_start:idx_end]

    peak_size = len(peak_shape)
    bkg_info = {k: bkg_data.get_df(k) for k in bkg_data.data}
    df_columns = ["x_knots", "y_knots", "alpha_skew", "loc_skew", "sigma_skew", "chi_sqr"]
    lkl_kwargs = {"min_log10mu": min_log10mu, "max_log10mu": max_log10mu}
    for i, test_freq in enumerate(frequencies):
        freq_idx = idx_start + i
        # Skip this entry if test_freq is out of bounds
        if freq_idx + peak_size > len(datagroup.freq):
            continue

        raw_Y = datagroup.logPSD[:, freq_idx:freq_idx + peak_size]
        Y = []

        freq_Hz = datagroup.freq[freq_idx:freq_idx + peak_size]
        bkg, model_args, peak_norm = [], [], []
        for j, (t0, t1, ifo) in enumerate(zip(datagroup.metadata["t0"],
                                              datagroup.metadata["t1"],
                                              datagroup.metadata["ifo"])):
            # Skip this entry if there was a NaN in the data
            if any(np.isnan(raw_Y[j])):
                continue
            label = bkg_data.get_label(t0=t0, t1=t1, ifo=ifo)
            df = bkg_info[label]
            mask = (df['fmin'] <= test_freq) & (df['fmax'] >= test_freq)
            if len(df[mask]) == 0:
                continue  # test_freq is not in data
            x_knots, y_knots, alpha_skew, loc_skew, sigma_skew, chi_sqr\
                = df[mask][df_columns].iloc[0]
            # Skip this entry if the fit is bad
            if chi_sqr >= max_chi_sqr:
                continue

            _bkg = bkg_model(np.log(freq_Hz), np.concatenate([x_knots, y_knots]))
            _model_args = [loc_skew, sigma_skew, alpha_skew]
            _beta = stats.get_beta(freq_Hz, f_A_star[ifo])

            # Fill arg arrays
            bkg.append(_bkg)
            model_args.append(_model_args)
            peak_norm.append(_beta)
            Y.append(raw_Y[j])
        if len(bkg) == 0:
            continue

        Y, bkg, peak_norm, model_args = list(map(np.array, [Y, bkg, peak_norm, model_args]))
        yield Y, bkg, model_args, peak_norm, peak_shape, test_freq, lkl_kwargs


def main(data_path=None, output_path=None, tf_path=None, peak_shape_file=None, data_prefix="",
         bkg_info_path=None, iteration=0, n_frequencies=1, ana_fmin=10, ana_fmax=5000,
         min_log10mu=-40, max_log10mu=-32, n_processes=1, max_chi_sqr=10):
    """Run q0 calculation for selected frequencies."""
    # TODO Calibration
    datagroup = LPSDDataGroup(data_path, data_prefix)
    if not os.path.exists(datagroup.buffer_path):
        datagroup.save_data()

    if peak_shape_file == "None":
        peak_shape = np.ones(1)
    else:
        peak_shape = np.load(peak_shape_file)
    if np.sum(peak_shape) != 1.:  # float compare should be ok for this
        raise ValueError("Peak_shape must contain an array that sums to 1.")

    f_A_star = get_A_star(tf_path)
    bkg_data = LPSDJSONIO(bkg_info_path)
    args = make_args(datagroup, bkg_data, f_A_star, iteration, n_frequencies, ana_fmin,
                     ana_fmax, peak_shape, max_chi_sqr, min_log10mu, max_log10mu)

    results = []
    with Pool(n_processes, maxtasksperchild=100) as pool,\
            tqdm(total=n_frequencies, position=0,
                 desc="Maximum likelihood", leave=True) as pbar:
        for result in pool.imap(process_q0, args):
            results.append(result)
            pbar.update(1)

    # Merge results
    q0_data = np.zeros((len(results), 5))
    for i, result in enumerate(results):
        q0_data[i, :] = result  # freqs, zero_lkl, max_lkl, mu, sigma
    outdir, outname = os.path.split(output_path)
    output = MaxLKLIO(data=q0_data)
    output.save(os.path.join(outdir, f"{os.path.splitext(outname)[0]}_{iteration}_result.npy"))


if __name__ == '__main__':
    main(**parse_args())
