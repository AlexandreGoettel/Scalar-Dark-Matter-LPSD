"""Calculate LPSD upper limits from max_lkl estimates."""
import os
import argparse
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy import constants

from ..LPSDIO import MaxLKLIO, get_A_star
from .stats import cumf_q_mu_scalar, kde_smoothing, RHO_LOCAL

PATH = os.path.abspath(os.path.split(os.path.split(__file__)[0])[0])


def parse_args():
    """Parse cmdl inputs, return as dict."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--max-lkl-path", type=str, required=True,
                        help="Path to max_lkl calculation output.")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Where to save output as .csv.")
    parser.add_argument("--alpha-CL", type=float, default=.95,
                        help="Confidence level to use for the upper limits.")
    parser.add_argument("--tol", type=float, default=1e-10,
                        help="'tol' to pass to scipy's Nelder-Mead minimizer for p-value.")
    parser.add_argument("--pruning", type=int, default=1,
                        help="Helpful during development and testing.")
    parser.add_argument("--plot-path", type=str, default="upper_limits.png",
                        help="Where to save the upper limit plot.")
    parser.add_argument("--kernel-size", type=int, default=int(1e5),
                        help="To use in KDE smoothing of results.")

    return vars(parser.parse_args())


def get_upper_limits_approx(sigma, alpha_CL=.95, tol=1e-10):
    """Calculate an asymptotic-formulae-based q_mu upper limit."""
    if np.isnan(sigma):
        return np.nan, np.nan

    def opt_pval(param, _sigma, n=1000):
        # Find the median of f_q_mu_0
        delta_q_mu = max(25, (param / _sigma) ** 2 + 1) / n
        q_mu = (param / _sigma) ** 2  # Initial guess
        F_q_mu_0 = cumf_q_mu_scalar(q_mu, param, 0, _sigma)
        if F_q_mu_0 < 0.5:  # Move upwards to find the median
            while F_q_mu_0 < 0.5:
                q_mu += delta_q_mu
                F_q_mu_0 = cumf_q_mu_scalar(q_mu, param, 0, _sigma)
        else:  # Move downwards
            while F_q_mu_0 > 0.5:
                q_mu -= delta_q_mu
                F_q_mu_0 = cumf_q_mu_scalar(q_mu, param, 0, _sigma)

        # Get p-value from f_q_mu_mu
        p_val = 1. - cumf_q_mu_scalar(q_mu, param, param, _sigma)
        return (p_val - (1 - alpha_CL))**2  # Significance target is alpha_CL!

    # Is there any way to solve this analytically?
    popt = minimize(opt_pval, sigma, args=(sigma,),
                    bounds=[(0, None)],
                    method="Nelder-Mead",
                    tol=tol
                    )
    assert popt.success

    # Convert mu back to Lamba_i^-1
    upper_Lambda = np.sqrt(popt.x)
    sigma_Lambda = sigma / (2. * upper_Lambda)
    return upper_Lambda, sigma_Lambda


def process_segment(_args):
    """Upper limit calculation wrapper for multiprocessing use."""
    # Use batching!
    freq_Hz, sigma, alpha_CL, tol = _args
    return freq_Hz, get_upper_limits_approx(sigma, alpha_CL=alpha_CL, tol=tol)


def main(max_lkl_path, output_file, pruning=1, alpha_CL=.95, tol=1e-10,
         plot_path="upper_limit.png", kernel_size=1e5):
    """Calculate an upper limit based on LPSD results."""
    max_lkl = MaxLKLIO(infile=max_lkl_path)

    # Idea: Since in the asymptotic limit, the UL only depends on sigma, use a look-up table!
    n = 1000
    x = np.geomspace(min(max_lkl.sigma), max(max_lkl.sigma), n)
    y, sigma = np.zeros_like(x), np.zeros_like(x)
    for i, xi in enumerate(tqdm(x, desc="Building look-up table")):
        y[i], sigma[i] = get_upper_limits_approx(xi, alpha_CL=alpha_CL, tol=tol)
    ul_table = interp1d(x, y, bounds_error=True)  # do geom?
    sigma_table = interp1d(x, sigma, bounds_error=True)

    result = np.zeros((len(max_lkl.frequency[::pruning]), 4))
    result[:, 0] = max_lkl.frequency[::pruning]
    # This can probably be batched if speed-up is required
    for i, sigma_val in enumerate(tqdm(max_lkl.sigma[::pruning], desc="Applying look-up tables")):
        result[i, 1:3] = ul_table(sigma_val), sigma_table(sigma_val)

    # Plot & save results
    plt.figure()
    ax = plt.subplot(111)
    kernel_size /= pruning

    # Comparison
    old = np.loadtxt(os.path.join(PATH, "data", "upper_limits_PRL.csv"))
    ax.plot(old[:, 0], old[:, 1], zorder=2, label="Göttel et al. 2024",
            linewidth=2, linestyle="-.")

    # TMP - Plot previous run results
    prefix = "/home/pczag4/work/cardiff/LPSD/LPSD_2025"
    # tmp = np.loadtxt(os.path.join(prefix, "O4a", "O4a_upper_limits.csv"),
    #                  delimiter="\t")
    # tmp_smooth = np.exp(kde_smoothing(np.log(tmp[:, 1]), kernel_size / 10))
    # ax.plot(tmp[:, 0], tmp_smooth, label="O4a-only", color="C4", zorder=1)
    # # Solo
    # tmp = np.loadtxt(os.path.join(prefix, "data", "upper_limits_o4b-only.csv"),
    #                  delimiter="\t")
    # tmp_smooth = np.exp(kde_smoothing(np.log(tmp[:, 1]), kernel_size / 10))
    # ax.plot(tmp[:, 0], tmp_smooth, label="O4b-only", color="C1", zorder=1)
    # tmp = np.load(os.path.join(prefix, "O4b", "upper_limits_o4ab.npy"))
    # mask = np.isnan(tmp[:, 2])
    # tmp = tmp[~mask, :]
    # tmp_smooth = np.exp(kde_smoothing(np.log(tmp[:, 1]), kernel_size / 10))
    # ax.plot(tmp[:, 0], tmp_smooth, label="O4a+b", color="C1", zorder=1)
    # HF
    tmp = np.loadtxt(os.path.join(prefix, "HF", "O4a_HF_upper_limits.csv"),
                     delimiter="\t")
    # tmp_smooth = np.exp(kde_smoothing(np.log(tmp[:, 1]), kernel_size / 10))
    ax.plot(tmp[:, 0], tmp[:, 1], label="O4a", color="C4", zorder=1)
    # tmp = np.loadtxt(os.path.join(prefix, "HF", "O4b_HF_upper_limits.csv"),
    #                  delimiter="\t")
    # ax.plot(tmp[:, 0], tmp[:, 1], label="O4b-only", color="C3", zorder=1)
    tmp = np.loadtxt(os.path.join(prefix, "HF", "O4ab_HF_upper_limits.csv"),
                     delimiter="\t")
    ax.plot(tmp[:, 0], tmp[:, 1], label="O4a+b", color="C3", zorder=1)
    # END TMP

    # Smoothing
    mask = np.isnan(result[:, 1])
    result = result[~mask]
    smooth_lim = np.exp(kde_smoothing(np.log(result[:, 1]), kernel_size))
    smooth_sigma = kde_smoothing(result[:, 2], kernel_size)
    ax.fill_between(result[:, 0], smooth_lim + smooth_sigma, smooth_lim - smooth_sigma,
                    alpha=.3, color="C2")
    ax.plot(result[:, 0], smooth_lim, label="O4a+b+c",
            linewidth=2, zorder=2, color="C2")

    # Plot GEO comparison
    geo_data = np.loadtxt(os.path.join(PATH, "data", "geo_limits.csv"), delimiter=",")
    ax.plot(geo_data[:, 0], geo_data[:, 1], label="GEO600",
            linewidth=2, color="C3", linestyle="--")

    # Calculate L1-strain equivalent
    f_A_star = get_A_star("/home/pczag4/work/cardiff/LPSD/LPSD_2025/transfer_functions/O4a_correction/")
    freq_Hz = result[:, 0]
    m_phi = freq_Hz * constants.h / constants.e * 1e-9  # GeV
    result[:, 3] = smooth_lim / f_A_star["L1"](freq_Hz) * np.sqrt(2 * RHO_LOCAL) / m_phi

    # Nice things
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$1/\Lambda_i$ (GeV)")
    ax.set_title("95% C.L. UPPER LIMITS")
    ax.grid(color="grey", linestyle="--", which="both")

    result[:, 1] = smooth_lim  # FIXME
    ax.set_xlim(1999, 5001)
    ax.set_ylim(4e-19, 5e-18)
    np.savetxt(output_file, result[::100, :], header="Frequency (Hz)\tUpper limit (GeV^-1)\tUncertainty (GeV^-1)\tL1-strain-equivalent", fmt="%.6e", delimiter="\t")

    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plot_path)


if __name__ == '__main__':
    args = parse_args()
    _max_lkl_path, _output_file = args.pop("max_lkl_path"), args.pop("output_file")
    main(_max_lkl_path, _output_file, **args)
