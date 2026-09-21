import os
import argparse
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
from scipy import constants
from scipy.optimize import minimize
from multiprocessing import Pool

from . import stats
from ..LPSDIO import MaxLKLIO, LPSDDataGroup, LPSDJSONIO, get_A_star
from .fit_background import bkg_model


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--max-lkl-path", type=str, required=True,
                        help="Path to the maximum likelihood output file.")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the directory containing lpsd output files.")
    parser.add_argument("--data-prefix", type=str, default='',
                        help="Prefix to find the lpsd output files.")
    parser.add_argument("--bkg-path", type=str, required=True,
                        help="Path to the background fits output file.")
    parser.add_argument("--tf-path", type=str, required=True,
                        help="Path to the folder containing transfer functions.")
    parser.add_argument("--buffer-path", type=str, default=None,
                        help="Path to the buffer directory.")
    parser.add_argument("--out-folder", type=str, required=True,
                        help="Output folder.")
    parser.add_argument("--sigma-limit", type=float, default=5,
                        help="Threshold for candidate selection.")
    parser.add_argument("--sigma-limit-consistency", type=float, default=5)
    parser.add_argument("--n-processes", type=int, default=1)

    return vars(parser.parse_args())


class DataManager:
    """Help organise LPSD data manipulation."""

    def __init__(self, data_path, bkg_path, tf_path, data_prefix="", buffer_path=None):
        self.data = LPSDDataGroup(data_path, data_prefix=data_prefix, buffer_path=buffer_path)
        self.bkg = LPSDJSONIO(bkg_path)
        self.f_A_star = get_A_star(tf_path)

        self.peak_size = 1
        self.max_chi_sqr = 10

    def get_data_around_candidate(self, freq, buffer=50):
        """Get relevant data around freq, with some buffer frequency bins."""
        # TODO: if a freq is on a boundary it'll be tricky
        freq_idx = np.argmin(np.abs(self.data.freq - freq))
        raw_Y = self.data.logPSD[:, freq_idx - buffer:freq_idx + self.peak_size + buffer]
        freq_Hz = self.data.freq[freq_idx - buffer:freq_idx + self.peak_size + buffer]

        bkg_info = {k: self.bkg.get_df(k) for k in self.bkg.data}
        Y, bkg, model_args, peak_norm, ifos = [], [], [], [], []
        for j, (t0, t1, ifo) in enumerate(zip(self.data.metadata["t0"],
                                              self.data.metadata["t1"],
                                              self.data.metadata["ifo"])):
            if any(np.isnan(raw_Y[j])):
                continue
            label = self.bkg.get_label(t0=t0, t1=t1, ifo=ifo)
            df = bkg_info[label]
            mask = (df['fmin'] <= freq) & (df['fmax'] >= freq)
            _, _, x_knots, y_knots, alpha_skew, loc_skew, sigma_skew, chi_sqr\
                = df[mask].iloc[0]
            # Skip this entry if the fit is bad
            if chi_sqr >= self.max_chi_sqr:
                continue

            _bkg = bkg_model(np.log(freq_Hz), np.concatenate([x_knots, y_knots]))
            _model_args = [loc_skew, sigma_skew, alpha_skew]
            _beta = stats.get_beta(freq_Hz, self.f_A_star[ifo])

            # Fill arg arrays
            bkg.append(_bkg)
            model_args.append(_model_args)
            peak_norm.append(_beta)
            Y.append(raw_Y[j])
            ifos.append(ifo)

        Y, bkg, model_args, peak_norm, freq_Hz = list(
            map(np.array, [Y, bkg, model_args, peak_norm, freq_Hz]))
        return Y, bkg, model_args, peak_norm, ifos, freq_Hz

    def plot_candidate(self, freq, mu=None, buffer=50):
        """Candidate-style double-ifo plot around freq."""
        Y, bkg, _, peak_norm, ifos, freq_Hz =\
            self.get_data_around_candidate(freq, buffer=buffer)

        # Prepare the plots
        fig = plt.figure()
        gs = GridSpec(1, 2, figure=fig)
        axH1 = fig.add_subplot(gs[0, 0])  # First row, first column
        axL1 = fig.add_subplot(gs[0, 1], sharex=axH1, sharey=axH1)

        # Plot the PSDs
        for i in range(Y.shape[0]):
            if ifos[i] == "":
                continue
            ax = axH1 if ifos[i] == "H1" else axL1
            ax.plot(freq_Hz, np.sqrt(np.exp(Y[i, :])), alpha=.75, zorder=1)
            ax.plot(freq_Hz, np.sqrt(np.exp(bkg[i, :])), alpha=.5, color="C0", zorder=0)

        # Find the first H1 and L1 peak norm terms
        peak_norm_H1 = peak_norm[ifos.index("H1"), :] if "H1" in ifos else 0
        peak_norm_L1 = peak_norm[ifos.index("L1"), :] if "L1" in ifos else 0

        for ifo, ax, peak_norm_ifo in zip(["H1", "L1"], [axH1, axL1], [peak_norm_H1, peak_norm_L1]):
            if mu is not None:
                ax.scatter([freq_Hz[buffer]], [np.sqrt(mu ** 2 * peak_norm_ifo[buffer])],
                        color="k", zorder=2)

            # Nice things
            ax.set_title(ifo + r", $\Lambda_i^{-1}$:" +
                            f" {mu:.1e}")
            ax.set_xlabel("Frequency (Hz)")
            ax.grid(linestyle="--", linewidth=1, color="grey", alpha=.33)
        axL1.set_ylabel("ASD")
        axL1.set_yticklabels([])
        axH1.set_yscale("log")


def cluster(mask, mus):
    """Get highest mu in each cluster, return each max's index and height."""
    index, in_cluster, peak = 0, mask[0], mus[0]
    indices, peaks = [], []
    for i, (flag, mu) in enumerate(zip(mask[1:], mus[1:])):
        if in_cluster:
            if flag and mu > peak:
                index, peak = i + 1, mu
            elif not flag:
                in_cluster = False
                indices.append(index)
                peaks.append(peak)
        elif flag:
            in_cluster = True
            index, peak = i + 1, mu

    return np.array(indices), np.sqrt(np.array(peaks))


def analyze_per_ifo(Y, bkg, peak_norm, ifos, model_args):
    """Reconstruct mu in both ifos (with uncertainty)."""
    isH1 = np.array([ifo == "H1" for ifo in ifos], dtype=bool)
    idx = (Y.shape[1] - 1) // 2

    q0s, mus, sigmas = [], [], []
    for mask in isH1, ~isH1:
        _Y, _bkg, _beta = list(map(lambda x: x[mask, idx][:, None],
                                   [Y, bkg, peak_norm]))
        if len(_Y) == 0:
            return [], [], []
        _model_args = model_args[mask, :]

        # Maximise the likelihood
        def log_lkl(mu):
            return stats.log_likelihood(mu, _Y, _bkg, _beta, np.ones(1), _model_args)

        test_mus = np.logspace(-40, -32, 100)
        test_lkl_pos = np.array([-log_lkl(mu) for mu in test_mus])
        test_lkl_neg = np.array([-log_lkl(-mu) for mu in test_mus])
        test_lkl_pos[np.isnan(test_lkl_pos)] = np.inf
        test_lkl_neg[np.isnan(test_lkl_neg)] = np.inf
        min_lkl_pos, min_lkl_neg = np.min(test_lkl_pos), np.min(test_lkl_neg)

        if min_lkl_pos < min_lkl_neg:
            mu_hat_init = test_mus[np.argmin(test_lkl_pos)]
        else:
            mu_hat_init = -test_mus[np.argmin(test_lkl_neg)]

        popt = minimize(
            lambda x: -log_lkl(x),
            mu_hat_init,
            method="Nelder-Mead",
            tol=1e-10
        )
        zero_lkl = log_lkl(0)
        mu_hat, max_lkl = popt.x[0], -popt.fun

        # Get uncertainty on mu
        try:
            sigma = stats.sigma_at_point(log_lkl,
                                        mu_hat,
                                        initial_dx=abs(mu_hat)/10,
                                        tolerance=1e-4)
        except ValueError:
            tqdm.write("Failed to converge on sigma")
            continue

        q0s.append(-2 * (zero_lkl - max_lkl))
        mus.append(mu_hat)
        sigmas.append(sigma)
    return q0s, mus, sigmas


def process_candidate(args):
    """Process a single candidate for parallel execution."""
    frequency, lambda_c = args
    Y, bkg, model_args, peak_norm, ifos, _ = global_mngr.get_data_around_candidate(frequency)
    if any(i == 0 for i in Y.shape):
        return None
    q0s, mus, sigmas = analyze_per_ifo(Y, bkg, peak_norm, ifos, model_args)
    if len(q0s) < 2:
        return None

    # Compare reconstructed mus
    t_value = abs(mus[1] - mus[0]) / np.sqrt(sigmas[1]**2 + sigmas[0]**2)
    if t_value >= global_sigma_limit:
        return None
    return (frequency, lambda_c, q0s, mus, sigmas)


def main(max_lkl_path, data_path, bkg_path, tf_path,
         data_prefix, buffer_path, out_folder,
         sigma_limit=5, sigma_limit_consistency=5, n_processes=1):
    global global_mngr, global_sigma_limit
    
    lkl = MaxLKLIO(infile=max_lkl_path)
    q0 = np.zeros_like(lkl.zero_lkl)
    mask = lkl.mu > 0
    q0[mask] = -2 * (lkl.zero_lkl[mask] - lkl.max_lkl[mask])

    # Cleaning
    mask = np.isnan(q0) | (q0 >= 500)
    q0 = q0[~mask]

    # Find 5sigma candidates (correct for look-elsewhere-effect)
    significance_sigma = -norm.ppf(norm.cdf(-sigma_limit) / len(q0))
    print(f"Corrected significance: {significance_sigma:.2f} sigma")
    candidate_mask = q0 > significance_sigma ** 2
    print(f"Number of candidates: {sum(candidate_mask)}")

    # Clustering
    candidate_idx, candidate_lambda = cluster(candidate_mask, lkl.mu[~mask])
    print(f"Number of clusters: {len(candidate_idx)}")

    # Gather LPSD data
    global_mngr = DataManager(data_path, bkg_path, tf_path,
                       data_prefix=data_prefix, buffer_path=buffer_path)
    global_sigma_limit = sigma_limit_consistency
    global_mngr.data.save_data()

    # Consistency tests - parallel version
    candidate_freqs = lkl.frequency[~mask][candidate_idx]
    candidate_lambdas = candidate_lambda
    
    # Create argument tuples for each candidate
    args_list = [(freq, lam) for freq, lam in zip(candidate_freqs, candidate_lambdas)]
    
    candidate_data = []
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(pool.imap(process_candidate, args_list),
                           total=len(args_list),
                           desc="Consistency tests"))
        for result in results:
            if result is not None:
                candidate_data.append(result)

    print(f"After consistency test: {len(candidate_data)}")
    nonzero_data = []
    for frequency, lambda_c, q0s, mus, sigmas in candidate_data:
        if any(q < global_sigma_limit ** 2 for q in q0s):
            continue
        nonzero_data.append((frequency, lambda_c, q0s, mus, sigmas))
    candidate_data = nonzero_data.copy()
    del nonzero_data
    print(f"After requiring power in both ifos: {len(candidate_data)}")
    import pdb; pdb.set_trace()

    # Print candidate list
    candidate_frequencies = []
    for freq, *_ in candidate_data:
        candidate_frequencies.append(freq)
    with open(os.path.join(out_folder, "candidates.csv"), "w") as f:
        f.write("\n".join([f"{x:.6e}" for x in candidate_frequencies]))

    # Plot candidates
    for i, (frequency, lambda_c, _, _, _) in enumerate(candidate_data):
        global_mngr.plot_candidate(frequency, lambda_c)
        plt.savefig(os.path.join(out_folder, f"candidate_{i}.pdf"))

if __name__ == '__main__':
    main(**parse_args())
