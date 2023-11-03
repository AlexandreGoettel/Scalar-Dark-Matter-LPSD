"""Calculate experimental upper limits based on real data."""
from multiprocessing import Pool
from tqdm import tqdm, trange
import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy import constants
from scipy.optimize import minimize
from scipy.stats import norm, skewnorm
from scipy.interpolate import interp1d
from scipy.special import erf
# Project imports
import sensutils
import models
import hist


# Global constant for peak normalisation in likelihood calculation
rho_local = 0.4 / (constants.hbar / constants.e * 1e-7 * constants.c)**3  # GeV/cm^3 to GeV^4


def logskewnorm_pdf(x, alpha, mu, sigma):
    return 1. / (np.sqrt(2*np.pi)*x*sigma) * np.exp(-0.5*(((np.log(x) - mu)/sigma)**2)) * (1 + erf(alpha*((np.log(x) - mu) / (np.sqrt(2)*sigma))))


def cumf_q_mu(q_mu, mu, mu_prime, sigma):
    """Cumulative probability function for f(q_mu|mu_prime)."""
    out = norm.cdf(np.sqrt(q_mu) - (mu - mu_prime) / sigma)
    mask = q_mu > (mu / sigma)**2
    if len(mask):
        out[mask] = norm.cdf((q_mu[mask] - (mu**2 - 2*mu*mu_prime) / sigma**2) / (2*mu/sigma))
    return out


def profile_calc(Y, bkg, peak_norm, alpha_skew, loc_skew, sigma_skew, alpha_CL):
    """TMP - Test out full profile calculation for comparison."""
    # Get max lkl
    # Skew normal lkl with mu = Lambda^-2 (to avoid double-peak lkl)
    def lkl(param):
        res = Y - np.log(np.exp(bkg) + peak_norm*param)
        return skewnorm.logpdf(res, alpha_skew, loc=loc_skew, scale=sigma_skew)

    mode_skew = sensutils.get_mode_skew(alpha_skew, loc_skew, sigma_skew)
    popt = minimize(lambda x: -lkl(x),
                    (np.exp(Y - mode_skew) - np.exp(bkg)) / peak_norm,
                    method="Nelder-Mead")

    def lkl_new(param, data):
        res = data - np.log(np.exp(bkg) + peak_norm*param)
        return skewnorm.logpdf(res, alpha_skew, loc=loc_skew, scale=sigma_skew)

    sigma_distr = sigma_skew*np.sqrt(1 - 2/np.pi*(alpha_skew**2 / (1 + alpha_skew**2)))
    sigma = np.abs(sigma_distr*np.exp(Y - mode_skew) / peak_norm)

    # Generate a bunch of q_mus by fluctuating background
    def get_q_mu(data, mu_test):
        mu_hat = (np.exp(data - mode_skew) - np.exp(bkg)) / peak_norm
        max_lkl = lkl_new(mu_hat, data)
        zero_lkl = lkl_new(0, data)
        q_mu = np.zeros_like(data)
        mask = mu_hat < mu_test
        q_mu[mask] = -2 * (lkl_new(mu_test, data[mask]) - max_lkl[mask])
        mask = mu_hat < 0
        q_mu[mask] = -2 * (lkl_new(mu_test, data[mask]) - zero_lkl[mask])
        return q_mu

    n_q_mu_0 = 47500
    n_q_mu_mu = 47500
    data_mu_0 = bkg + skewnorm.rvs(alpha_skew, loc=loc_skew, scale=sigma_skew, size=n_q_mu_0)
    full_bkg = bkg + skewnorm.rvs(alpha_skew, loc=loc_skew, scale=sigma_skew, size=n_q_mu_mu)
    def get_p_val(mu_test, verbose=False):
        data_mu_mu = np.log(np.exp(full_bkg) + peak_norm*mu_test)
        q_mu_0 = get_q_mu(data_mu_0, mu_test)
        q_mu_mu = get_q_mu(data_mu_mu, mu_test)

        # Calculate p-value
        med_q_mu_0 = np.median(q_mu_0)
        k = np.sum(np.array(q_mu_mu >= med_q_mu_0, dtype=int))
        p_val, p_sigma = k / float(n_q_mu_mu), np.sqrt(sensutils.get_eff_var(k, n_q_mu_mu))
        if verbose:
            bins = np.linspace(0, max(q_mu_0), 100)
            ax = hist.plot_hist(q_mu_0, bins, logy=True, label="f(q_mu|0)")
            ax = hist.plot_hist(q_mu_mu, bins, ax=ax, label="f(q_mu|mu)")
            ax.axvline(med_q_mu_0, linestyle="--", color="r")
            ax.set_title(f"{p_val:.3f} +- {p_sigma:.4f}")
            plt.show()
        return (p_val - (1 - alpha_CL))**2

    # Find good starting value and minimize
    test = np.linspace(.1*sigma, 5*sigma, 50)
    diffs = [get_p_val(val) for val in test]
    popt = minimize(get_p_val, test[np.argmin(diffs)],
                    bounds=[(0, 5*sigma)],
                    method="Nelder-Mead",
                    tol=1e-6)

    # Remember to convert from mu to Lambda^-1
    Lambda = np.sqrt(popt.x)
    sigma /= 2*Lambda
    return Lambda, sigma




def get_upper_limits_approx(X, Y, peak_norm, model_args,
                            alpha_CL=.95):
    """Calculate an upper limit based on asymptotic formulae"""
    # Bkg model was already optimised
    # So take popt from model_args to get Lambda
    # Just need mu_hat as free parameter
    x_knots, y_knots, alpha_skew, loc_skew, sigma_skew = model_args
    bkg = models.model_xy_spline(np.concatenate([x_knots, y_knots]), extrapolate=True)(X)

    # Get mu_hat / sigma from maximum likelihood
    mode_skew = sensutils.get_mode_skew(alpha_skew, loc_skew, sigma_skew)
    sign = 1 if np.exp(Y - mode_skew) - np.exp(bkg) > 0 else -1
    mu_hat = sign*np.sqrt(np.abs(np.exp(Y - mode_skew) - np.exp(bkg)) / peak_norm)

    sigma_distr = sigma_skew*np.sqrt(1 - 2/np.pi*(alpha_skew**2 / (1 + alpha_skew**2)))
    sigma = np.abs(sigma_distr*np.exp(Y - mode_skew) / (2. * peak_norm * mu_hat))

    # Test area
    # Get mu_hat with mu = Lambda^2
    def lkl(param):
        res = Y - np.log(np.exp(bkg) + peak_norm*param)
        return skewnorm.logpdf(res, alpha_skew, loc=loc_skew, scale=sigma_skew)
    popt = minimize(lambda x: -lkl(x),
                    (np.exp(Y - mode_skew) - np.exp(bkg)) / peak_norm,
                    method="Nelder-Mead")
    mu_hat = popt.x
    max_lkl = -popt.fun

    # Calculate the uncertainty vs the lower side because that is the relevant
    # one for upper limits
    def lkl_minus(param):
        return (lkl(param) - (max_lkl - 0.5))**2
    sigma = sigma_distr*np.exp(Y - mode_skew) / peak_norm
    popt = minimize(lkl_minus,
                    (max(-np.exp(bkg)/peak_norm, mu_hat - sigma) + mu_hat)/2.,
                    method="Nelder-Mead",
                    tol=1e-20,
                    bounds=[(-np.exp(bkg)/peak_norm, mu_hat)])
    _sigma = mu_hat - popt.x
    # return np.sqrt(np.abs(mu_hat)) + 1.64*_sigma/(2*np.sqrt(np.abs(mu_hat))), _sigma/(2*np.sqrt(np.abs(mu_hat)))
    if popt.fun > 1e-10:
        x = np.linspace(mu_hat-2*sigma, mu_hat+2*sigma, 100)
        plt.figure()
        ax = plt.subplot(111)
        ax.set_yscale("log")
        plt.plot(x, -lkl(x), label="Likelihood")
        plt.axhline(-lkl(mu_hat)+0.5, color="r", linestyle="--")
        plt.axvline(mu_hat, color="r", linestyle="--", label=r"$\hat{\mu}$")
        plt.axvline(mu_hat-_sigma, color="g", linestyle="--", label=r"$\mu - \sigma$")
        plt.axvline(mu_hat-sigma, color="k", linestyle="--", label=r"Initial guess for $\mu - \sigma$")
        plt.xlabel(r"$\mu$")
        plt.ylabel("-logL")
        plt.legend(loc="best")
        plt.show()
        raise ValueError

    def opt_pval(param):
        # Find the median of f_q_mu_0
        q_mu = np.linspace(0, max(25, (param/sigma)**2+1), 1000)  # TODO: maybe adaptive?
        F_q_mu_0 = cumf_q_mu(q_mu, param, 0, sigma)
        med_idx = np.where(F_q_mu_0 >= .5)[0][0]

        # Get p-value from f_q_mu_mu
        p_val = 1. - cumf_q_mu(np.ones(1)*q_mu[med_idx], param, param, sigma)
        return (p_val - (1 - alpha_CL))**2

    # Convert sigma from mu to Lambda^-1
    sigma = _sigma
    sigma /= 2*np.sqrt(np.abs(mu_hat))
    # return np.sqrt(np.abs(mu_hat)) + 1.64*sigma, sigma
    popt = minimize(opt_pval, sigma, bounds=[(0, None)],
                    method="Nelder-Mead", tol=1e-10)

    # Second testing area
    # Return upper limit with uncertainty
    # return popt.x, sigma
    test_mu, test_sigma = [], []
    for i in trange(10):
        Y = bkg + skewnorm.rvs(alpha_skew, loc=loc_skew, scale=sigma_skew)
        new_x, new_s = profile_calc(Y, bkg, peak_norm, alpha_skew, loc_skew, sigma_skew, alpha_CL)
        test_mu.append(new_x)
        test_sigma.append(new_s)

    print(popt.x, sigma)
    print(new_x, new_s)
    print(np.mean(test_mu), np.std(test_mu), np.mean(sigma), np.std(sigma))
    print("---")
    error
    return new_x, new_s


def process_segment(args):
    """Calculate an upper limit - wrapper for multiprocessing."""
    i, freq, logPSD, fi, lo, peak_norm, model_args, alpha_CL = args
    return i, lo+fi, get_upper_limits_approx(freq, logPSD, peak_norm,
                                             model_args, alpha_CL=alpha_CL)


def main():
    """Get all necessary data and launch analysis."""
    # Analysis variables
    n_frequencies = 3  # Test frequencies per segment (of 10k)
    num_processes = 1
    alpha_CL = 0.95

    # TODO: rel. paths
    json_path = "data/processing_results.json"
    data_path = "data/result_epsilon10_1243393026_1243509654_H1.txt"
    calib_path = "../shared_git_data/Calibration_factor_A_star.txt"

    # 0. Get A_star
    calib = np.loadtxt(calib_path, delimiter="\t")
    f_calib = {"H1": interp1d(calib[:, 1], calib[:, 0]),
               "L1": interp1d(calib[:, 1], calib[:, 0])}  # TODO: other file
    # 0.1 Get data
    with h5py.File(sensutils.get_corrected_path(data_path)) as _f:
        peak_mask = np.array(_f["peak_mask"][()], dtype=bool)
        freq = np.log(_f["frequency"][()])
        logPSD = _f["logPSD"][()]
    # 0.2 Get pre-processing results
    df_key = "splines_" + sensutils.get_df_key(data_path)
    df = sensutils.get_results(df_key, json_path)
    assert df is not None

    # 1. Generate args
    ifo = data_path.split(".")[-2].split("_")[-1]
    jump_idx_space = np.where(~peak_mask)[0]
    del peak_mask
    def args():
        """Loop over segments and ifos to give data and args."""
        # Loop over segments/ifos
        count = 0
        for _, (frequencies, x_knots, y_knots, alpha_skew, loc_skew,
                sigma_skew, chi_sqr) in df.iterrows():
            [lo, hi] = jump_idx_space[frequencies]
            freq_subset = freq[lo:hi].copy()
            if np.exp(freq_subset[0]) < calib[0, 1] or np.exp(freq_subset[-1]) > calib[-1, 1]:
                continue
            logPSD_subset = logPSD[lo:hi].copy()
            test_frequencies = map(lambda x: int(x) - lo,
                                   np.linspace(lo, hi, n_frequencies+2)[1:-1])

            # TMP
            bkg = models.model_xy_spline(np.concatenate([x_knots, y_knots]), extrapolate=True)(freq_subset)
            def f(x, A, alpha, mu, sigma):
                return A*logskewnorm_pdf(x, alpha, mu, sigma)
            _, bins = np.histogram(np.exp(logPSD_subset - bkg), 100)

            popt, pcov, chi = hist.fit_hist(f, np.exp(logPSD_subset - bkg),
                                            bins, [1e4, alpha_skew, loc_skew, sigma_skew],
                                            get_chi_sqr=True)
            axes = hist.plot_func_hist(f, popt, np.exp(logPSD_subset - bkg), bins)
            axes[0].set_title(f"chi:{chi:.1f}")

            def f_eps(x, A, alpha, mu, sigma, eps, omega):
                return A*logskewnorm_pdf((x - eps)/omega, alpha, mu, sigma)

            data = np.exp(logPSD_subset) - np.exp(bkg)
            _, bins = np.histogram(data, 100)
            print(loc_skew, sigma_skew)
            error
            p0 = [(bins[1] - bins[0])**-1, alpha_skew, loc_skew, sigma_skew,
                  -1e-39, 1e-36]
            popt, pcov, chi = hist.fit_hist(f_eps, data, bins, p0,
                                            get_chi_sqr=True)

            plt.figure()
            x = (bins[1:] + bins[:-1]) / 2
            plt.plot(x, f_eps(x, *p0))
            axes = hist.plot_func_hist(f_eps, popt, data, bins)
            hist.plot_func_hist(f_eps, p0, data, bins)
            axes[0].set_title(f"chi:{chi:.1f}")
            print(p0)
            print(popt)
            plt.show()
            error
            # END TMP #

            # Loop over frequencies
            for fi in test_frequencies:
                # Calculate a normalisation constant to pass to the peak
                # This includes calibration and conversion
                freq_Hz = np.exp(freq_subset[fi])
                A_star_sqr = f_calib[ifo](freq_Hz)**2
                peak_norm = rho_local / (np.pi*freq_Hz**3 * A_star_sqr
                                         * (constants.e / constants.h)**2)
                # These are the arguments to pass to a single computation
                model_args = x_knots, y_knots, alpha_skew, loc_skew, sigma_skew
                yield (count, freq_subset[fi], logPSD_subset[fi], fi, lo, peak_norm,
                       model_args, alpha_CL)
                count += 1

    # 2. Create job Pool
    n_upper_limits = len(df) * n_frequencies
    with Pool(num_processes, maxtasksperchild=10) as pool:
        results = []
        with tqdm(total=n_upper_limits, position=0, desc="Calc. upper lim") as pbar:
            for result in pool.imap_unordered(process_segment, args()):
                results.append(result)
                pbar.update(1)

    # 3. Merge results
    upper_limit_data = np.zeros((4, len(results)))
    for i, fi, upper_limit in results:
        upper_limit_data[0, i] = np.exp(freq[fi])
        upper_limit_data[1:3, i] = upper_limit  # mu_upper, sigma
        upper_limit_data[3, i] = logPSD[fi]
    idx = np.argsort(upper_limit_data[0, :])
    upper_limit_data = upper_limit_data[:, idx]

    # 4. Plot!
    def smooth_curve(y, w):
        return sensutils.kde_smoothing(y, w)

    w = 33
    smooth_lim = smooth_curve(upper_limit_data[1, :], w)
    smooth_sigma = smooth_curve(upper_limit_data[2, :], w)
    geo_data = np.loadtxt("geo_limits.csv", delimiter=",")

    plt.figure()
    ax = plt.subplot(111)
    # ax.errorbar(upper_limit_data[0, :], smooth_lim, smooth_sigma,
    #             fmt=".", label="LIGO", linewidth=2., color="C1")
    # ax.errorbar(upper_limit_data[0, :], upper_limit_data[1, :], upper_limit_data[2, :],
    #             fmt=".", label="LIGO", linewidth=2., color="C1")
    ax.plot(upper_limit_data[0, :], smooth_lim,
            linewidth=2, color="C1", label="Approx sigma cumf")
    ax.fill_between(upper_limit_data[0, :], smooth_lim+smooth_sigma, smooth_lim-smooth_sigma,
                    color="C1", alpha=.33)
    ax.plot(geo_data[:, 0], geo_data[:, 1], label="GEO600", linewidth=2, color="C0")

    old = np.load("tmp.npy")
    smooth_old = smooth_curve(old[1, :], w)
    smooth_sigma_old = smooth_curve(old[2, :], w)
    ax.plot(old[0, :], smooth_old, linewidth=2, color="C2")
    ax.fill_between(old[0, :], smooth_old+smooth_sigma_old, smooth_old-smooth_sigma_old,
                    color="C2", alpha=.33, label="Full approx")
    # np.save("tmp.npy", upper_limit_data)

    # Nice things
    ax.set_yscale("log")
    ax.set_xscale("log")
    axTop = ax.twiny()
    axTop.set_xscale("log")
    axTop.set_xlim(ax.get_xlim())
    ticks = np.array([1e-13, 1e-12, 1e-11])
    axTop.set_xticks(ticks * constants.e / constants.h)
    axTop.set_xticklabels(ticks)
    ax.set_xlabel("Frequency (Hz)")
    axTop.set_xlabel("DM mass (eV)")
    ax.set_ylabel(r"$1/\Lambda_i$ (GeV)")
    ax.set_title("PRELIMINARY 95% UPPER LIMITS")
    ax.legend(loc="best")
    ax.grid(color="grey", alpha=.33, linestyle="--", linewidth=1.5, which="both")
    ax.minorticks_on()


    # TMP - ALTERNATIVE CALCULATION FOR COMPARISON
    # for _, (frequencies, x_knots, y_knots, alpha_skew, loc_skew,
    #         sigma_skew, chi_sqr) in tqdm(df.iterrows(), total=len(df)):
    #     [lo, hi] = jump_idx_space[frequencies]
    #     if np.exp(freq[lo]) < calib[0, 1] or np.exp(freq[hi]) > calib[-1, 1]:
    #         continue
    plt.show()


if __name__ == '__main__':
    main()