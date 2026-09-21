"""Statistical tools for the LPSD analysis.

Copyright (C) 2026 Alexandre Goettel

License: GPLv3 or later <https://www.gnu.org/licenses/gpl-3.0.html>
Contact: Alexandre Goettel <alexandre.goettel@nottingham.ac.uk>
"""
import numpy as np
from scipy.stats import norm, skewnorm, skew
from scipy.special import owens_t
from scipy.signal import convolve
from scipy import constants
from findiff import FinDiff


_LOG_2_PI = 0.5 * np.log(2 * np.pi)
RHO_LOCAL = 0.4 * 1e6 * (constants.hbar * constants.c /
                         constants.e * 1e-9) ** 3  # GeV/cm^3 to GeV^4


def get_beta(freq_Hz, f_A_star):
    """Get the beta factor given frequency in Hz and A_cal."""
    m_phi = freq_Hz * constants.h / constants.e * 1e-9  # GeV
    return RHO_LOCAL / (np.pi * freq_Hz * 1e-6 * (f_A_star(freq_Hz) * m_phi)**2)


def log_likelihood(params, Y, bkg, peak_norm, peak_shape, model_args):
    """Likelihood of finding dark matter in the data."""
    # Actual likelihood calculation
    mu_DM = params
    assert Y.shape[0] == bkg.shape[0] and Y.shape[1] == peak_shape.shape[0]
    try:
        log_term = np.exp(bkg) + peak_norm*peak_shape*mu_DM
        log_term[log_term <= 0] = np.nan  # Protect against unwanted NaNs
        residuals = Y - np.log(log_term)
    except Exception as err:
        raise err
    lkl = pdf_skewnorm(residuals, 1, *[model_args[:, i][:, None] for i in range(3)])
    lkl_is_pos = lkl > 0  # Protect against DivideByZeroError
    log_lkl = np.full_like(lkl, np.nan)
    log_lkl[lkl_is_pos] = np.log(lkl[lkl_is_pos])
    assert log_lkl.shape == Y.shape

    # Add infinity protection - breaks norm but ok in empirical tests
    mask = np.isinf(log_lkl)
    if np.sum(mask):
        row_mins = np.min(np.where(mask, np.inf, log_lkl), axis=1)
        log_lkl[mask] = np.take(row_mins, np.where(mask)[0])

    return np.sum(np.sum(log_lkl, axis=1))


def sigma_at_point(func, point, initial_dx=1e-5, tolerance=1e-6, max_iterations=100):
    """
    Compute the derivative of a function at a given max. lkl. using findiff.

    Parameters:
    - func: The function to differentiate.
    - point: The point at which to compute the derivative.
    - initial_dx: Starting value for dx.
    - tolerance: Tolerance for derivative stabilization.
    - max_iterations: Maximum number of iterations to adjust dx.

    Returns:
    - Derivative estimate at the given point.
    """
    dx = initial_dx
    prev_derivative = None

    for _ in range(max_iterations):
        # Create the differential operator for the first derivative with respect to x
        d_dx = FinDiff(0, dx, 2)

        # Compute the derivative using a small interval around the point
        x_values = np.array([point - 2*dx, point - dx, point, point + dx, point + 2*dx])
        y_values = np.array([func(x) for x in x_values])
        derivative = d_dx(y_values)[2]

        # Check if the derivative estimate has stabilized
        if prev_derivative is not None and\
                abs((derivative - prev_derivative)/prev_derivative) < tolerance:
            return np.sqrt(-1. / derivative)
        prev_derivative = derivative
        dx /= 2  # Double the dx for the next iteration

    raise ValueError("Failed to converge to a stable derivative estimate.")


def pdf_skewnorm(x, A, loc=0, scale=1, alpha=0):
    """Wrapper for skew-normal amplitude with amplitude A."""
    return A*skewnorm.pdf(x, a=alpha, loc=loc, scale=scale)


def cdf_skewnorm(x, loc=0, scale=1, alpha=0):
    """Normalised decentralised skew normal CDF"""
    z = (x - loc) / scale
    return norm.cdf(z) - 2 * owens_t(z, alpha)


def logpdf_norm(x, loc=0, scale=1):
    """Implement Gaus logPDF."""
    return -_LOG_2_PI - np.log(scale) - 0.5*((x - loc) / scale)**2


def get_skewnorm_p0(data, max_alpha=5):
    """Get estimate alpha, mu, sigma from data."""
    # Get moments from data
    # https://modelingwithdata.org/pdfs/moments.pdf
    n = len(data)
    mean = np.mean(data)
    var = ((data - mean)**2).sum() / (n - 1)
    _skew = ((data - mean)**3).sum() * n / ((n - 1) * (n - 2))
    _skew = skew(data, bias=False)

    # Correct for numerical problems
    if np.abs(_skew) > .99:
        _skew = np.sign(_skew)*.99

    # Get skew normal parameters
    gamma = np.abs(_skew)**(2/3.)
    delta = np.sign(_skew) * np.sqrt(0.5*np.pi*gamma / (gamma + (0.5*(4 - np.pi))**(2/3.)))
    alpha = delta / np.sqrt(1. - delta**2)
    sigma = np.sqrt(var / (1. - 2/np.pi*delta**2))
    mu = mean - sigma * delta * np.sqrt(2 / np.pi)

    # Correct for "unphysicalness"
    alpha = np.sign(alpha)*max_alpha if abs(alpha) > max_alpha else alpha

    return alpha, mu, sigma


def kde_smoothing(timeseries, kernel_size):
    """Smooth a 1-D ndarray using a Gaussian kernel."""
    kernel_bound = int(kernel_size) + 1
    kernel_range = np.linspace(-kernel_bound, kernel_bound, 2 * kernel_bound + 1)
    kernel = norm.pdf(kernel_range, scale=kernel_size / 3)  # Scale is roughly 1/3 of kernel size
    kernel /= kernel.sum()  # Normalize the kernel

    # Reflect the time series at both ends & convolve
    padded_timeseries = np.concatenate((timeseries[:kernel_bound][::-1],
                                        timeseries,
                                        timeseries[-kernel_bound:][::-1]))
    smoothed_timeseries = convolve(padded_timeseries, kernel, mode='valid')
    return smoothed_timeseries


def cumf_q_mu(q_mu, mu, mu_prime, sigma):
    """Cumulative probability function for f(q_mu|mu_prime)."""
    out = norm.cdf(np.sqrt(q_mu) - (mu - mu_prime) / sigma)
    mask = q_mu > (mu / sigma)**2
    if len(mask):
        out[mask] = norm.cdf((q_mu[mask] - (mu**2 - 2*mu*mu_prime) / sigma**2) / (2*mu/sigma))
    return out


def cumf_q_mu_scalar(q_mu, mu, mu_prime, sigma):
    """Non-vectorized version of cumf_q_mu."""
    if q_mu > (mu / sigma) ** 2:
        return norm.cdf((q_mu - (mu ** 2 - 2 * mu * mu_prime) / sigma ** 2) / (2 * mu / sigma))
    else:
        return norm.cdf(np.sqrt(q_mu) - (mu - mu_prime) / sigma)
