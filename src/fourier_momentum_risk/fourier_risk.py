"""VaR and CVaR from a characteristic function using the COS method.

Reference: F. Fang and C. W. Oosterlee (2008), "A novel pricing method for
European options based on Fourier-cosine series expansions", SIAM J. Sci.
Comput. 31(2). The density of a return X on a truncation interval [a, b] is
approximated by a cosine series whose coefficients come directly from the
characteristic function phi(u) = E[exp(i u X)]:

    f(x) ~ sum'_{k=0}^{N-1} A_k cos(w_k (x - a)),   w_k = k pi / (b - a)
    A_k  = 2 / (b - a) * Re[ phi(w_k) exp(-i w_k a) ]

(the prime means the k = 0 term is halved). The CDF and the partial
expectation E[X 1{X <= x}] then have closed-form series, which give

    VaR_alpha  = -q,  where F(q) = 1 - alpha
    CVaR_alpha = -E[X | X <= q] = -E[X 1{X <= q}] / (1 - alpha)

for losses expressed as positive numbers. The method is only needed when the
characteristic function is known but the distribution function is not (for
example a sum of independent non-Gaussian components, whose characteristic
function is the product of the components'). For the normal and normal-mixture
cases used here, closed forms exist and are used to validate the COS output.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

CharFn = Callable[[np.ndarray], np.ndarray]


def cf_normal(u: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return np.exp(1j * mu * u - 0.5 * sigma**2 * u**2)


def cf_normal_mixture(u: np.ndarray, weights, mus, sigmas) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    out = np.zeros(u.shape, dtype=complex)
    for w, m, s in zip(weights, mus, sigmas):
        out += w * cf_normal(u, m, s)
    return out


def truncation_range(mean: float, std: float, width: float = 12.0) -> tuple[float, float]:
    """Interval [mean - width*std, mean + width*std]. Use a wider one for fatter tails."""
    return mean - width * std, mean + width * std


def _coefficients(cf: CharFn, a: float, b: float, n_terms: int) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(n_terms)
    w = k * np.pi / (b - a)
    A = 2.0 / (b - a) * np.real(cf(w) * np.exp(-1j * w * a))
    A[0] *= 0.5
    return w, A


def cos_pdf(x, cf: CharFn, a: float, b: float, n_terms: int = 512) -> np.ndarray:
    w, A = _coefficients(cf, a, b, n_terms)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return np.cos(np.outer(x - a, w)) @ A


def cos_cdf(x, cf: CharFn, a: float, b: float, n_terms: int = 512) -> np.ndarray:
    w, A = _coefficients(cf, a, b, n_terms)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    d = x - a
    terms = np.empty((x.size, n_terms))
    terms[:, 0] = d
    terms[:, 1:] = np.sin(np.outer(d, w[1:])) / w[1:]
    return terms @ A


def cos_partial_expectation(x, cf: CharFn, a: float, b: float, n_terms: int = 512) -> np.ndarray:
    """E[X 1{X <= x}] from the cosine series."""
    w, A = _coefficients(cf, a, b, n_terms)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    d = x - a
    terms = np.empty((x.size, n_terms))
    terms[:, 0] = 0.5 * (x**2 - a**2)
    wk = w[1:]
    arg = np.outer(d, wk)
    terms[:, 1:] = x[:, None] * np.sin(arg) / wk + (np.cos(arg) - 1.0) / wk**2
    return terms @ A


def cos_var_cvar(cf: CharFn, alpha: float, a: float, b: float, n_terms: int = 512) -> tuple[float, float]:
    """VaR and CVaR at confidence ``alpha`` (e.g. 0.99) for the return with this characteristic function.

    Both are reported as positive loss numbers (losses are -X).
    """
    if not 0.5 < alpha < 1:
        raise ValueError("alpha must be in (0.5, 1)")
    p = 1.0 - alpha
    q = brentq(lambda x: cos_cdf(x, cf, a, b, n_terms)[0] - p, a, b, xtol=1e-14, rtol=1e-14, maxiter=500)
    pe = cos_partial_expectation(q, cf, a, b, n_terms)[0]
    return float(-q), float(-pe / p)


def normal_var_cvar(alpha: float, mu: float, sigma: float) -> tuple[float, float]:
    """Closed-form Gaussian VaR/CVaR (positive losses)."""
    z = norm.ppf(1 - alpha)
    var = -(mu + sigma * z)
    cvar = -mu + sigma * norm.pdf(z) / (1 - alpha)
    return float(var), float(cvar)


def normal_mixture_var_cvar(alpha: float, weights, mus, sigmas) -> tuple[float, float]:
    """Mixture VaR/CVaR from its closed-form CDF (root finding) and partial expectation."""
    weights, mus, sigmas = map(np.asarray, (weights, mus, sigmas))
    p = 1 - alpha
    lo = float(np.min(mus - 40 * sigmas))
    hi = float(np.max(mus + 40 * sigmas))
    q = brentq(lambda x: float(np.sum(weights * norm.cdf((x - mus) / sigmas))) - p, lo, hi, xtol=1e-14, rtol=1e-14)
    zq = (q - mus) / sigmas
    pe = np.sum(weights * (mus * norm.cdf(zq) - sigmas * norm.pdf(zq)))
    return float(-q), float(-pe / p)


def historical_var_cvar(returns: np.ndarray, alpha: float) -> tuple[float, float]:
    """Empirical VaR (lower quantile) and CVaR (mean of returns at or below it)."""
    r = np.asarray(returns, dtype=float)
    q = np.quantile(r, 1 - alpha)
    return float(-q), float(-r[r <= q].mean())
