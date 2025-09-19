from __future__ import annotations
import numpy as np
from numpy.fft import fft

# Simple characteristic function-based VaR/CVaR for normal-mixture toy PnL

def cf_normal(u: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(1j * mu * u - 0.5 * (sigma**2) * u**2)


def cf_mixture(u: np.ndarray, w: float, mu1: float, s1: float, mu2: float, s2: float) -> np.ndarray:
    return w * cf_normal(u, mu1, s1) + (1 - w) * cf_normal(u, mu2, s2)


def fft_var_cvar(alpha: float = 0.95, mu: float = 0.0, sigma: float = 1.0) -> tuple[float, float]:
    # Grid in Fourier domain
    N = 2**12
    du = 0.01
    u = np.arange(N) * du

    phi = cf_normal(u - 1j * 0.5, mu, sigma)
    # Invert to get pdf via FFT of damped characteristic function (COS/FFT style, simplified)
    # Here we compute CDF numerically from pdf grid for demonstration.
    x_max = 5 * sigma
    dx = 2 * x_max / N
    x = -x_max + np.arange(N) * dx

    # pdf approx via FFT of phi
    # Using relationship of cf and pdf with damping; for demo only
    pdf = np.real(fft(phi))
    pdf = np.clip(pdf, 0, None)
    pdf = pdf / (np.sum(pdf) * dx)

    cdf = np.cumsum(pdf) * dx
    var_idx = np.searchsorted(cdf, 1 - alpha)
    var = -x[var_idx]
    cvar = -np.sum(x[:var_idx] * pdf[:var_idx]) / np.sum(pdf[:var_idx])
    return float(var), float(cvar)