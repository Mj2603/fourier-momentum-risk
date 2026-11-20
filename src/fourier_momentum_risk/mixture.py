"""Two-or-more component univariate Gaussian mixture fitted by EM."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def fit_normal_mixture(x, n_components: int = 2, n_iter: int = 500, tol: float = 1e-10, seed: int = 0):
    """Return (weights, means, stds) sorted by std, ascending."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    w = np.full(n_components, 1.0 / n_components)
    mu = rng.choice(x, n_components, replace=False)
    sd = np.std(x) * np.linspace(0.5, 2.0, n_components)
    prev = -np.inf
    for _ in range(n_iter):
        dens = w * norm.pdf(x[:, None], mu, sd)
        total = dens.sum(axis=1, keepdims=True)
        ll = np.log(total).sum()
        resp = dens / total
        nk = resp.sum(axis=0)
        w = nk / x.size
        mu = (resp * x[:, None]).sum(axis=0) / nk
        sd = np.sqrt((resp * (x[:, None] - mu) ** 2).sum(axis=0) / nk)
        sd = np.maximum(sd, 1e-12)
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    order = np.argsort(sd)
    return w[order], mu[order], sd[order]
