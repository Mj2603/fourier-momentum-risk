"""Synthetic price generator (SYNTHETIC data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_prices(
    n_assets: int = 20,
    n_days: int = 1500,
    seed: int = 7,
    corr: float = 0.3,
    high_vol_mult: float = 1.0,
    p_enter_high: float = 0.0,
    p_exit_high: float = 0.1,
) -> pd.DataFrame:
    """Simulate correlated geometric Brownian motions with optional volatility regimes.

    Each asset has a drift drawn from N(8%, 3%) and a volatility drawn from
    U(15%, 35%) (annualised). Daily log returns share a constant pairwise
    correlation ``corr``. If ``p_enter_high > 0`` a common two-state Markov
    regime scales all volatilities by ``high_vol_mult`` while in the high state,
    which produces fat-tailed, volatility-clustered portfolio returns.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    mu = rng.normal(0.08, 0.03, size=n_assets)
    vol = rng.uniform(0.15, 0.35, size=n_assets)
    cov = np.outer(vol, vol) * (corr + (1 - corr) * np.eye(n_assets))
    chol = np.linalg.cholesky(cov)

    z = rng.standard_normal((n_days - 1, n_assets))
    shocks = z @ chol.T  # each row ~ N(0, cov)

    # Common volatility regime (state 1 = high volatility)
    u = rng.random(n_days - 1)
    state = np.zeros(n_days - 1, dtype=int)
    s = 0
    for t in range(n_days - 1):
        if s == 0 and u[t] < p_enter_high:
            s = 1
        elif s == 1 and u[t] < p_exit_high:
            s = 0
        state[t] = s
    scale = np.where(state == 1, high_vol_mult, 1.0)[:, None]

    var_t = (vol * scale) ** 2
    log_ret = (mu - 0.5 * var_t) * dt + scale * shocks * np.sqrt(dt)
    logp = np.vstack([np.zeros(n_assets), np.cumsum(log_ret, axis=0)])

    dates = pd.bdate_range("2015-01-01", periods=n_days)
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(100 * np.exp(logp), index=dates, columns=cols)
