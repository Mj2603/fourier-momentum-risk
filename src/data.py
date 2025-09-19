from __future__ import annotations
import numpy as np
import pandas as pd


def simulate_prices(n_assets: int = 20, n_days: int = 1500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Correlated geometric Brownian motions
    dt = 1/252
    mu = rng.normal(0.08, 0.03, size=n_assets)
    vol = rng.uniform(0.15, 0.35, size=n_assets)
    corr = 0.3
    cov = (vol[:, None] * vol[None, :]) * (corr + (1 - corr) * np.eye(n_assets))
    L = np.linalg.cholesky(cov)

    logp = np.zeros((n_days, n_assets))
    for t in range(1, n_days):
        z = rng.normal(size=n_assets)
        shock = L @ z
        logp[t] = logp[t-1] + (mu - 0.5 * vol**2) * dt + shock * np.sqrt(dt)

    prices = 100 * np.exp(logp)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=[f"A{i:02d}" for i in range(n_assets)])