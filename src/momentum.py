from __future__ import annotations
import numpy as np
import pandas as pd


def momentum_signal(prices: pd.DataFrame, lookback_days: int = 252) -> pd.DataFrame:
    rets = prices.pct_change()
    roll = (1 + rets).rolling(lookback_days).apply(lambda x: np.prod(1 + x) - 1, raw=False)
    return roll


def rank_weights(signal: pd.DataFrame, top_quantile: float = 0.3) -> pd.DataFrame:
    ranks = signal.rank(axis=1, method="first")
    cutoff = np.ceil((1 - top_quantile) * ranks.shape[1])
    long_mask = ranks.gt(cutoff, axis=0)
    w = long_mask.div(long_mask.sum(axis=1), axis=0).fillna(0.0)
    return w