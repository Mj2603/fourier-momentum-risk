"""Cross-sectional momentum signal and long-only top-quantile weights."""

from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_signal(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Total return from t-lookback to t-skip: P[t-skip] / P[t-lookback] - 1.

    The value on date t uses prices up to and including t only. The default
    (252, 21) is the usual 12-1 month momentum definition.
    """
    if not 0 <= skip < lookback:
        raise ValueError("need 0 <= skip < lookback")
    return prices.shift(skip) / prices.shift(lookback) - 1.0


def top_quantile_weights(signal: pd.DataFrame, top_quantile: float = 0.3) -> pd.DataFrame:
    """Equal weights on the assets in the top ``top_quantile`` of the signal each row.

    Assets with a missing signal are excluded; rows with no valid signal get
    zero weight. Weights in each active row sum to 1.
    """
    if not 0 < top_quantile <= 1:
        raise ValueError("top_quantile must be in (0, 1]")
    n_valid = signal.notna().sum(axis=1)
    ranks = signal.rank(axis=1, method="first", ascending=False)  # 1 = best
    n_long = np.ceil(top_quantile * n_valid)
    long_mask = ranks.le(n_long, axis=0) & signal.notna()
    counts = long_mask.sum(axis=1).replace(0, np.nan)
    return long_mask.astype(float).div(counts, axis=0).fillna(0.0)
