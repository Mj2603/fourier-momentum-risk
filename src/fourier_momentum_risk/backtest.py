"""Simple daily backtest with periodic rebalancing and no transaction costs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def backtest(prices: pd.DataFrame, target_weights: pd.DataFrame, rebalance_every: int = 21) -> pd.DataFrame:
    """Return daily portfolio returns and turnover.

    Target weights computed on date t (from data up to t) are traded at the
    close of t and earn returns from t+1 onward, so there is no look-ahead.
    Between rebalances the holdings drift with prices.
    """
    rets = prices.pct_change().fillna(0.0).to_numpy()
    tw = target_weights.reindex(prices.index).fillna(0.0).to_numpy()
    n_days, n_assets = rets.shape
    w = np.zeros(n_assets)
    port = np.zeros(n_days)
    turnover = np.zeros(n_days)
    for t in range(n_days):
        # earn today's return on holdings set at the previous close
        port[t] = w @ rets[t]
        w = w * (1 + rets[t]) / (1 + port[t])
        if t % rebalance_every == 0:
            new_w = tw[t]
            turnover[t] = np.abs(new_w - w).sum()
            w = new_w.copy()
    return pd.DataFrame({"ret": port, "turnover": turnover}, index=prices.index)


def performance_summary(ret: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Annualised return (geometric), volatility, Sharpe (zero risk-free rate), max drawdown."""
    ret = ret.dropna()
    equity = (1 + ret).cumprod()
    years = len(ret) / periods_per_year
    ann_ret = equity.iloc[-1] ** (1 / years) - 1
    ann_vol = ret.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = ret.mean() / ret.std(ddof=1) * np.sqrt(periods_per_year)
    max_dd = (equity / equity.cummax() - 1).min()
    return {"ann_return": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": float(sharpe), "max_drawdown": float(max_dd)}
