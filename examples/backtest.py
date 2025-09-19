import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data import simulate_prices
from src.momentum import momentum_signal, rank_weights
from src.fourier_risk import fft_var_cvar


def run_backtest() -> None:
    prices = simulate_prices()
    signal = momentum_signal(prices)
    weights = rank_weights(signal).shift(1).fillna(0.0)

    rets = prices.pct_change().fillna(0.0)
    pnl = (weights * rets).sum(axis=1)

    mu, sigma = pnl.mean(), pnl.std()
    var, cvar = fft_var_cvar(alpha=0.95, mu=mu, sigma=sigma)

    os.makedirs("outputs", exist_ok=True)

    (1 + pnl).cumprod().plot(figsize=(7,4), title=f"Momentum Equity Curve | VaR95={var:.3f}, CVaR95={cvar:.3f}")
    plt.tight_layout()
    plt.savefig("outputs/equity_curve.png", dpi=150)

    print(f"Mean={mu:.5f} Std={sigma:.5f} VaR95={var:.5f} CVaR95={cvar:.5f}")


if __name__ == "__main__":
    run_backtest()
