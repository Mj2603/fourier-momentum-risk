"""Reproducible experiment on SYNTHETIC data. Writes results/results.md and CSV files.

Run from the repository root:  python experiments/run_experiment.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from fourier_momentum_risk import (
    backtest,
    cf_normal,
    cf_normal_mixture,
    cos_var_cvar,
    fit_normal_mixture,
    momentum_signal,
    normal_var_cvar,
    performance_summary,
    simulate_prices,
    top_quantile_weights,
    truncation_range,
)
from fourier_momentum_risk.fourier_risk import historical_var_cvar, normal_mixture_var_cvar

OUT = Path(__file__).resolve().parents[1] / "results"

SIM = dict(n_assets=20, n_days=2520, seed=7, corr=0.3, high_vol_mult=2.5, p_enter_high=0.02, p_exit_high=0.1)
LOOKBACK, SKIP, TOP_Q, REBAL = 252, 21, 0.3, 21


def cos_accuracy() -> pd.DataFrame:
    rows = []
    cases = {
        "normal(0.0005, 0.012)": (
            lambda u: cf_normal(u, 0.0005, 0.012),
            normal_var_cvar(0.99, 0.0005, 0.012),
            truncation_range(0.0005, 0.012, 12),
        ),
        "mixture 0.85*N(0.0006,0.008)+0.15*N(-0.002,0.025)": (
            lambda u: cf_normal_mixture(u, [0.85, 0.15], [0.0006, -0.002], [0.008, 0.025]),
            normal_mixture_var_cvar(0.99, [0.85, 0.15], [0.0006, -0.002], [0.008, 0.025]),
            (-0.25, 0.25),
        ),
    }
    for name, (cf, (v0, c0), (a, b)) in cases.items():
        for n in (16, 32, 64, 128, 256, 512):
            v, c = cos_var_cvar(cf, 0.99, a, b, n_terms=n)
            rows.append({"distribution": name, "n_terms": n, "var99_exact": v0, "var99_cos": v,
                         "abs_err_var": abs(v - v0), "cvar99_exact": c0, "cvar99_cos": c, "abs_err_cvar": abs(c - c0)})
    return pd.DataFrame(rows)


def kupiec_pvalue(n_exc: int, n: int, p: float) -> float:
    phat = n_exc / n
    if n_exc in (0, n):
        ll1 = 0.0
    else:
        ll1 = n_exc * np.log(phat) + (n - n_exc) * np.log(1 - phat)
    ll0 = n_exc * np.log(p) + (n - n_exc) * np.log(1 - p)
    return float(chi2.sf(-2 * (ll0 - ll1), df=1))


def main() -> None:
    OUT.mkdir(exist_ok=True)

    acc = cos_accuracy()
    acc.to_csv(OUT / "cos_accuracy.csv", index=False)

    prices = simulate_prices(**SIM)
    sig = momentum_signal(prices, LOOKBACK, SKIP)
    w_mom = top_quantile_weights(sig, TOP_Q)
    valid = sig.notna().any(axis=1)
    w_ew = pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns).where(valid, 0.0)
    start = int(np.argmax(valid.to_numpy()))
    # Start rebalancing on the first date with a valid signal.
    p_live = prices.iloc[start:]
    mom = backtest(p_live, w_mom.iloc[start:], REBAL)
    ew = backtest(p_live, w_ew.iloc[start:], REBAL)
    mom_r, ew_r = mom["ret"].iloc[1:], ew["ret"].iloc[1:]

    perf = pd.DataFrame({
        "momentum_top30": performance_summary(mom_r),
        "equal_weight": performance_summary(ew_r),
    }).T
    perf["avg_turnover_per_rebalance"] = [
        mom["turnover"].iloc[REBAL::REBAL].mean(), ew["turnover"].iloc[REBAL::REBAL].mean()
    ]
    perf["n_days"] = len(mom_r)
    perf.to_csv(OUT / "backtest_performance.csv")

    # Risk: fit on first half of momentum returns, test exceedances on second half.
    r = mom_r.to_numpy()
    half = len(r) // 2
    train, test = r[:half], r[half:]
    w, m, s = fit_normal_mixture(train, 2)
    mu, sd = train.mean(), train.std(ddof=1)
    risk_rows = []
    for alpha in (0.95, 0.99):
        est = {
            "historical": historical_var_cvar(train, alpha),
            "gaussian_closed_form": normal_var_cvar(alpha, mu, sd),
            "gaussian_cos": cos_var_cvar(lambda u: cf_normal(u, mu, sd), alpha, *truncation_range(mu, sd), 256),
            "mixture2_cos": cos_var_cvar(lambda u: cf_normal_mixture(u, w, m, s), alpha, -0.5, 0.5, 1024),
        }
        for method, (v, c) in est.items():
            exc = int((test < -v).sum())
            risk_rows.append({
                "alpha": alpha, "method": method, "var_train": v, "cvar_train": c,
                "test_days": len(test), "exceedances": exc, "exceedance_rate": exc / len(test),
                "expected_rate": 1 - alpha, "kupiec_pvalue": kupiec_pvalue(exc, len(test), 1 - alpha),
            })
    risk = pd.DataFrame(risk_rows)
    risk.to_csv(OUT / "var_backtest.csv", index=False)
    mix = pd.DataFrame({"weight": w, "mean": m, "std": s})
    mix.to_csv(OUT / "mixture_fit.csv", index_label="component")

    fmt = lambda df: df.to_markdown(floatfmt=".6g")
    md = [
        "# Results (SYNTHETIC data)",
        "",
        "All numbers below are produced by `python experiments/run_experiment.py` on simulated prices. "
        "No market data is used.",
        "",
        f"Simulation: {SIM}. Momentum: lookback={LOOKBACK}, skip={SKIP}, long top {TOP_Q:.0%}, "
        f"rebalance every {REBAL} days, no transaction costs.",
        "",
        "## 1. COS inversion accuracy (alpha = 0.99) against closed forms",
        "",
        fmt(acc.drop(columns=["var99_exact", "cvar99_exact"]).set_index(["distribution", "n_terms"])),
        "",
        "## 2. Backtest on synthetic prices",
        "",
        fmt(perf),
        "",
        "## 3. Daily VaR/CVaR of the momentum portfolio",
        "",
        f"Estimated on the first {half} daily returns, exceedances counted on the remaining {len(test)}. "
        "Unconditional estimates; the simulated volatility regimes make exceedances cluster.",
        "",
        "Fitted two-component mixture (training half):",
        "",
        fmt(mix),
        "",
        fmt(risk.set_index(["alpha", "method"])),
        "",
    ]
    (OUT / "results.md").write_text("\n".join(md))
    print((OUT / "results.md").read_text())


if __name__ == "__main__":
    main()
