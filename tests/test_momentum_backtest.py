import numpy as np
import pandas as pd
import pytest

from fourier_momentum_risk import (
    backtest,
    fit_normal_mixture,
    momentum_signal,
    performance_summary,
    simulate_prices,
    top_quantile_weights,
)


def test_momentum_signal_is_skip_lookback_return():
    p = simulate_prices(4, 400, seed=1)
    s = momentum_signal(p, lookback=60, skip=5)
    t = 300
    expected = p.iloc[t - 5] / p.iloc[t - 60] - 1
    np.testing.assert_allclose(s.iloc[t].to_numpy(), expected.to_numpy())
    assert s.iloc[:60].isna().all().all()


def test_momentum_signal_has_no_look_ahead():
    p = simulate_prices(5, 400, seed=2)
    s1 = momentum_signal(p, 100, 10)
    p2 = p.copy()
    p2.iloc[250:] *= 3.0  # change the future only
    s2 = momentum_signal(p2, 100, 10)
    pd.testing.assert_frame_equal(s1.iloc[:250], s2.iloc[:250])


def test_top_quantile_weights():
    sig = pd.DataFrame([[0.1, 0.5, -0.2, 0.3, np.nan], [np.nan] * 5])
    w = top_quantile_weights(sig, top_quantile=0.5)
    # 4 valid assets -> top 2: columns 1 and 3
    np.testing.assert_allclose(w.iloc[0].to_numpy(), [0, 0.5, 0, 0.5, 0])
    np.testing.assert_allclose(w.iloc[1].to_numpy(), 0.0)


def test_backtest_buy_and_hold_matches_price_ratio():
    p = simulate_prices(3, 300, seed=3)
    w = pd.DataFrame(0.0, index=p.index, columns=p.columns)
    w.iloc[0] = [0.2, 0.3, 0.5]
    res = backtest(p, w, rebalance_every=10_000)  # rebalance only at t=0
    equity = (1 + res["ret"]).prod()
    expected = float((np.array([0.2, 0.3, 0.5]) * p.iloc[-1] / p.iloc[0]).sum())
    assert equity == pytest.approx(expected, rel=1e-10)


def test_backtest_uses_previous_close_weights():
    idx = pd.bdate_range("2020-01-01", periods=3)
    p = pd.DataFrame({"A": [100, 110, 121], "B": [100, 90, 81]}, index=idx, dtype=float)
    w = pd.DataFrame({"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 1.0]}, index=idx)
    res = backtest(p, w, rebalance_every=1)
    # day 1 return earned on day-0 weights (all A) = +10%; day 2 on day-1 weights (all B) = -10%
    np.testing.assert_allclose(res["ret"].to_numpy(), [0.0, 0.10, -0.10])


def test_simulated_moments_match_parameters():
    p = simulate_prices(n_assets=3, n_days=200_000, seed=4, corr=0.5)
    lr = np.log(p).diff().dropna()
    vol = lr.std() * np.sqrt(252)
    rng = np.random.default_rng(4)
    rng.normal(0.08, 0.03, size=3)
    true_vol = rng.uniform(0.15, 0.35, size=3)
    np.testing.assert_allclose(vol.to_numpy(), true_vol, rtol=0.01)
    assert lr.corr().iloc[0, 1] == pytest.approx(0.5, abs=0.01)


def test_mixture_em_recovers_parameters():
    rng = np.random.default_rng(5)
    n = 200_000
    k = rng.random(n) < 0.2
    x = np.where(k, rng.normal(-0.01, 0.03, n), rng.normal(0.001, 0.01, n))
    w, m, s = fit_normal_mixture(x, 2)
    np.testing.assert_allclose(w, [0.8, 0.2], atol=0.01)
    np.testing.assert_allclose(s, [0.01, 0.03], rtol=0.03)
    np.testing.assert_allclose(m, [0.001, -0.01], atol=0.001)


def test_performance_summary_constant_return():
    r = pd.Series([0.001] * 252 + [0.0005] * 252)
    out = performance_summary(r)
    assert out["max_drawdown"] == 0.0
    expected = ((1.001**252) * (1.0005**252)) ** 0.5 - 1
    assert out["ann_return"] == pytest.approx(expected)
