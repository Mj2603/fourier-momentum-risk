# fourier-momentum-risk

A small Python package that backtests a long-only cross-sectional momentum portfolio on
simulated prices and estimates its daily Value-at-Risk (VaR) and Conditional VaR (CVaR) by
inverting a characteristic function with the COS method (Fang and Oosterlee, 2008). The COS
implementation is checked against closed-form normal and normal-mixture results and against
Monte Carlo for a sum of independent mixtures. All data in this repository is SYNTHETIC.

## What is implemented

- `simulate_prices` (`data.py`): correlated geometric Brownian motions with an optional common
  two-state Markov volatility regime, which gives fat-tailed, clustered portfolio returns.
- `momentum_signal` (`momentum.py`): 12-1 momentum, `P[t-skip] / P[t-lookback] - 1`, using only
  data up to date t.
- `top_quantile_weights`: equal weights on the top 30% of assets by signal.
- `backtest` (`backtest.py`): weights decided at the close of day t earn returns from t+1;
  holdings drift between monthly rebalances; turnover is recorded; no transaction costs.
- `cos_var_cvar` (`fourier_risk.py`): VaR and CVaR from any characteristic function. The CDF and
  the partial expectation `E[X 1{X <= x}]` are written as cosine series whose coefficients come
  from the characteristic function; VaR is found by root finding on the CDF.
- `fit_normal_mixture` (`mixture.py`): EM fit of a Gaussian mixture, whose characteristic
  function is then passed to `cos_var_cvar`.
- Closed-form Gaussian and Gaussian-mixture VaR/CVaR and historical VaR/CVaR for comparison.

## How to run

```bash
git clone https://github.com/Mj2603/fourier-momentum-risk.git
cd fourier-momentum-risk
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[test]"
pytest                             # 24 tests
python experiments/run_experiment.py
```

The experiment writes `results/results.md` and CSV files (`cos_accuracy.csv`,
`backtest_performance.csv`, `var_backtest.csv`, `mixture_fit.csv`). It is seeded and
deterministic.

## Results (SYNTHETIC data, from `results/results.md`)

Simulation: 20 assets, 2,520 business days, seed 7, pairwise correlation 0.3, volatility
multiplied by 2.5 in the high regime. Momentum 252/21, top 30%, rebalanced every 21 days,
2,267 trading days after warm-up.

COS accuracy at alpha = 0.99 (absolute error versus closed form):

| Distribution | Terms | VaR error | CVaR error |
|---|---:|---:|---:|
| Normal(0.0005, 0.012) | 32 | 9.2e-07 | 1.7e-06 |
| Normal(0.0005, 0.012) | 64 | 9.0e-17 | 1.9e-16 |
| Mixture 0.85 N(0.0006, 0.008) + 0.15 N(-0.002, 0.025) | 64 | 9.9e-05 | 1.7e-05 |
| Mixture (same) | 128 | 7.4e-12 | 1.0e-13 |

Backtest:

| Portfolio | Ann. return | Ann. vol | Sharpe | Max drawdown | Turnover per rebalance |
|---|---:|---:|---:|---:|---:|
| Momentum top 30% | -5.82% | 18.77% | -0.23 | -63.14% | 0.48 |
| Equal weight | -1.95% | 17.20% | -0.03 | -46.05% | 0.05 |

Daily 99% VaR, estimated on the first 1,133 momentum returns and tested on the next 1,134
(expected 1% exceedances):

| Method | VaR | CVaR | Exceedances | Rate | Kupiec p-value |
|---|---:|---:|---:|---:|---:|
| Historical | 3.47% | 4.12% | 10 | 0.88% | 0.68 |
| Gaussian (closed form and COS, identical) | 2.77% | 3.18% | 18 | 1.59% | 0.067 |
| Two-component mixture via COS | 3.19% | 3.98% | 12 | 1.06% | 0.85 |

The 95% results are in `results/var_backtest.csv`.

## Interpretation and limitations

- The data is simulated. Asset drifts are random and constant, so there is no momentum effect
  to find; the negative backtest returns reflect volatility drag in the high-volatility regime
  and the tilt of momentum toward high-volatility assets. The backtest numbers say nothing
  about real markets.
- One seed and one parameter set; no robustness study.
- For the Gaussian and mixture models used here, closed-form VaR/CVaR exist, so the COS method
  adds no accuracy. It is useful when only the characteristic function is available, for
  example for sums of independent non-Gaussian components; the tests include such a case.
- The COS result depends on the truncation interval and the number of terms; the caller chooses
  both.
- VaR estimates are unconditional. With clustered volatility, a conditional model (for example
  GARCH) would be more appropriate; the Kupiec test checks only the exceedance count, not
  independence.
- No transaction costs, borrowing, or short positions.

## Project layout

```
src/fourier_momentum_risk/   data.py, momentum.py, backtest.py, fourier_risk.py, mixture.py
tests/                       pytest tests against closed forms, simulations and invariants
experiments/run_experiment.py
results/                     committed output of the experiment
```

## Reference

F. Fang and C. W. Oosterlee (2008). A novel pricing method for European options based on
Fourier-cosine series expansions. SIAM Journal on Scientific Computing 31(2), 826-848.

## License

MIT, see `LICENSE`.
