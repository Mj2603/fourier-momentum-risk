# Momentum Portfolio Strategies under Fourier Calculated Risk

This repo implements momentum strategies and estimates risk using Fourier-domain methods (characteristic functions, FFT-based VaR/CVaR) for scalable tail risk evaluation.

Inspired by: samlayton99/Quantitative_Finance (https://github.com/samlayton99/Quantitative_Finance)

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python examples/backtest.py
```

## Components
- src/data.py — synthetic price simulation and simple data loader
- src/momentum.py — cross-sectional momentum signals
- src/fourier_risk.py — FFT-based risk (VaR/CVaR) via characteristic functions
- examples/backtest.py — end-to-end monthly rebalanced backtest

Outputs saved to outputs/.