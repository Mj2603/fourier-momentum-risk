"""Cross-sectional momentum backtest with characteristic-function (COS) VaR/CVaR."""

from .data import simulate_prices
from .momentum import momentum_signal, top_quantile_weights
from .backtest import backtest, performance_summary
from .fourier_risk import (
    cf_normal,
    cf_normal_mixture,
    cos_cdf,
    cos_var_cvar,
    normal_var_cvar,
    truncation_range,
)
from .mixture import fit_normal_mixture

__all__ = [
    "simulate_prices",
    "momentum_signal",
    "top_quantile_weights",
    "backtest",
    "performance_summary",
    "cf_normal",
    "cf_normal_mixture",
    "cos_cdf",
    "cos_var_cvar",
    "normal_var_cvar",
    "truncation_range",
    "fit_normal_mixture",
]
