import numpy as np
import pytest
from scipy.stats import norm

from fourier_momentum_risk.fourier_risk import (
    cf_normal,
    cf_normal_mixture,
    cos_cdf,
    cos_pdf,
    cos_var_cvar,
    historical_var_cvar,
    normal_mixture_var_cvar,
    normal_var_cvar,
    truncation_range,
)


def test_normal_closed_form_known_values():
    var, cvar = normal_var_cvar(0.95, 0.0, 1.0)
    assert var == pytest.approx(1.6448536, abs=1e-6)
    assert cvar == pytest.approx(2.0627128, abs=1e-6)


@pytest.mark.parametrize("alpha", [0.95, 0.99, 0.999])
@pytest.mark.parametrize("mu,sigma", [(0.0, 1.0), (0.0005, 0.012), (-0.2, 3.0)])
def test_cos_matches_normal_closed_form(alpha, mu, sigma):
    a, b = truncation_range(mu, sigma)
    var, cvar = cos_var_cvar(lambda u: cf_normal(u, mu, sigma), alpha, a, b, n_terms=256)
    v0, c0 = normal_var_cvar(alpha, mu, sigma)
    assert var == pytest.approx(v0, rel=1e-8)
    assert cvar == pytest.approx(c0, rel=1e-8)


def test_cos_pdf_and_cdf_match_normal():
    a, b = truncation_range(0.0, 1.0)
    x = np.linspace(-4, 4, 41)
    cf = lambda u: cf_normal(u, 0.0, 1.0)
    np.testing.assert_allclose(cos_pdf(x, cf, a, b), norm.pdf(x), atol=1e-12)
    np.testing.assert_allclose(cos_cdf(x, cf, a, b), norm.cdf(x), atol=1e-12)


@pytest.mark.parametrize("alpha", [0.95, 0.99])
def test_cos_matches_mixture_closed_form(alpha):
    w, m, s = [0.85, 0.15], [0.0006, -0.002], [0.008, 0.025]
    mean = np.dot(w, m)
    std = np.sqrt(np.dot(w, np.square(s) + np.square(m)) - mean**2)
    a, b = truncation_range(mean, std, width=20)
    var, cvar = cos_var_cvar(lambda u: cf_normal_mixture(u, w, m, s), alpha, a, b, n_terms=1024)
    v0, c0 = normal_mixture_var_cvar(alpha, w, m, s)
    assert var == pytest.approx(v0, rel=1e-7)
    assert cvar == pytest.approx(c0, rel=1e-7)


def test_cos_on_convolution_matches_monte_carlo():
    # Sum of two independent mixtures: cf is the product, CDF has no simple closed form.
    w1, m1, s1 = [0.9, 0.1], [0.0, -0.01], [0.01, 0.04]
    w2, m2, s2 = [0.7, 0.3], [0.001, -0.002], [0.005, 0.015]
    cf = lambda u: cf_normal_mixture(u, w1, m1, s1) * cf_normal_mixture(u, w2, m2, s2)
    var, cvar = cos_var_cvar(cf, 0.99, -0.5, 0.5, n_terms=2048)

    rng = np.random.default_rng(0)
    n = 4_000_000
    def draw(w, m, s):
        k = rng.choice(len(w), size=n, p=w)
        return np.asarray(m)[k] + np.asarray(s)[k] * rng.standard_normal(n)
    x = draw(w1, m1, s1) + draw(w2, m2, s2)
    v_mc, c_mc = historical_var_cvar(x, 0.99)
    assert var == pytest.approx(v_mc, rel=5e-3)
    assert cvar == pytest.approx(c_mc, rel=5e-3)


def test_cvar_at_least_var_and_monotone_in_alpha():
    cf = lambda u: cf_normal_mixture(u, [0.8, 0.2], [0.0, 0.0], [1.0, 3.0])
    out = [cos_var_cvar(cf, a, -40, 40, 2048) for a in (0.9, 0.95, 0.99)]
    for v, c in out:
        assert c >= v
    assert out[0][0] < out[1][0] < out[2][0]


def test_historical_var_cvar_simple():
    r = np.arange(-50, 50) / 100.0  # 100 equally spaced returns
    var, cvar = historical_var_cvar(r, 0.95)
    assert var == pytest.approx(-np.quantile(r, 0.05))
    assert cvar == pytest.approx(-r[r <= np.quantile(r, 0.05)].mean())
