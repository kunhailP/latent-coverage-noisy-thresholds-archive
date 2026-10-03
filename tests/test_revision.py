"""Checks for the results added in the revision: exact two-sided level for symmetric unimodal
noise, unimodal latent laws, and the binomial form of the smallest valid rank."""
import importlib.util
import math
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
from scipy import stats

from uai.noise_classes import psi_su_enclosure
from uai.procedures import certified_rank

mp.mp.dps = 40


def _su_parts(q):
    q = mp.mpf(q)
    ell = mp.log(q)
    d = lambda r: q * mp.e ** (-r) * (1 + r) - (1 + ell)
    r = mp.findroot(d, (-ell, 1 / (1 + ell) - 1), solver='anderson')
    s = q * mp.e ** (-r)
    return r, s, (1 + s) / 2


@pytest.mark.parametrize('q', ['0.45', '0.8', '0.9', '0.95', '0.999'])
def test_su_second_derivative_closed_form_and_sign(q):
    r, s, _ = _su_parts(q)
    qq = mp.mpf(q)
    closed = (1 - s) * (1 - s * (1 + r) ** 2) / (2 * qq ** 2 * r ** 3 * s)
    numeric = mp.diff(lambda x: _su_parts(x)[2], qq, 2)
    assert closed > 0
    assert abs(closed - numeric) < 1e-8 * abs(closed)


def test_su_key_inequality():
    # k(x) = 2 log x - x + 1/x > 0 on (0, 1): the step that makes Psi convex
    # near x = 1 it is of order (1 - x)^3, below double precision, so evaluate in mpmath
    for x in np.linspace(1e-6, 1 - 1e-6, 2001):
        x = mp.mpf(float(x))
        assert 2 * mp.log(x) - x + 1 / x > 0


def test_su_exact_level_at_09():
    lo, hi = psi_su_enclosure('0.9')
    assert 0.90178731 < lo <= hi < 0.90178732


def _load_e42():
    path = Path(__file__).resolve().parents[1] / 'experiments' / 'e42_unimodal_boundary.py'
    spec = importlib.util.spec_from_file_location('e42', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_e42_closed_form_matches_quadrature():
    e = _load_e42()
    pieces = e.build(0.2, 1e-4, 1e-5)[0]
    for G, F in ((e.G_gauss, lambda y: stats.norm.cdf(y / 0.01)),
                 (e.G_cexp, lambda y: 0.0 if y < -0.01 else 1 - math.exp(-(y / 0.01 + 1)))):
        closed = e.noisy_cov(pieces, G, 0.01)
        quad = 0.0
        for a, b, mass in pieces:
            pts = sorted({a, b, *[x for x in (1 - 0.05, 1 + 0.05, -1 - 0.05, -1 + 0.05, 1.01, 0.99, -0.99, -1.01, -1.5, -3) if a < x < b]})
            quad += mass / (b - a) * float(mp.quad(lambda w: F(float(1 - w)) - F(float(-1 - w)), pts))
        assert closed == pytest.approx(quad, abs=1e-9)


@pytest.mark.parametrize('name, limit', [('Gaussian', 0.8), ('uniform', 0.8),
                                         ('centred exponential', 1 - math.e / 10)])
def test_unimodal_latent_reaches_shape_free_bound(name, limit):
    e = _load_e42()
    m, s, A, nc = e.worst(0.9, e.NOISE[name], 0.001, delta=1e-6, tau=1e-4)
    assert nc == pytest.approx(0.9, abs=1e-12)
    assert limit < m < limit + 0.002
    pieces = e.build(s, 1e-4, 1e-6)[0]
    assert abs(sum(mass * (a + b) / 2 for a, b, mass in pieces)) < 1e-9


def _binom_rank(K, p, delta=0.05):
    for k in range(1, K + 1):
        if stats.binom.cdf(k - 1, K, p) >= 1 - delta:
            return k
    return None


@pytest.mark.parametrize('K', [29, 46, 110, 300, 1000])
def test_binomial_rank_matches_certified_rank(K):
    assert _binom_rank(K, 0.901) == certified_rank(K, 0.9, 0.05)


def test_rank_existence_thresholds():
    first = lambda p: next(K for K in range(1, 200) if 1 - p ** K >= 0.95)
    assert first(0.9017873124) == 29
    assert first(0.906) == 31
    assert first(0.95) == 59
