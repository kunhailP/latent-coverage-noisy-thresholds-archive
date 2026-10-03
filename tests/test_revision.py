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
    src = path.read_text().split('fam = {')[0]
    ns = {}
    exec(compile(src, str(path), 'exec'), ns)
    return ns


def test_unimodal_latent_reaches_shape_free_bound():
    ns = _load_e42()
    m, s, A = ns['worst'](0.9, lambda sg: stats.norm(0, sg), 0.01, delta=1e-5, tau=1e-4)
    pieces, m2, _ = ns['build'](s, 1e-4, 1e-5)
    assert ns['noisy_cov'](pieces, stats.norm(0, 0.01)) == pytest.approx(0.9, abs=1e-8)
    assert 0.8 < m < 0.81                       # latent coverage close to 2p - 1
    mean = sum(mass * (a + b) / 2 for a, b, mass in pieces)
    assert abs(mean) < 1e-9


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
