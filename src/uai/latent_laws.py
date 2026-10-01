"""Closed-form CDFs of the latent-score laws used in E12, and oracle width matching.

Each law is standardised as in experiments/e12_conditional_synth.py (Var W = 1 except the
shifted normal, which is N(0.3, 1)). Matching is a diagnostic that uses the true law; it is
not a usable interval rule.
"""
import numpy as np
from scipy import special, stats
from scipy.optimize import brentq

_TL_C = 2.705                                         # truncated Laplace: support [-c, c]
_TL_SD = np.sqrt((2 - np.exp(-_TL_C) * (_TL_C**2 + 2 * _TL_C + 2)) / (1 - np.exp(-_TL_C)))
_TE_B = 4.0                                           # truncated exponential exp(b y), y in [0,1]
_TE_MEAN = 1 / (1 - np.exp(-_TE_B)) - 1 / _TE_B
_TE_SD = np.sqrt(1 / _TE_B**2 - np.exp(_TE_B) / np.expm1(_TE_B)**2)


def _trunc_laplace(x):
    y = np.clip(np.abs(x) * _TL_SD, 0, _TL_C)
    half = -np.expm1(-y) / -np.expm1(-_TL_C) / 2           # P(0 < Y <= |y|)
    return np.where(x >= 0, 0.5 + half, 0.5 - half)


def _trunc_exp(x):
    y = np.clip(_TE_MEAN + x * _TE_SD, 0, 1)
    return np.expm1(_TE_B * y) / np.expm1(_TE_B)


def _bimodal_params(p=0.4, ratio=2.5):
    """Mixture p N(m1, s^2) + (1 - p) N(m2, s^2) with mean 0, variance 1 and separation
    (m2 - m1)/s = ratio. For p = 0.4 it is bimodal for ratio above about 2.443 and bi-log-concave
    up to about 2.59; it is not log-concave. Bi-log-concavity at ratio 2.5 is proved by
    uai.interval.bimodal_blc_certificate (Supplementary Lemma S1); the interior maxima of
    (f'F - f^2)/f^2 and {-f'(1 - F) - f^2}/f^2 are -0.141 and -0.864."""
    d = ratio / np.sqrt(1 + p * (1 - p) * ratio**2)          # m2 - m1 after standardisation
    s = d / ratio
    return p, -(1 - p) * d, p * d, s


BIMODAL = _bimodal_params()


def _bimodal(x):
    p, m1, m2, s = BIMODAL
    return p * special.ndtr((x - m1) / s) + (1 - p) * special.ndtr((x - m2) / s)


LAWS = {
    'normal': lambda x: special.ndtr(x),
    'shifted_normal': lambda x: special.ndtr(x - 0.3),
    'laplace': lambda x: stats.laplace.cdf(x * np.sqrt(2)),
    'gamma2_skew': lambda x: special.gammainc(2, np.maximum(np.sqrt(2) * x + 2, 0)),
    'trunc_laplace': _trunc_laplace,
    'trunc_exp': _trunc_exp,
    't3_not_LC': lambda x: stats.t.cdf(np.sqrt(3) * x, 3),
    'bimodal_blc': _bimodal,
}


def cdf(kind, x):
    return LAWS[kind](np.asarray(x, dtype=float))


def interval_mass(kind, centre, lo, hi, lam=1.0):
    return cdf(kind, centre + lam * hi) - cdf(kind, centre + lam * lo)


def matched_scales(kind, centre, lo, hi, level=0.90, reliability=0.95):
    """Scale factors on the offsets (lo < 0 < hi) that reach (a) mean coverage = level and
    (b) coverage >= level in a `reliability` share of replications. For (b) the scale each
    replication needs is found separately and its empirical quantile taken."""
    centre, lo, hi = (np.asarray(a, dtype=float) for a in (centre, lo, hi))
    top = 1.0
    while interval_mass(kind, centre, lo, hi, top).min() < level:
        top *= 2
    lam_mean = brentq(lambda l: interval_mass(kind, centre, lo, hi, l).mean() - level, 0, top,
                      xtol=1e-12)
    need = np.array([brentq(lambda l: interval_mass(kind, c, a, b, l) - level, 0, top, xtol=1e-12)
                     for c, a, b in zip(centre, lo, hi)])
    lam_pac = np.sort(need)[int(np.ceil(reliability * len(need))) - 1]
    width = hi - lo
    return dict(lam_mean=lam_mean, width_mean_matched=float(np.mean(lam_mean * width)),
                lam_pac=float(lam_pac), width_pac_matched=float(np.mean(lam_pac * width)),
                achieved_reliability=float(np.mean(interval_mass(kind, centre, lo, hi, lam_pac)
                                                   >= level - 1e-12)))
