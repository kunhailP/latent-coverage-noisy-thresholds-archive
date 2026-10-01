"""E39: sensitivity of the analytic rules to the assumption that the latent residuals share one
law G, independent of the noise variances (Supplementary Material, S6).

Part 1, the design of Table 1 (K = 110, mean D = 0.577, lognormal spread 0.7), with
  scale_up / scale_down  W_i = sigma(D_i) X_i, sigma(d) proportional to d^{+-1/2}, var W = 1:
                         residual dispersion rises or falls with the noise variance; the new
                         area draws (D_new, W_new) from the same population;
  shift_+-0.2, +-0.5     G for the calibration areas, the new area's residual shifted by that
                         many latent standard deviations (a systematically different area).
Latent coverage of the new area is computed from closed-form distribution functions, averaged
over D_new and also conditional on D_new at its 10% and 90% quantiles. Rules: noisy threshold at
the certified rank (Proposition 3) and Corollary 2.

Part 2, the boundary-layer law of E38 (K = 110, equal variances x = 1e-4), with the new area
shifted by +-0.1 and +-0.25 latent standard deviations; noisy threshold at the certified rank.

  python experiments/e39_sensitivity.py [reps]
Writes results/sensitivity_summary.csv and results/sensitivity_stress.csv.
"""
import sys
import zlib

import numpy as np
import pandas as pd

from _common import RESULTS
from e12_conditional_synth import draw
from e38_stress import LAM, Q, S, U, latent_cdf, noisy_cov
from uai.latent_laws import cdf
from uai.procedures import certified_rank, noisy_threshold_halfwidth, simple_shrink_halfwidth
from scipy.optimize import brentq

K, DBAR = 110, 0.577
SHAPES = ['normal', 'gamma2_skew', 'trunc_exp']
DESIGNS = {'baseline': (0, 0.0), 'scale_up': (0.5, 0.0), 'scale_down': (-0.5, 0.0),
           'shift_+0.2': (0, 0.2), 'shift_-0.2': (0, -0.2), 'shift_+0.5': (0, 0.5),
           'shift_-0.5': (0, -0.5)}
DPOP = DBAR * np.random.default_rng(7).lognormal(0, .7, 4000) / np.exp(.7**2 / 2)
DQ = {'D10': np.quantile(DPOP, .1), 'D90': np.quantile(DPOP, .9)}


def sigma(d, g):
    return (d / DBAR)**g / np.sqrt(np.mean((DPOP / DBAR)**(2 * g)))


def coverage(shape, r, g, shift, d=DPOP):
    s = sigma(np.atleast_1d(d), g)
    return float(np.mean(cdf(shape, (r - shift) / s) - cdf(shape, (-r - shift) / s)))


def part1(reps):
    rows = []
    for shape in SHAPES:
        for name, (g, shift) in DESIGNS.items():
            rng = np.random.default_rng(zlib.crc32(f"{shape}-{name}".encode()))
            cov = {(r, c): [] for r in ('noisy_threshold', 'corollary_2') for c in ('mean', 'D10', 'D90')}
            for _ in range(reps):
                D = DBAR * rng.lognormal(0, .7, K) / np.exp(.7**2 / 2)
                V = sigma(D, g) * draw(shape, K, rng) + rng.normal(0, np.sqrt(D))
                for rule, h in (('noisy_threshold', noisy_threshold_halfwidth(V)),
                                ('corollary_2', simple_shrink_halfwidth(V, D.min()))):
                    cov[rule, 'mean'].append(coverage(shape, h, g, shift))
                    for c, d in DQ.items():
                        cov[rule, c].append(coverage(shape, h, g, shift, d))
            for (rule, cond), c in cov.items():
                c = np.array(c)
                rows.append(dict(shape=shape, design=name, rule=rule, d_new=cond, mean_cov=c.mean(),
                                 reliability=(c >= Q).mean(), q05=np.quantile(c, .05), reps=reps))
            print(shape, name, flush=True)
    return pd.DataFrame(rows)


def part2(reps):
    D = np.full(K, S**2)
    A = brentq(lambda a: noisy_cov(1.0, a, D) - Q, 0.5, 1.5, xtol=1e-14)
    sd = 1 / LAM                                    # latent standard deviation of A - sE
    k = certified_rank(K, Q, .05)
    rng = np.random.default_rng(39)
    Ts = np.array([np.sort(np.abs(A - S * rng.exponential(1 / U, K) + rng.normal(0, S, K)))[k - 1]
                   for _ in range(reps)])
    rows = []
    for shift in (-0.25, -0.1, 0.0, 0.1, 0.25):
        a = A + shift * sd
        c = latent_cdf(Ts, a) - latent_cdf(-Ts, a)
        rows.append(dict(shift_sd=shift, mean_cov=c.mean(), reliability=(c >= Q).mean(),
                         se_rel=np.sqrt((c >= Q).mean() * (1 - (c >= Q).mean()) / reps), reps=reps))
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    part2(20000).to_csv(RESULTS / 'sensitivity_stress.csv', index=False)
    part1(reps).to_csv(RESULTS / 'sensitivity_summary.csv', index=False)
