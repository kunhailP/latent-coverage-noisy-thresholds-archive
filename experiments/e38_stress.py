"""E38: stress test at the extremal boundary-layer law (Supplementary Material, S6).

The latent residual W = A - s E, E ~ Exp(u), with s = x^{1/2} and u = 0.0508, the minimizer of
latent coverage in the small-noise limit at q = 0.9; A solves the population noisy coverage
pr(|W + e| <= 1) = 0.9, averaged over the noise variances. For calibration sizes K, data sets
are drawn and the exact conditional latent coverage pr(|W_new| <= T | data) is recorded for the
noisy threshold T at three ranks: the marginal split-conformal rank ceil(0.9 (K + 1)), the usual
high-probability rank (p_k >= 0.9) and the certified rank of Proposition 3 (p_k >= 0.901 and
k >= K p_k + 1), delta = 0.05.

  python experiments/e38_stress.py
Writes results/stress_summary.csv and the per-data-set results/stress.csv.gz (not tracked).
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr

from _common import RESULTS
from uai.procedures import certified_rank, pac_rank

U, XBAR, Q = 0.0508, 1e-4, 0.90
S = np.sqrt(XBAR)
LAM = U / S                                       # rate of s E


def noisy_cdf(c, A, D):
    """pr(A - sE + e <= c), e ~ N(0, D) (vectorized over D)."""
    sd = np.sqrt(D)
    z = c - A
    return ndtr(z / sd) + np.exp(LAM * z + LAM**2 * D / 2 + log_ndtr(-z / sd - LAM * sd))


def noisy_cov(r, A, D):
    return np.mean(noisy_cdf(r, A, D) - noisy_cdf(-r, A, D))


def latent_cdf(w, A):
    return np.exp(-LAM * np.maximum(A - w, 0))


def latent_cov(r, A):
    return latent_cdf(r, A) - latent_cdf(-r, A)


def variances(design, n, rng):
    if design == 'homogeneous':
        return np.full(n, XBAR)
    w = rng.lognormal(0, .7, n)
    return XBAR * w / np.exp(.7**2 / 2)


def run(design, K, reps, seed):
    rng = np.random.default_rng(seed)
    Dpop = variances(design, 200000, np.random.default_rng(1))
    A = brentq(lambda a: noisy_cov(1.0, a, Dpop) - Q, 0.5, 1.5, xtol=1e-14)
    ranks = {'marginal': int(np.ceil(Q * (K + 1))), 'usual_PAC': pac_rank(K, Q, .05),
             'certified': certified_rank(K, Q, .05)}
    rows = []
    for r in range(reps):
        D = variances(design, K, rng)
        V = A - S * rng.exponential(1 / U, K) + rng.normal(0, np.sqrt(D))
        absV = np.sort(np.abs(V))
        for rule, k in ranks.items():
            T = absV[k - 1]
            rows.append(dict(design=design, K=K, rep=r, rule=rule, rank=k, T=T,
                             latent=latent_cov(T, A), noisy=noisy_cov(T, A, D)))
    return A, latent_cov(1.0, A), rows


if __name__ == '__main__':
    out, summ = [], []
    for design in ('homogeneous', 'heterogeneous'):
        for K, reps in ((110, 20000), (1000, 20000), (10000, 10000)):
            A, pop_latent, rows = run(design, K, reps, seed=K + (design == 'heterogeneous'))
            df = pd.DataFrame(rows)
            out.append(df)
            for rule, g in df.groupby('rule', sort=False):
                rel = (g.latent >= Q).mean()
                summ.append(dict(design=design, K=K, rule=rule, rank=int(g['rank'].iloc[0]),
                                 pop_latent_at_noisy_q=pop_latent, mean_latent=g.latent.mean(),
                                 se_mean=g.latent.std() / np.sqrt(len(g)), reliability=rel,
                                 se_rel=np.sqrt(rel * (1 - rel) / len(g)),
                                 noisy_reliability=(g.noisy >= Q).mean(), reps=len(g)))
                print(summ[-1], flush=True)
    pd.concat(out).to_csv(RESULTS / 'stress.csv.gz', index=False)
    pd.DataFrame(summ).to_csv(RESULTS / 'stress_summary.csv', index=False)
