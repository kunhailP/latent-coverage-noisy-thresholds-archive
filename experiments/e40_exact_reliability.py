"""E40: exact latent reliability of the noisy threshold at the boundary-layer law of E38
(equal variances, x = 1e-4), Proposition 4 and Supplementary Material, S6.

With r_q = Q_q(|W|) and H the distribution function of |V|, the latent coverage at T = |V|_(k)
is at least q iff T >= r_q, iff fewer than k of the |V_i| fall below r_q, so the latent
reliability is pr{Bin(K, H(r_q)) <= k - 1}, with no Monte Carlo error. Ranks: marginal
ceil(q (K + 1)), usual high-probability (p_k >= q), certified (Proposition 3), delta = 0.05.

  python experiments/e40_exact_reliability.py   -> results/exact_reliability.csv,
                                                    paper/fig_reliability.pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

from _common import RESULTS, ROOT
from e38_stress import Q, S, latent_cov, noisy_cov
from uai.procedures import CERTIFIED_SLACK_LEVEL

DELTA = 0.05


def first_rank(K, level, hoeffding=False):
    ks = np.arange(1, K + 1)
    p = stats.beta.ppf(DELTA, ks, K + 1 - ks)
    ok = p >= level
    if hoeffding:
        ok &= ks >= K * p + 1
    return int(ks[np.argmax(ok)]) if ok.any() else None


if __name__ == '__main__':
    D = np.full(1, S**2)
    A = brentq(lambda a: noisy_cov(1.0, a, D) - Q, 0.5, 1.5, xtol=1e-14)
    r_q = brentq(lambda t: latent_cov(t, A) - Q, 0.5, 1.5, xtol=1e-14)
    H = noisy_cov(r_q, A, D)
    print(f'r_q = {r_q:.7f}, H(r_q) = {H:.7f}')
    rows = []
    QUOTED = [110, 1000, 10000, 100000, 1000000]
    for K in np.unique(np.r_[np.round(np.geomspace(30, 1e6, 60)).astype(int), QUOTED]):
        ranks = {'marginal': int(np.ceil(Q * (K + 1))), 'usual': first_rank(K, Q),
                 'certified': first_rank(K, CERTIFIED_SLACK_LEVEL[Q], hoeffding=True)}
        for rule, k in ranks.items():
            if k is not None and k <= K:
                rows.append(dict(K=K, rule=rule, rank=k, reliability=stats.binom.cdf(k - 1, K, H)))
    small = [(K, stats.binom.cdf(first_rank(K, Q) - 1, K, H)) for K in range(29, 301)]
    K_min, rel_min = min(small, key=lambda t: t[1])
    print(f'usual rank, 29 <= K <= 300: min reliability {rel_min:.5f} at K = {K_min}')
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / 'exact_reliability.csv', index=False)
    print(df[df.K.isin(QUOTED)].to_string(index=False))
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    for rule, ls, lab in (('certified', '-', 'certified rank'), ('usual', '--', 'usual rank'),
                          ('marginal', ':', 'marginal rank')):
        g = df[df.rule == rule]
        ax.plot(g.K, g.reliability, 'k', ls=ls, lw=1.4, label=lab)
    ax.axhline(1 - DELTA, color='0.6', lw=0.9)
    ax.set_xscale('log'); ax.set_ylim(0, 1)
    ax.set_xlabel('Calibration sample size, $K$'); ax.set_ylabel('Latent reliability')
    ax.legend(frameon=False, loc='lower left')
    fig.tight_layout()
    fig.savefig(ROOT / 'paper' / 'fig_reliability.pdf')
