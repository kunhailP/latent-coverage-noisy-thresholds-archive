"""E43: rules with the same information (Supplementary Material, S9; Table 2).

No rule is told the noise law or the noise variances: each is the noisy conformal threshold
T = |V|_(k) at the rank of Theorem 2 for a stated noise class, at q = 0.9, delta = 0.05. The
ranks differ only through the noisy level they need:

  usual           0.9                 no latent guarantee
  Gaussian        0.901               bi-log-concave latent law, Gaussian noise
  symm. unimodal  Psi_SU(0.9)         bi-log-concave latent law, symmetric unimodal noise
  log-concave     0.906               bi-log-concave latent law, mean-zero log-concave noise
  shape-free      0.95                any latent law, median-zero noise
  shape-free LC   1 - 0.1/e           any latent law, mean-zero log-concave noise

Latent residuals have variance 1; noise variances D_i = 0.577 L_i / E(L_i), L_i lognormal with
log-scale 0.7, as in Supplementary Table S5, and the noise law is one of Gaussian, uniform, Laplace, t_3 (all
symmetric unimodal) or the centred exponential (mean-zero log-concave, skewed), scaled to
variance D_i. The latent coverage given the data, P(|W| <= T), is computed exactly from the
latent distribution function; reliability is the fraction of data sets with coverage >= 0.9.
Widths are 2T divided by the oracle width 2 Q_0.9(|W|).

  python experiments/e43_same_information.py   -> results/same_information.csv
"""
import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

from _common import RESULTS

Q, DELTA = 0.9, 0.05
LEVELS = {'usual': 0.9, 'Gaussian': 0.901, 'symm. unimodal': 0.9017873124108066,
          'log-concave': 0.906, 'shape-free': 0.95, 'shape-free LC': 1 - 0.1 / math.e}
# which noise laws each rule covers (bi-log-concave latent law for the first four)
COVERS = {'usual': (), 'Gaussian': ('Gaussian',),
          'symm. unimodal': ('Gaussian', 'uniform', 'Laplace', 't3'),
          'log-concave': ('Gaussian', 'uniform', 'Laplace', 'centred exp'),
          'shape-free': ('Gaussian', 'uniform', 'Laplace', 't3'),
          'shape-free LC': ('Gaussian', 'uniform', 'Laplace', 'centred exp')}


def rank(K, p):
    ks = np.arange(1, K + 1)
    ok = stats.binom.cdf(ks - 1, K, p) >= 1 - DELTA
    return int(ks[np.argmax(ok)]) if ok.any() else None


class TruncExp:
    """Density proportional to e^{4w} on [0, 1], standardized."""
    b = 4.0
    m = (math.exp(b) * (b - 1) + 1) / (b * (math.exp(b) - 1))
    s = math.sqrt((math.exp(b) * (b * b - 2 * b + 2) - 2) / (b * b * (math.exp(b) - 1)) - m * m)

    def rvs(self, size, random_state):
        u = random_state.random(size)
        x = np.log1p(u * (math.exp(self.b) - 1)) / self.b
        return (x - self.m) / self.s

    def cdf(self, w):
        x = np.clip(np.asarray(w) * self.s + self.m, 0, 1)
        return (np.exp(self.b * x) - 1) / (math.exp(self.b) - 1)


LATENT = {'normal': stats.norm(), 'Laplace': stats.laplace(scale=1 / math.sqrt(2)),
          'centred gamma': stats.gamma(2, loc=-math.sqrt(2), scale=1 / math.sqrt(2)),
          'truncated exponential': TruncExp()}


def noise(name, D, rng):
    sd = np.sqrt(D)
    n = len(D)
    if name == 'Gaussian':
        return sd * rng.standard_normal(n)
    if name == 'uniform':
        return sd * math.sqrt(3) * rng.uniform(-1, 1, n)
    if name == 'Laplace':
        return sd / math.sqrt(2) * rng.laplace(size=n)
    if name == 't3':
        return sd / math.sqrt(3) * rng.standard_t(3, n)
    if name == 'centred exp':
        return sd * (rng.exponential(size=n) - 1)
    raise ValueError(name)


def oracle_radius(law):
    return brentq(lambda r: law.cdf(r) - law.cdf(-r) - Q, 1e-6, 20)


def run(K, reps=2000, seed=43):
    rng = np.random.default_rng(seed)
    ranks = {r: rank(K, p) for r, p in LEVELS.items()}
    rows = []
    for lname, law in LATENT.items():
        r_or = oracle_radius(law)
        for nname in ('Gaussian', 'uniform', 'Laplace', 't3', 'centred exp'):
            cov = {r: np.empty(reps) for r in ranks}
            wid = {r: np.empty(reps) for r in ranks}
            for b in range(reps):
                L = rng.lognormal(0, 0.7, K)
                D = 0.577 * L / math.exp(0.7 ** 2 / 2)
                W = law.rvs(size=K, random_state=rng)
                V = np.sort(np.abs(W + noise(nname, D, rng)))
                for r, k in ranks.items():
                    if k is None:
                        cov[r][b] = wid[r][b] = np.nan
                        continue
                    T = V[k - 1]
                    cov[r][b] = law.cdf(T) - law.cdf(-T)
                    wid[r][b] = T / r_or
            for r, k in ranks.items():
                rows.append(dict(K=K, latent=lname, noise=nname, rule=r, rank=k,
                                 guaranteed=nname in COVERS[r],
                                 reliability=np.nanmean(cov[r] >= Q) if k else np.nan,
                                 mean_coverage=np.nanmean(cov[r]) if k else np.nan,
                                 rel_width=np.nanmean(wid[r]) if k else np.nan))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = pd.concat([run(110), run(1000)], ignore_index=True)
    df.to_csv(RESULTS / 'same_information.csv', index=False)
    pd.set_option('display.width', 200)
    piv = df.pivot_table(index=['K', 'rule', 'rank'], values=['reliability', 'rel_width'],
                         aggfunc=['min', 'max'])
    print(piv.round(3))
