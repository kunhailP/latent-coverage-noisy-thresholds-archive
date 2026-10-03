"""E30. Area-wise estimated variances without pooling: model H (Supplementary Material, Section S6).

Heterogeneous within-area variances (as E29 n23_het: sigma_i^2 lognormal(0, .5), n_i on {2, 3},
mean D = .577), but every area's variance is estimated from nu i.i.d. normal units of its own,
D_hat_i = D_i chi2_nu / nu, for nu in {2, 10, 50, 200, 1000}. Rule H: order-statistic threshold
at delta = .04 and the envelope kernel of `uai.estimated.h_kernel` (area-wise intervals of level
1 - alpha, at most N* misses at level eta = .01), radius T R[u] from `radius_generic`
(exploratory grid values). Compared with noisy conformal (CP, same T) and the known-D oracle at
the same level. The question is how many degrees of freedom per area the guarantee needs before
it beats noisy conformal.
  python experiments/e30_areawise_variance.py [reps] [procs]
Writes results/areawise_variance.csv and results/areawise_variance_summary.csv.
"""
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from _common import RESULTS
from e12_conditional_synth import draw
from uai.estimated import h_kernel, radius_generic
from uai.latent_laws import cdf
from uai.procedures import hetldc_parts, pac_rank

warnings.filterwarnings('ignore')
K, DBAR, LEV, DELTA = 110, 0.577, 0.90, 0.04
SHAPES = ['normal', 'laplace', 'trunc_laplace']
NUS = [2, 10, 50, 200, 1000]
ALPHAS = [9e-5, 1e-3, 5e-3]


def one(args):
    shape, seed = args
    rng = np.random.default_rng(seed)
    n = rng.integers(2, 4, K); L = rng.lognormal(0, .5, K); L /= L.mean()
    a = 1.0 / n; D = DBAR / np.mean(a * L) * L * a
    V = draw(shape, K, rng) + rng.normal(0, np.sqrt(D))
    k = pac_rank(K, LEV, DELTA); T = np.sort(np.abs(V))[k - 1]
    p_k = stats.beta.ppf(DELTA, k, K + 1 - k)
    orc = hetldc_parts(V, D, k, delta=DELTA)
    rows = [dict(shape=shape, seed=seed, rule='CP', nu=np.nan, alpha=np.nan, half=T),
            dict(shape=shape, seed=seed, rule='Oracle', nu=np.nan, alpha=np.nan, half=orc[0] * orc[4])]
    for nu in NUS:
        Dh = D * rng.chisquare(nu, K) / nu
        for al in ALPHAS:
            w, u, _ = h_kernel(Dh, nu, T, alpha=al, eta=.01)
            rows.append(dict(shape=shape, seed=seed, rule='H', nu=nu, alpha=al,
                             half=T * radius_generic(p_k, LEV, w, u)))
    for r in rows:
        r['cov'] = float(cdf(shape, r['half']) - cdf(shape, -r['half']))
    return rows


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    jobs = [(s, 300000 + 1000 * j + r) for j, s in enumerate(SHAPES) for r in range(reps)]
    with Pool(procs) as pool:
        r = pd.DataFrame([row for rows in pool.imap_unordered(one, jobs) for row in rows])
    r.to_csv(RESULTS / 'areawise_variance.csv', index=False)
    cp = r[r.rule == 'CP'].set_index(['shape', 'seed']).half
    r['rel_cp'] = r.half.values / cp.loc[list(zip(r['shape'], r.seed))].values - 1
    s = r.groupby(['rule', 'nu', 'alpha'], dropna=False).agg(
        n=('half', 'size'), rel_to_cp=('rel_cp', 'mean'),
        pr_cov_ge_90=('cov', lambda c: np.mean(c >= LEV))).reset_index()
    s.to_csv(RESULTS / 'areawise_variance_summary.csv', index=False)
    print(s.round(4).to_string(index=False))
