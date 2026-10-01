"""E29. Estimated noise variances with a guarantee: rule S (Supplementary Material, Section S4).

Areas have n_i sampled units; unit values are normal with within-area variance sigma_i^2, so the
direct estimate has noise N(0, D_i), D_i = sigma_i^2 / n_i, and the within-area sample variance
s_i^2 ~ sigma_i^2 chi2_{n_i - 1} / (n_i - 1) is independent of it. Model S assumes sigma_i^2 =
sigma^2 for all i and pools: nu = sum(n_i - 1), sigma_hat^2 = sum (n_i - 1) s_i^2 / nu.
Settings (mean D scaled to .577 as in E20, var W = 1, K = 110):
  n23        n_i uniform on {2, 3}, common sigma^2 (model S holds; nu ~ 165)
  n510       n_i uniform on {5, ..., 10}, common sigma^2 (nu ~ 740)
  n23_het    n_i on {2, 3}, sigma_i^2 = sigma^2 L_i, L_i lognormal(0, .5) normalised (model S wrong)
Rules (target Pr[cov >= .90] >= .95):
  CP_PAC      noisy conformal, k = pac_rank(110, .9, .05)
  Oracle      HetLDC with the true D_i (delta = .05)
  Oracle04    the same at delta = .04, the level S uses: S - Oracle04 is the cost of the
              variance uncertainty alone
  Plug        HetLDC with sigma_hat^2 a_i used as known (no guarantee)
  Dlow        HetLDC with lo * a_i, lo the lower end of the scale interval (no guarantee: R is
              not monotone in the noise level)
  S           sup of R^mix over the scale interval, delta = .04, eta = .01 (guarantee under S)
  S_cert      rule S with certified upper bounds U_j at every grid point (the form Supplementary Lemma S2
              requires; branch and bound in double precision with a margin, `uai.certify`)
Radii of the other rules are converged grid values (not certified), as HetLDC in E20.
  python experiments/e29_estimated_scale.py [reps] [procs] [cert]
With `cert`, only S_cert is computed for the same seeds and merged into the existing CSV.
Writes results/estimated_scale.csv and results/estimated_scale_summary.csv.
"""
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from _common import RESULTS
from e12_conditional_synth import draw
from uai.estimated import radius_at, s_rule, scale_interval
from uai.latent_laws import cdf
from uai.procedures import conformal_threshold, hetldc_parts, pac_rank

warnings.filterwarnings('ignore')
K, DBAR, LEV = 110, 0.577, 0.90
SHAPES = ['normal', 'laplace', 'trunc_laplace']
SETTINGS = ['n23', 'n510', 'n23_het']


def data(setting, shape, rng):
    n = rng.integers(2, 4, K) if setting.startswith('n23') else rng.integers(5, 11, K)
    a = 1.0 / n
    L = np.ones(K)
    if setting.endswith('het'):
        L = rng.lognormal(0, .5, K); L /= L.mean()
    sig2 = DBAR / np.mean(a * L)                   # mean D = .577
    D = sig2 * L * a
    V = draw(shape, K, rng) + rng.normal(0, np.sqrt(D))
    s2 = sig2 * L * rng.chisquare(n - 1) / (n - 1)
    nu = int(np.sum(n - 1))
    return V, D, a, float(np.sum((n - 1) * s2) / nu), nu


def one(args):
    setting, shape, seed = args
    rng = np.random.default_rng(seed)
    V, D, a, s2_hat, nu = data(setting, shape, rng)
    k5 = pac_rank(K, LEV, .05); T5 = conformal_threshold(V, k5)
    orc = hetldc_parts(V, D, k5)
    orc4 = hetldc_parts(V, D, pac_rank(K, LEV, .04), delta=.04)
    plug = hetldc_parts(V, s2_hat * a, k5)
    hS, info = s_rule(V, a, s2_hat, nu, q=LEV, delta=.04, eta_lo=.009, eta_hi=.001)
    lo, _ = scale_interval(s2_hat, nu, .009, .001)
    Rlow = radius_at(info['p_k'], LEV, lo * a / info['T']**2)[0]
    halves = {'CP_PAC': T5, 'Oracle': orc[0] * orc[4], 'Oracle04': orc4[0] * orc4[4], 'Plug': plug[0] * plug[4],
              'Dlow': info['T'] * Rlow, 'S': hS}
    return [dict(setting=setting, shape=shape, seed=seed, method=m, half=h, cells=info['cells'],
                 lo_ratio=lo / (DBAR / np.mean(a)) if not setting.endswith('het') else np.nan,
                 cov=float(cdf(shape, h) - cdf(shape, -h))) for m, h in halves.items()]


def one_cert(args):
    """S_cert only, from the same data as `one`."""
    import time
    setting, shape, seed = args
    rng = np.random.default_rng(seed)
    V, D, a, s2_hat, nu = data(setting, shape, rng)
    t0 = time.time()
    h, info = s_rule(V, a, s2_hat, nu, q=LEV, delta=.04, eta_lo=.009, eta_hi=.001, certify=True)
    return [dict(setting=setting, shape=shape, seed=seed, method='S_cert', half=h,
                 cells=info['cells'], lo_ratio=np.nan, seconds=time.time() - t0,
                 cov=float(cdf(shape, h) - cdf(shape, -h)))]


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    cert = len(sys.argv) > 3 and sys.argv[3] == 'cert'
    jobs = [(g, s, 290000 + 10000 * i + 1000 * j + r) for i, g in enumerate(SETTINGS)
            for j, s in enumerate(SHAPES) for r in range(reps)]
    rows = []
    with Pool(procs) as pool:
        for i, out in enumerate(pool.imap_unordered(one_cert if cert else one, jobs)):
            rows += out
            if i % 25 == 0:
                print(f'{i + 1}/{len(jobs)} done', flush=True)
    r = pd.DataFrame(rows)
    if cert:
        old = pd.read_csv(RESULTS / 'estimated_scale.csv')
        r = pd.concat([old[old.method != 'S_cert'], r])
    r.to_csv(RESULTS / 'estimated_scale.csv', index=False)
    s = r.groupby(['setting', 'shape', 'method']).agg(
        reps=('cov', 'size'), mean_cov=('cov', 'mean'),
        pr_cov_ge_90=('cov', lambda c: np.mean(c >= LEV)),
        width=('half', lambda h: 2 * h.mean())).reset_index()
    s.to_csv(RESULTS / 'estimated_scale_summary.csv', index=False)
    print(s.round(4).to_string(index=False))
