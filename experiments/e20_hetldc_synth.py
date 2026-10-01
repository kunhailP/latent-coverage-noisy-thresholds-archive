"""E20. Heterogeneous known noise: HetLDC against the rules of E12, same conditional target.

Setting as E12's heterogeneous regime (K = 110, Var W = 1, D_i = 0.577 lognormal(0, .7)/mean),
but D_i are given to every rule as known (the usual Fay-Herriot convention), so the only
difference between LDC_PAC and HetLDC is the noise kernel: one Gaussian with the mean variance
versus the exact average kernel. Target Pr_D[Pr(W_new in C | D) >= .90] >= .95; conditional
coverage from closed-form CDFs (src/uai/latent_laws.py).
Rules: FH_normal, FH_PAC (bootstrap tolerance), CP_PAC (k = 105), LDC_PAC (k = 105, mean-D
Gaussian kernel, E04 table), HetLDC (k = 105, exact average kernel, grid value), and with
certified radii (E24-E26): LDC_cert_pk (mean variance in the certified R_{.9068,.9} table,
the same level p_k = .9068 as HetLDC) and HetLDC_cert
(branch-and-bound certificate for the mixture kernel; union bound if none clears).
  python experiments/e20_hetldc_synth.py [reps] [procs] [ldc | shapes=a,b,...]
With a third argument `ldc`, only LDC_cert_pk is recomputed (same seeds) and patched into the
existing CSV; the other rules, including the slow HetLDC certificates, are kept. With
`shapes=a,b`, only those laws are (re)computed and merged into the existing CSV. Seeds depend
only on the position of a law in SHAPES, so adding laws leaves earlier results unchanged.
Writes results/hetldc_synth.csv (per replication) and results/hetldc_synth_summary.csv.
"""
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from _common import RESULTS
from e12_conditional_synth import draw
from uai.latent_laws import cdf
from uai.procedures import (CertifiedShrinkTable, ShrinkTable, conformal_threshold,
                            fay_herriot_boot_pac, fay_herriot_reml, hetldc_certified,
                            hetldc_parts, pac_rank)

warnings.filterwarnings('ignore')
K, DBAR, LEV = 110, 0.577, 0.90
SHAPES = ['normal', 'laplace', 'gamma2_skew', 'trunc_laplace',
          'trunc_exp', 'bimodal_blc', 't3_not_LC']   # log-affine extremal, bi-log-concave, outside
CERTIFY = True


def one(args):
    shape, seed = args
    rng = np.random.default_rng(seed); table = ShrinkTable('0.1')
    w = rng.lognormal(0, .7, K); D = DBAR * w / np.exp(.7**2 / 2)
    V = draw(shape, K, rng) + rng.normal(0, np.sqrt(D))
    kp = pac_rank(K, LEV, .05); T = conformal_threshold(V, kp)
    mu, A, vmu = fay_herriot_reml(V, D); se = np.sqrt(A + vmu)
    _, lam = fay_herriot_boot_pac(V, D, mu, A, rng, level=LEV)
    z = stats.norm.ppf(.5 + LEV / 2)
    parts = hetldc_parts(V, D, kp)
    het_cert, tol = hetldc_certified(V, D, kp, parts=parts) if CERTIFY else (np.nan, np.nan)
    ints = {'FH_normal': (mu, z * se), 'FH_PAC': (mu, lam * se), 'CP_PAC': (0, T),
            'LDC_PAC': (0, T * table(D.mean() / T**2)), 'HetLDC': (0, parts[0] * parts[4])}
    if CERTIFY:
        # certified table at p = .9068 <= p_k(delta = .05), the level HetLDC uses as well
        ints['LDC_cert_pk'] = (0, T * CertifiedShrinkTable(0.9068)(D.mean() / T**2))
        ints['HetLDC_cert'] = (0, het_cert)
    return [dict(shape=shape, seed=seed, method=m, centre=c, half=h, cert_tol=tol,
                 cov=float(cdf(shape, c + h) - cdf(shape, c - h))) for m, (c, h) in ints.items()]


def one_ldc(args):
    """LDC_cert_pk only, from the same data as `one` (the draws before the FH bootstrap)."""
    shape, seed = args
    rng = np.random.default_rng(seed)
    w = rng.lognormal(0, .7, K); D = DBAR * w / np.exp(.7**2 / 2)
    V = draw(shape, K, rng) + rng.normal(0, np.sqrt(D))
    T = conformal_threshold(V, pac_rank(K, LEV, .05))
    h = T * CertifiedShrinkTable(0.9068)(D.mean() / T**2)
    return dict(shape=shape, seed=seed, method='LDC_cert_pk', centre=0.0, half=h, cert_tol=np.nan,
                cov=float(cdf(shape, h) - cdf(shape, -h)))


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    jobs = [(s, 40000 + 1000 * i + r) for i, s in enumerate(SHAPES) for r in range(reps)]
    only = sys.argv[3][7:].split(',') if len(sys.argv) > 3 and sys.argv[3].startswith('shapes=') else None
    if only:
        jobs = [j for j in jobs if j[0] in only]
        with Pool(procs) as pool:
            new = pd.DataFrame([row for rows in pool.imap_unordered(one, jobs, chunksize=1) for row in rows])
        old = pd.read_csv(RESULTS / 'hetldc_synth.csv')
        r = pd.concat([old[~old['shape'].isin(only)], new.sort_values(['shape', 'seed', 'method'])])
    elif len(sys.argv) > 3 and sys.argv[3] == 'ldc':
        with Pool(procs) as pool:
            new = pd.DataFrame(pool.map(one_ldc, jobs, chunksize=4))
        old = pd.read_csv(RESULTS / 'hetldc_synth.csv')
        r = pd.concat([old[old.method != 'LDC_cert_pk'], new]).sort_values(['shape', 'seed', 'method'])
    else:
        with Pool(procs) as pool:
            r = pd.DataFrame([row for rows in pool.imap_unordered(one, jobs, chunksize=2) for row in rows])
    r.to_csv(RESULTS / 'hetldc_synth.csv', index=False)
    s = r.groupby(['shape', 'method']).agg(reps=('cov', 'size'), mean_cov=('cov', 'mean'),
                                          pr_cov_ge_90=('cov', lambda c: np.mean(c >= LEV)),
                                          q05_cov=('cov', lambda c: np.quantile(c, .05)),
                                          width=('half', lambda h: 2 * h.mean())).reset_index()
    s.to_csv(RESULTS / 'hetldc_synth_summary.csv', index=False)
    print(s.round(4).to_string(index=False))
