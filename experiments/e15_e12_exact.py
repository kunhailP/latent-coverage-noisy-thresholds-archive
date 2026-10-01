"""E15. E12 recomputed with analytic latent CDFs (no reference sample).

Reads data/conditional_synth_reps.pkl (per-replication centre and offsets from E12) and
recomputes the conditional coverage G(c + hi) - G(c + lo) exactly, then the oracle matching:
mean-matched scale by root finding, PAC-matched scale as the 95% quantile of the per-
replication scale that reaches coverage .90 (no root finding on a step function).
  python experiments/e15_e12_exact.py [procs]
Writes results/conditional_synth_exact_summary.csv, results/conditional_synth_exact_matched.csv.
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from _common import DATA, RESULTS
from uai.latent_laws import cdf, matched_scales

LEV = 0.90


def job(args):
    (shape, regime, method), g = args
    cov = cdf(shape, g.centre + g.hi) - cdf(shape, g.centre + g.lo)
    s = dict(shape=shape, regime=regime, method=method, reps=len(g), mean_cov=cov.mean(),
             sd_cov=cov.std(), pr_cov_ge_90=np.mean(cov >= LEV), q05_cov=np.quantile(cov, .05),
             width=(g.hi - g.lo).mean())
    m = dict(shape=shape, regime=regime, method=method,
             **matched_scales(shape, g.centre.values, g.lo.values, g.hi.values, LEV, .95))
    return s, m


if __name__ == '__main__':
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    r = pd.read_pickle(DATA / 'conditional_synth_reps.pkl')
    with Pool(procs) as pool:
        out = pool.map(job, list(r.groupby(['shape', 'regime', 'method'])))
    S = pd.DataFrame([a for a, _ in out]); M = pd.DataFrame([b for _, b in out])
    S.to_csv(RESULTS / 'conditional_synth_exact_summary.csv', index=False)
    M.to_csv(RESULTS / 'conditional_synth_exact_matched.csv', index=False)
    old = pd.read_csv(RESULTS / 'conditional_synth_summary.csv').merge(
        pd.read_csv(RESULTS / 'conditional_synth_matched.csv'), on=['shape', 'regime', 'method'])
    new = S.merge(M, on=['shape', 'regime', 'method'])
    d = new.merge(old, on=['shape', 'regime', 'method'], suffixes=('', '_mc'))
    for c in ['mean_cov', 'pr_cov_ge_90', 'width_mean_matched', 'width_pac_matched']:
        print(f'{c:20s} max |exact - MC reference| = {np.abs(d[c] - d[c + "_mc"]).max():.4f}')
