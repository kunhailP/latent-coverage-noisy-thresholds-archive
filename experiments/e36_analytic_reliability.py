"""E36: reliability of the cheap rules of Supplementary Table S5 with many data sets.

Same design as E20 (K = 110, var W = 1, D_i = 0.577 lognormal(0, 0.7)/mean, seven latent laws), but
2000 data sets per law and only the rules that need no numerical optimization: the noisy threshold
(Theorem 2), Supplementary Corollary S1 and the shape-free Markov rule at rank 107. Seeds differ from E20.
Reports reliability with a Clopper-Pearson 95% interval, mean conditional coverage and mean
full width.
  python experiments/e36_analytic_reliability.py [reps] [procs]
Writes results/analytic_reliability.csv and results/analytic_reliability_summary.csv.
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from _common import RESULTS
from e12_conditional_synth import draw
from e20_hetldc_synth import SHAPES
from uai.latent_laws import cdf
from uai.procedures import noisy_threshold_halfwidth, shape_free_markov, simple_shrink_halfwidth

K, DBAR, LEV, DELTA = 110, 0.577, 0.90, 0.05


def one(args):
    shape, seed = args
    rng = np.random.default_rng(seed)
    w = rng.lognormal(0, .7, K); D = DBAR * w / np.exp(.7**2 / 2)
    V = draw(shape, K, rng) + rng.normal(0, np.sqrt(D))
    halves = {'noisy_threshold': noisy_threshold_halfwidth(V, q=LEV, delta=DELTA),
              'corollary2': simple_shrink_halfwidth(V, D.min(), q=LEV, delta=DELTA),
              'markov_k107': shape_free_markov(V, D, 107, q=LEV, delta=DELTA)[0]}
    return [dict(shape=shape, seed=seed, method=m, half=h, cov=float(cdf(shape, h) - cdf(shape, -h)))
            for m, h in halves.items()]


def clopper_pearson(x, n, a=0.05):
    lo = 0.0 if x == 0 else stats.beta.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x == n else stats.beta.ppf(1 - a / 2, x + 1, n - x)
    return lo, hi


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    jobs = [(s, 70000 + 10000 * i + r) for i, s in enumerate(SHAPES) for r in range(reps)]
    with Pool(procs) as pool:
        r = pd.DataFrame([row for rows in pool.map(one, jobs, chunksize=50) for row in rows])
    r.to_csv(RESULTS / 'analytic_reliability.csv', index=False)
    rows = []
    for (shape, m), g in r.groupby(['shape', 'method'], sort=False):
        x, n = int(np.sum(g['cov'] >= LEV)), len(g)
        lo, hi = clopper_pearson(x, n)
        rows.append(dict(shape=shape, method=m, reps=n, successes=x, reliability=x / n, ci_lo=lo, ci_hi=hi,
                         mean_cov=g['cov'].mean(), q05_cov=g['cov'].quantile(.05), width=2 * g.half.mean()))
    s = pd.DataFrame(rows)
    s.to_csv(RESULTS / 'analytic_reliability_summary.csv', index=False)
    print(s.round(4).to_string(index=False))
