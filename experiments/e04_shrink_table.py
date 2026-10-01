"""E04. Table of the log-concave deconvolution shrink factor r_alpha(x), x = D / t^2.

If W is log-concave, e ~ N(0, D) independent, and P(|W + e| <= t) >= 1 - alpha, then
P(|W| <= t * r_alpha(D/t^2)) >= 1 - alpha, and r_alpha is sharp (numerically; extremal =
log-affine law on a segment). Grid x = 0.001 + 0.005 i (i < 75), alpha in {.10,.09,.08,.07}.
Each point is a differential-evolution search (3 seeds) plus Dirac candidates.
Writes results/r_table.json  {alpha: {x: r}}.  ~5 min on 100+ cores.
"""
import json
import sys
from multiprocessing import Pool

import numpy as np

from _common import RESULTS
from uai.logconcave import shrink_factor


def job(args):
    x, a = args
    return a, x, shrink_factor(x, a)


if __name__ == '__main__':
    alphas = [.10, .09, .08, .07]
    xs = [round(0.001 + 0.005 * i, 4) for i in range(75)]
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    with Pool(procs) as pool:
        res = pool.map(job, [(x, a) for a in alphas for x in xs])
    tab = {}
    for a, x, r in res:
        tab.setdefault(str(a), {})[str(x)] = r
    for a in tab:
        rs = [tab[a][str(x)] for x in xs]
        assert all(rs[i + 1] <= rs[i] + 1e-6 for i in range(len(rs) - 1)), f'non-monotone at {a}'
    json.dump(tab, open(RESULTS / 'r_table.json', 'w'), indent=1)
    print('wrote results/r_table.json')
