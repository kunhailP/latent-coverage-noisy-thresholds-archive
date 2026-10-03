"""E24. Numerical (double-precision branch-and-bound) upper bounds for R_{p,q}(x), q = .90,
p in {.90, .9036, .9068} (the last two: order-statistic levels at delta = .04, .05).

For each x on a grid: a lower bound L (feasible extremal law: coarse (beta, ell) grid of exact
endpoint values, polished by Nelder-Mead) and a certified upper bound U (monotone branch and
bound, `uai.certify`, first relative tolerance in TOLS that clears). Between grid points the
lookup uses R(x + h) <= R(x) + c_q sqrt(h) (Supplementary Proposition S5); below x = .01 it uses
Theorem 3 and its p >= q form (Section 4.3; Supplementary equation (S2)).
  python experiments/e24_certified_table.py [procs] [step] [x_lo] [x_hi] [p,p,...]
Writes results/certified_R.csv, merged with existing rows (same p and x are replaced), so a
second run over a new x range extends the table. The default range is [.010, .366]; the
table was extended to [.002, .0095] by `e24_certified_table.py 14 5e-4 0.002 0.0095`, and
p = .9068 (Beta(105, 6) quantile at delta = .05) added by `... 14 5e-4 0.002 0.366 0.9068`.
"""
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from _common import RESULTS
from uai.certify import certified_upper
from uai.extremal import _endpoint_values, dirac_value

warnings.filterwarnings('ignore')
Q = 0.90
PS = [0.90, 0.9036]
TOLS = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
BETAS = np.concatenate([[0], np.geomspace(.05, 60, 22)])
ELLS = np.geomspace(1e-3, 4, 22)


def lower(p, x):
    best, arg = dirac_value(p, Q, x), None
    for be in BETAS:
        for el in ELLS:
            v = _endpoint_values(p, Q, x, be, el)
            if v > best:
                best, arg = v, (be, el)
    if arg is None:
        return best, np.nan, np.nan
    f = lambda z: -_endpoint_values(p, Q, x, np.exp(z[0]), np.exp(z[1]))
    z0 = np.log([max(arg[0], 1e-3), arg[1]])
    r = minimize(lambda z: f(z) if np.isfinite(f(z)) else 10.0, z0, method='Nelder-Mead',
                 options={'xatol': 1e-6, 'fatol': 1e-10, 'maxiter': 400})
    if -r.fun > best:
        return -r.fun, float(np.exp(r.x[0])), float(np.exp(r.x[1]))
    return best, float(arg[0]), float(arg[1])


def one(args):
    p, x = args
    t = time.time()
    L, be, el = lower(p, x)
    U, tol, n = certified_upper(p, Q, x, L, rel_tols=TOLS, max_boxes=8_000_000)
    return dict(p=p, q=Q, x=x, L=L, U=U, tol=tol, evals=n, beta=be, ell=el,
                seconds=time.time() - t)


if __name__ == '__main__':
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    step = float(sys.argv[2]) if len(sys.argv) > 2 else 5e-4
    x_lo = float(sys.argv[3]) if len(sys.argv) > 3 else 0.010
    x_hi = float(sys.argv[4]) if len(sys.argv) > 4 else 0.366
    ps = [float(v) for v in sys.argv[5].split(',')] if len(sys.argv) > 5 else PS
    xs = np.round(np.arange(x_lo, x_hi + step / 2, step), 6)
    jobs = [(p, x) for p in ps for x in xs]
    with Pool(procs) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=1)):
            rows.append(r)
            if i % 50 == 0:
                print(i, len(jobs), r, flush=True)
    d = pd.DataFrame(rows)
    out = RESULTS / 'certified_R.csv'
    if out.exists():
        old = pd.read_csv(out)
        keep = ~old.set_index(['p', 'x']).index.isin(d.set_index(['p', 'x']).index)
        d = pd.concat([old[keep], d])
    d = d.sort_values(['p', 'x'])
    d.to_csv(out, index=False)
    print(d.groupby('p').agg(n=('x', 'size'), certified=('U', lambda u: np.isfinite(u).mean()),
                             max_gap=('tol', 'max'), med_gap=('tol', 'median')))
