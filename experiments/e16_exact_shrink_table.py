"""E16. The shrink function R_{p,q}(x) from the exact two-dimensional reduction.

Extremals are Diracs and log-affine laws on a segment (src/uai/extremal.py). For a shape
(beta, length) the feasible shifts form an interval and the latent quantile is quasi-convex
in the shift, so only the two endpoint shifts matter: R is a maximum over (beta, length) of
closed-form quantities. No optimiser and no quadrature. Grid: beta in {0} U logspace(1e-2, 3e3) (121),
length in logspace(1e-4, 30) (160), then Nelder-Mead polish of the exact 2-D function from the
best grid points. The result is a feasible value, i.e. a lower bound on R that converges to it;
it replaces the differential-evolution table of E04 as the reference (certification: E17).
  python experiments/e16_exact_shrink_table.py [procs] [p] [q]
Writes results/r_exact_p{p}_q{q}.csv (x, R, argmax, old DE value where available).
"""
import json
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from _common import RESULTS
from scipy.optimize import minimize

from uai.extremal import _endpoint_values, dirac_value, shape_values

warnings.filterwarnings('ignore')
BETAS = np.r_[0.0, np.exp(np.linspace(np.log(1e-2), np.log(3e3), 120))]
ELLS = np.exp(np.linspace(np.log(1e-4), np.log(30), 160))


def column(args):
    p, q, x, beta = args
    v = shape_values(p, q, beta, ELLS, [x], m=160)
    top = np.argsort(v)[-3:]
    return x, beta, [(v[i], ELLS[i]) for i in top]


def polish(p, q, x, starts):
    """Local refinement of the exact endpoint value in (log beta, log length)."""
    if not starts:
        return (-np.inf, np.nan, np.nan)
    best = max(starts)
    f = lambda z: -_endpoint_values(p, q, x, np.exp(z[0]), np.exp(z[1]))
    for v, beta, ell in sorted(starts)[-3:]:
        z0 = [np.log(max(beta, 1e-3)), np.log(ell)]
        r = minimize(f, z0, method='Nelder-Mead', options={'xatol': 1e-9, 'fatol': 1e-13, 'maxiter': 800})
        if np.isfinite(r.fun) and -r.fun > best[0]:
            best = (-r.fun, float(np.exp(r.x[0])), float(np.exp(r.x[1])))
    return best


if __name__ == '__main__':
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    p = float(sys.argv[2]) if len(sys.argv) > 2 else 0.90
    q = float(sys.argv[3]) if len(sys.argv) > 3 else 0.90
    old = json.load(open(RESULTS / 'r_table.json')).get('0.1', {}) if p == q == 0.9 else {}
    xs = sorted({float(k) for k in old} | {1e-6, 1e-5, 1e-4, 5e-4, 2e-3, 5e-3, 8e-3})
    with Pool(procs) as pool:
        out = pool.map(column, [(p, q, x, b) for x in xs for b in BETAS])
        starts = {x: [] for x in xs}
        for x, beta, top in out:
            starts[x] += [(v, beta, ell) for v, ell in top if np.isfinite(v)]
        pol = pool.starmap(polish, [(p, q, x, sorted(starts[x])[-6:]) for x in xs])
    rows = []
    for x, (v, beta, ell) in zip(xs, pol):
        d = dirac_value(p, q, x)
        R, beta, ell = (v, beta, ell) if v >= d else (d, np.nan, 0.0)
        rows.append(dict(x=x, R=R, beta=beta, length=ell, dirac=d,
                         R_DE=old.get(f'{x:g}', old.get(str(x), np.nan))))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f'r_exact_p{p:g}_q{q:g}.csv', index=False)
    pd.set_option('display.width', 200)
    print(df.to_string(index=False))
