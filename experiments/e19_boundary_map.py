"""E19. Where noisy calibration must widen, may keep, or may shrink (common known noise).

For q in {.80, .85, .90, .95} and p = q, compute R_{q,q}(x) on a fine small-x grid with the
exact reduction (E16 method), locate the crossing x*(q) where R = 1, and record the maximum
widening sup_x R. Below x* the noisy threshold must be widened; above it, it can shrink.
  python experiments/e19_boundary_map.py [procs]
Writes results/boundary_map.csv (q, x, R) and prints x*(q), max R.
"""
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from _common import RESULTS
from e16_exact_shrink_table import BETAS, ELLS, polish
from uai.extremal import dirac_value, shape_values

warnings.filterwarnings('ignore')


def column(args):
    q, x, beta = args
    v = shape_values(q, q, beta, ELLS, [x], m=160)
    top = np.argsort(v)[-3:]
    return q, x, [(v[i], beta, ELLS[i]) for i in top if np.isfinite(v[i])]


if __name__ == '__main__':
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    qs = [.80, .85, .90, .95]
    xs = np.r_[np.exp(np.linspace(np.log(1e-4), np.log(4e-2), 22))]
    with Pool(procs) as pool:
        out = pool.map(column, [(q, x, b) for q in qs for x in xs for b in BETAS])
        starts = {}
        for q, x, top in out:
            starts.setdefault((q, x), []).extend(top)
        keys = sorted(starts)
        pol = pool.starmap(polish, [(q, q, x, sorted(starts[(q, x)])[-6:]) for q, x in keys])
    rows = [dict(q=q, x=x, R=max(v, dirac_value(q, q, x))) for (q, x), (v, _, _) in zip(keys, pol)]
    df = pd.DataFrame(rows); df.to_csv(RESULTS / 'boundary_map.csv', index=False)
    for q, g in df.groupby('q'):
        g = g.sort_values('x'); r = g.R.values - 1; xv = g.x.values
        i = np.where((r[:-1] > 0) & (r[1:] <= 0))[0]
        xstar = (np.exp(np.interp(0, [-r[i[0]], -r[i[0] + 1]], np.log(xv[i[0]:i[0] + 2])))
                 if len(i) else np.nan)
        print(f'q={q}: x*={xstar:.5f}, max R={g.R.max():.6f} at x={xv[np.argmax(g.R.values)]:.5f}')
