"""E33: the noise-dominated regime (behaviour of R near the feasibility edge).

  python experiments/e33_edge.py [procs]
Writes results/edge.json:
  constants[q]   M_q (log-concave), the symmetric log-concave constant (results/lc_constants.json),
                 the Gaussian z_{(1+q)/2} and the shape-free (1 - q)^{-1/2}
  homogeneous[p] certified L, U of R_{p,0.9}(x) (results/certified_R.csv) against M_q (x_p - x)^{1/2}
  heterogeneous  the E20 data sets (40 per law): grid radius R^mix against M_q c*^{1/2}, c* the
                 latent variance budget of the average kernel
"""
import json
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from _common import RESULTS
from e12_conditional_synth import draw
from uai.extremal import edge_constant, edge_level, variance_budget
from uai.procedures import conformal_threshold, hetldc_parts, pac_rank

warnings.filterwarnings('ignore')
K, DBAR, LEV = 110, 0.577, 0.90
SHAPES = ['normal', 'laplace', 'gamma2_skew', 'trunc_laplace']
M90 = None


def het(args):
    shape, seed = args
    rng = np.random.default_rng(seed)
    w = rng.lognormal(0, .7, K); D = DBAR * w / np.exp(.7**2 / 2)
    V = draw(shape, K, rng) + rng.normal(0, np.sqrt(D))
    T, p_k, eps, _, R = hetldc_parts(V, D, pac_rank(K, LEV, .05))
    c = variance_budget(p_k - eps, D / T**2)
    return dict(shape=shape, seed=seed, xbar=float(np.mean(D) / T**2), R_grid=R,
                R_closed=M90 * np.sqrt(c))


if __name__ == '__main__':
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    sym = json.loads((RESULTS / 'lc_constants.json').read_text())
    const = {}
    for q in (0.8, 0.9, 0.95):
        const[str(q)] = {'log_concave': edge_constant(q),
                         'symmetric_log_concave': sym[str(round(1 - q, 2))]['quantile_sd_constant'],
                         'gaussian': float(stats.norm.ppf((1 + q) / 2)),
                         'shape_free': float((1 - q) ** -0.5)}
    M90 = const['0.9']['log_concave']
    cert = pd.read_csv(RESULTS / 'certified_R.csv')
    hom = {}
    for p in (0.9, 0.9036):
        sub = cert[(cert.p == p) & (cert.q == 0.9)].sort_values('x')
        pred = M90 * np.sqrt(np.clip(edge_level(p) - sub.x, 0, None))
        hom[str(p)] = {'x_p': edge_level(p), 'rows': [
            {'x': float(x), 'L': float(l), 'U': float(u), 'closed': float(c)}
            for x, l, u, c in zip(sub.x, sub.L, sub.U, pred)
            if np.min(np.abs(x - np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.34, 0.355]))) < 1e-9]}
        ok = (sub.x >= 0.1) & (sub.x < edge_level(p)) & np.isfinite(sub.L)
        hom[str(p)]['max_L_over_closed_x>=0.1'] = float((sub.L[ok] / pred[ok]).max())
        hom[str(p)]['min_L_over_closed_x>=0.1'] = float((sub.L[ok] / pred[ok]).min())
    jobs = [(s, 40000 + 1000 * i + r) for i, s in enumerate(SHAPES) for r in range(40)]
    with Pool(procs) as pool:
        d = pd.DataFrame(pool.map(het, jobs))
    d['ratio'] = d.R_closed / d.R_grid
    g = d.groupby('shape').agg(xbar=('xbar', 'mean'), R_grid=('R_grid', 'mean'),
                               R_closed=('R_closed', 'mean'), ratio_min=('ratio', 'min'),
                               ratio_mean=('ratio', 'mean'), ratio_max=('ratio', 'max'))
    out = {'constants': const, 'homogeneous': hom,
           'heterogeneous': {'n_data_sets': len(d), 'by_law': g.reset_index().to_dict('records')}}
    (RESULTS / 'edge.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(const, indent=1)); print(g.round(4).to_string())
    for p in hom:
        print(p, {k: v for k, v in hom[p].items() if k != 'rows'})
