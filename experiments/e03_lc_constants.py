"""E03. Log-concave constants.

(a) sup over log-concave laws of the (1-alpha)-quantile of |X-EX|/sd. The extremal is a
    truncated Laplace law (found by a piecewise log-affine search, 1-4 pieces, and confirmed
    in closed form): 1.7559 at 90% (Gaussian 1.645), 2.1691 at 95% (Gaussian 1.960).
(b) worst coverage of mean +- z sd over the truncated Laplace family:
    87.75% at nominal 90%, 92.97% at nominal 95%.
    NOTE: (b) is the infimum over the truncated Laplace family only; that it is the infimum
    over all log-concave laws is not yet verified.
(c) optional --search: re-run the piecewise log-affine global search for (a).
Writes results/lc_constants.json.
"""
import json
import sys

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm

from _common import RESULTS
from uai.logconcave import quantile_sd_constant, worst_coverage_normal_interval


def piecewise_search(alpha, k, G=20001):
    x = np.linspace(0, 1, G)

    def ratio(params):
        b = np.sort(params[:k - 1]); s = -np.sort(-params[k - 1:])
        edges = np.r_[0, b, 1]; seg = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, k - 1)
        lv = np.r_[0, np.cumsum(s * np.diff(edges))]
        lf = lv[seg] + s[seg] * (x - edges[seg]); f = np.exp(lf - lf.max()); w = f / f.sum()
        m = (x * w).sum(); sd = np.sqrt(((x - m)**2 * w).sum()); d = np.abs(x - m) / sd
        o = np.argsort(d); c = np.cumsum(w[o]); return d[o][np.searchsorted(c, 1 - alpha)]
    r = differential_evolution(lambda p: -ratio(p), [(0, 1)] * (k - 1) + [(-80, 80)] * k, seed=0,
                               tol=1e-10, maxiter=3000, popsize=40, polish=True)
    return -r.fun


out = {}
for alpha in (.20, .10, .05):
    const, c = quantile_sd_constant(alpha)
    cov, c2 = worst_coverage_normal_interval(alpha)
    out[str(alpha)] = dict(quantile_sd_constant=const, truncation=c, gaussian=norm.ppf(1 - alpha / 2),
                           worst_normal_interval_coverage_trunc_laplace=cov, worst_at_truncation=c2)
    print(alpha, out[str(alpha)])
if '--search' in sys.argv:
    for k in (1, 2, 3):
        out[f'search_alpha0.1_pieces{k}'] = piecewise_search(.10, k)
        print('pieces', k, out[f'search_alpha0.1_pieces{k}'])
json.dump(out, open(RESULTS / 'lc_constants.json', 'w'), indent=1)
