"""E18. Heterogeneous survey noise: exact mixture kernel vs plugging in a mean variance.

Calibration areas have noise variances D_i (scaled: x_i = D_i / t^2). The exact constraint
uses the average kernel gbar = mean_i P(|w + e_i| <= 1); the common-variance shortcut uses one
Gaussian with the mean variance xbar. Both give a shrink factor through the same extremal
reduction. If R_mix > R(xbar), the shortcut is anti-conservative for the latent target.
Spread: x_i = xbar * lognormal(0, sigma) / E[.], represented by 12 equal-weight quantiles.
  python experiments/e18_hetero_kernel.py [procs]
Writes results/hetero_kernel.csv.
"""
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from _common import RESULTS
from uai.extremal import _endpoint_values_mix, shape_values

warnings.filterwarnings('ignore')
BETAS = np.r_[0.0, np.exp(np.linspace(np.log(1e-2), np.log(1e3), 40))]
ELLS = np.exp(np.linspace(np.log(1e-3), np.log(20), 70))


def spread(xbar, sigma, n=12):
    if sigma == 0:
        return np.array([xbar])
    u = (np.arange(n) + 0.5) / n
    z = np.exp(sigma * stats.norm.ppf(u))
    return xbar * z / z.mean()


def R_of(p, q, xs):
    best = (-np.inf, None)
    for beta in BETAS:
        v = shape_values(p, q, beta, ELLS, xs, m=120)
        i = int(np.argmax(v))
        if v[i] > best[0]:
            best = (v[i], (beta, ELLS[i]))
    if best[1] is None:
        return -np.inf
    f = lambda z: -_endpoint_values_mix(p, q, xs, None, np.exp(z[0]), np.exp(z[1]))
    r = minimize(f, [np.log(max(best[1][0], 1e-3)), np.log(best[1][1])], method='Nelder-Mead',
                 options={'xatol': 1e-8, 'fatol': 1e-12, 'maxiter': 400})
    return max(best[0], -r.fun if np.isfinite(r.fun) else -np.inf)


def job(args):
    p, q, xbar, sigma = args
    xs = spread(xbar, sigma)
    return dict(p=p, q=q, xbar=xbar, sigma=sigma, R_mix=R_of(p, q, xs),
                R_mean_plugin=R_of(p, q, np.array([xbar])),
                R_min_plugin=R_of(p, q, np.array([xs.min()])))


if __name__ == '__main__':
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    jobs = [(p, .9, xb, sg) for p in (.9, .9036) for xb in (.02, .05, .1, .136, .2)
            for sg in (0, .35, .7, 1.0)]
    with Pool(procs) as pool:
        df = pd.DataFrame(pool.map(job, jobs))
    df['mix_minus_mean'] = df.R_mix - df.R_mean_plugin
    df.to_csv(RESULTS / 'hetero_kernel.csv', index=False)
    pd.set_option('display.width', 200)
    print(df.round(5).to_string(index=False))
