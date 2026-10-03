"""E32: centring and the boundary-distance transition (Theorems 4 and 5, Supplementary Proposition S4 and Supplementary Lemma S2 of the paper).

  python experiments/e32_centering.py
Writes results/centering.json:
  transition[q]   u*, c_q, kappa* = k(u*), the crossing kappa_x of the two upper bounds, and
                  L_q(kappa), U_q(kappa), L/U on a list of kappa (double precision, not certified)
  centred         mean-zero truncated-exponential lower-bound coefficients: the beta = .2, ell = 2
                  example, the best beta*ell on a grid, and finite-noise checks of the expansion
  sqrt_bound      the mean-zero bound Q_q(|W|) <= sqrt(t^2 + D) and its mean-offset form checked on
                  a grid of segment laws (number of violations; must be 0)
  leakage         q(1 - e^{-2A}), A = 1 - log(1/q): the zero-noise central coverage of the naive
                  kappa -> infinity construction at d = 1, which is why A/2 is not a centred bound
"""
import json
import time

import numpy as np

from _common import RESULTS
from uai.extremal import (centred_exp_coefficient, centred_exp_law, latent_abs_quantile,
                          noisy_abs_quantile, tail_optimum, transition_constant, transition_upper)

QS = (0.8, 0.9, 0.95)
KAPPAS = (0, 5, 10, 15, 20, 25, 30, 40, 50, 100, 300, 1000)


def transition(q):
    u, c, k = tail_optimum(q)
    kx = (1 - c * c) / (2 * c)
    rows = []
    for kap in sorted(set(KAPPAS) | {round(k, 6), kx}):
        L, U = transition_constant(q, kap), transition_upper(q, kap)
        rows.append({'kappa': kap, 'L': L, 'U': U, 'ratio': L / U})
    A = 1 - np.log(1 / q)
    return {'u_star': u, 'c_q': c, 'kappa_star': k, 'kappa_cross': kx,
            'large_kappa_ratio': A, 'rows': rows,
            'min_ratio': min(r['ratio'] for r in rows)}


def centred(q):
    ex = centred_exp_coefficient(q, 0.2, 2.0)
    grid = [(bl, centred_exp_coefficient(q, bl, 1.0)) for bl in np.concatenate([np.arange(0.05, 8.0, 0.0025), np.geomspace(8, 200, 60)])]
    grid = [(bl, c) for bl, c in grid if np.isfinite(c)]
    bl, cbest = max(grid, key=lambda g: g[1])
    a, b = centred_exp_law(0.2, 2.0)
    r0 = latent_abs_quantile(q, a, b, 0.2)
    checks = []
    for sd in (0.03, 0.01, 0.005):
        t = noisy_abs_quantile(q, a, b, 0.2, sd * sd)
        x = sd * sd / t ** 2
        checks.append({'sd': sd, 'x': x, 'ratio_over_x': (r0 / t - 1) / x})
    return {'example_beta0.2_ell2': ex, 'best_beta_ell': bl, 'best_coefficient': cbest,
            'finite_noise_example': checks}


def sqrt_bound():
    worst, n, bad = -np.inf, 0, 0
    for q in (0.5, 0.8, 0.9, 0.97):
        for beta in (-4, -1, -0.2, 0.2, 1, 4):
            for ell in (0.5, 2.0, 6.0):
                a, b = centred_exp_law(beta, ell)
                for mu in (0.0, 0.1, 0.3):
                    for sd in (0.03, 0.3, 1.0):
                        x = sd * sd
                        t = noisy_abs_quantile(q, a + mu, b + mu, beta, x)
                        if t <= mu:
                            continue
                        lat = latent_abs_quantile(q, a + mu, b + mu, beta)
                        gap = lat - (mu + np.sqrt((t - mu) ** 2 + x))
                        worst, n, bad = max(worst, gap), n + 1, bad + (gap > 1e-9)
    return {'laws_checked': n, 'violations': bad, 'max_excess': worst}


if __name__ == '__main__':
    t0 = time.time()
    out = {'transition': {str(q): transition(q) for q in QS},
           'centred': {str(q): centred(q) for q in QS},
           'sqrt_bound': sqrt_bound(),
           'leakage': {str(q): q * (1 - np.exp(-2 * (1 - np.log(1 / q)))) for q in QS}}
    (RESULTS / 'centering.json').write_text(json.dumps(out, indent=1, default=float))
    for q in QS:
        tr = out['transition'][str(q)]
        print(f"q={q}: c_q={tr['c_q']:.10f} kappa*={tr['kappa_star']:.5f} "
              f"kappa_x={tr['kappa_cross']:.5f} min L/U={tr['min_ratio']:.6f} A={tr['large_kappa_ratio']:.6f}")
        ce = out['centred'][str(q)]
        print(f"   centred: example {ce['example_beta0.2_ell2']:.10f}, best {ce['best_coefficient']:.6f}"
              f" at beta*ell={ce['best_beta_ell']:.2f}")
    print('sqrt bound:', out['sqrt_bound'], ' leakage:', out['leakage'])
    print(f'{time.time() - t0:.0f}s')
