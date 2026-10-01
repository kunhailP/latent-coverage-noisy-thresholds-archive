"""E41: coverage transfer for non-Gaussian noise (Proposition 5, Theorem 4, Supplement S2).

  python experiments/e41_noise_classes.py [procs] [--no-splits]
Writes results/noise_classes.json:
  symmetric_unimodal  enclosures of Psi_SU(q) (closed form), two-sided split certificates for
                      the levels p_E and for the coverage form, and the failing levels;
  log_concave         branch-and-bound certificates Psi_LC(q) < target, the centred-exponential
                      lower bound, and two-sided split certificates;
  gaussian            sup over the scale of E min(1, q e^{-sigma Z}) (double precision);
  unimodal            the counterexample of Theorem 4(iii) (double precision).
"""
import json
import math
import sys
import time

import numpy as np
from scipy import integrate, stats
from scipy.optimize import minimize_scalar

from _common import RESULTS
from uai.interval import A, fdown, fup
from uai.noise_classes import (lc_certificate_parallel, lc_point, psi_su_enclosure,
                               split_certificate_lc, split_certificate_su)


def gaussian_psi(q):
    c = math.log(q)
    J = lambda s: q * (integrate.quad(lambda z: math.exp(-z) * stats.norm.cdf(z / s), c, 0)[0]
                       + integrate.quad(lambda z: math.exp(-z) * stats.norm.cdf(z / s), 0, np.inf)[0])
    r = minimize_scalar(lambda ls: -J(math.exp(ls)), bounds=(math.log(1e-4), math.log(2)),
                        method='bounded', options={'xatol': 1e-11})
    return -r.fun, math.exp(r.x)


def unimodal_example(q, rho=1e-3, a=1e3):
    """eps with density (1-rho)/a on [-a, 0] and rho/b on (0, b], b = (1-rho) a / rho."""
    c = math.log(q)
    return (1 - rho) * (1 + c / a)              # lower bound of E min(1, q e^{-eps})


if __name__ == '__main__':
    procs = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    t0 = time.time()
    out = {'symmetric_unimodal': {}, 'log_concave': {}, 'gaussian': {}, 'unimodal': {}}
    su = out['symmetric_unimodal']
    for q in ('0.8', '0.9', '0.95'):
        lo, hi = psi_su_enclosure(q)
        su[q] = dict(psi=[lo, hi])
    for p, q in (('0.8078', '0.8'), ('0.9018', '0.9'), ('0.9505', '0.95'), ('0.9', '0.8981'),
                 ('0.9068', '0.9')):
        ok, worst, budget = split_certificate_su(p, q)
        su.setdefault('split', []).append(dict(p=p, q=q, ok=ok, worst=worst, budget=budget))
    for p, q in (('0.8077', '0.8'), ('0.9017', '0.9'), ('0.9504', '0.95'), ('0.9', '0.8982')):
        lo, _ = psi_su_enclosure(q)
        su.setdefault('fails', []).append(dict(p=p, q=q, psi_lo=lo, fails=lo > float(p)))
    for q in (0.8, 0.9, 0.95):
        v, s = gaussian_psi(q)
        out['gaussian'][str(q)] = dict(psi=v, sigma=s)
    out['unimodal'] = dict(q=0.9, lower_bound=unimodal_example(0.9))
    lc = out['log_concave']
    for q, s in (('0.9', 0.1177686817), ('0.8942', 0.1259052792)):
        v = lc_point(q, s)
        lc.setdefault('exp_lower', []).append(dict(q=q, sigma=s, psi_lo=fdown(v)))
    for target in ('0.9068', '0.9053', '0.9052'):
        ok, d = lc_certificate_parallel('0.9', target, procs=procs, pieces=40)
        lc.setdefault('one_sided', []).append(dict(q='0.9', target=target, ok=ok, boxes=d['boxes'],
                                                   worst=d['worst']))
    if '--no-splits' not in sys.argv:
        for p, q in (('0.9068', '0.9'), ('0.9', '0.8935'), ('0.906', '0.9')):
            ok, worst, budget, info = split_certificate_lc(p, q, procs=procs, pieces=24)
            lc.setdefault('split', []).append(dict(p=p, q=q, ok=ok, worst=worst, budget=budget,
                                                   intervals=len(info['grid']) - 1,
                                                   nodes=info['nodes']))
    out['seconds'] = time.time() - t0
    (RESULTS / 'noise_classes.json').write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
