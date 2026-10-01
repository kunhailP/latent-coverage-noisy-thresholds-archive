"""E27: ball-arithmetic enclosures of c_q, c_{p,q} and certificates for C_{p,q} (src/uai/interval.py).

  python experiments/e27_interval_constants.py [procs]
Writes results/interval_constants.json:
  c[q]          rigorous [lo, hi] for c_q, q = .80, .85, .90, .95, .99
  c_pq          rigorous [lo, hi] for c_{p,.9}, p = .9036, .9068; lo also bounds C_{p,.9} below
  split         certificates C_{p,q} <= g (Psi upper bound < 1 - q)
  slack         for q = .80, .90, .95: c_{q+d_lo,q} > 0 (so C > 0, slack d_lo is not enough) and
                C_{q+d_hi,q} <= 0 (slack d_hi is enough)
"""
import json
import sys
import time
from multiprocessing import Pool

from _common import RESULTS
from uai.interval import c_enclosure, split_certificate

SLACK = {'0.8': ('0.0043', '0.0050'), '0.9': ('0.00079', '0.0010'), '0.95': ('0.00015', '0.00020')}
SPLIT = [('0.9036', '0.9', -0.05), ('0.9036', '0.9', -0.057), ('0.9068', '0.9', -0.114)]


def add(a, b):
    """Decimal string sum, exact for these inputs."""
    from decimal import Decimal
    return str(Decimal(a) + Decimal(b))


def c_job(p, q):
    lo, hi, info = c_enclosure(p, q, max_iter=200000)
    return p, q, lo, hi, info


if __name__ == '__main__':
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    t0 = time.time()
    out = {'c': {}, 'split': [], 'slack': {}}
    jobs = [(q, q) for q in ('0.8', '0.85', '0.9', '0.95', '0.99')] + [('0.9036', '0.9'), ('0.9068', '0.9')]
    jobs += [(add(q, lo), q) for q, (lo, _) in SLACK.items()]
    with Pool(procs) as pool:
        for p, q, lo, hi, info in pool.starmap(c_job, jobs):
            print(f'c_{{{p},{q}}} in [{lo:.9f}, {hi:.9f}]', flush=True)
            if p == q:
                out['c'][q] = [lo, hi]
            elif p in ('0.9036', '0.9068'):
                out.setdefault('c_pq', {})[p] = {'q': q, 'lo': lo, 'hi': hi}
            else:
                out['slack'].setdefault(q, {})['not_enough'] = {'p': p, 'c_lo': lo, 'positive': lo > 0}
        for p, q, g in SPLIT:
            psi, budget, ok, info = split_certificate(p, q, g, pool=pool, max_rounds=40)
            print(f'C_{{{p},{q}}} <= {g}: Psi <= {psi:.8f} vs 1 - q = {budget}: {ok} {info}', flush=True)
            out['split'].append(dict(p=p, q=q, g=g, psi=psi, budget=budget, ok=ok, **info))
        for q, (_, hi) in SLACK.items():
            p = add(q, hi)
            psi, budget, ok, info = split_certificate(p, q, 0.0, pool=pool)
            print(f'C_{{{p},{q}}} <= 0: Psi <= {psi:.8f} vs {budget}: {ok} {info}', flush=True)
            out['slack'].setdefault(q, {})['enough'] = dict(p=p, psi=psi, budget=budget, ok=ok, **info)
    out['seconds'] = time.time() - t0
    (RESULTS / 'interval_constants.json').write_text(json.dumps(out, indent=1, default=str))
    print(f'done in {out["seconds"]:.0f} s')
