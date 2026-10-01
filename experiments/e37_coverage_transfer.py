"""E37: coverage-scale form of Corollary 1. For noisy level p = q0 the certificate C_{p,q} <= 0
proves that every bi-log-concave latent law with noisy coverage p has latent coverage >= q at every
noise level; c_{p,q'} > 0 (a ball-arithmetic lower bound) shows that q' cannot be guaranteed.

  python experiments/e37_coverage_transfer.py [procs]
Writes results/coverage_transfer.json.
"""
import json
import sys
import time
from multiprocessing import Pool

from _common import RESULTS
from uai.interval import c_enclosure, split_certificate

# noisy level p, guaranteed latent level q, latent level q' that is not guaranteed
CASES = [('0.8', '0.795', '0.7954'), ('0.9', '0.8991', '0.8992'), ('0.95', '0.9498', '0.9499')]

if __name__ == '__main__':
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    t0, out = time.time(), []
    with Pool(procs) as pool:
        for p, q, q_bad in CASES:
            psi, budget, ok, info = split_certificate(p, q, 0.0, pool=pool)
            lo, hi, _ = c_enclosure(p, q_bad, max_iter=200000)
            row = dict(p=p, q=q, certified=ok, psi=psi, budget=budget, q_not=q_bad, c_lo=lo, c_hi=hi,
                       not_guaranteed=lo > 0)
            print(row, flush=True)
            out.append(row)
    (RESULTS / 'coverage_transfer.json').write_text(json.dumps(dict(cases=out, seconds=time.time() - t0), indent=1))
