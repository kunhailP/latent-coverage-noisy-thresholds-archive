"""E21. apipop: HetLDC next to FH_PAC, CP_PAC and mean-variance LDC_PAC.

Every rule plugs in the design-based D_hat_c as if known (the Fay-Herriot convention), so the
rules differ only in how they use them: FH through the area model, LDC_PAC through one
Gaussian with the pooled lower bound of the mean, HetLDC through the exact average kernel of
the individual D_hat_c. With n = 2-3 schools per district D_hat_c is very noisy, so no rule's
guarantee applies literally; this is the practical check. Coverage = share of the 105 unused
districts covered (finite-population conditional coverage).
  python experiments/e21_hetldc_apipop.py [reps] [procs]
Writes results/hetldc_apipop.csv and results/hetldc_apipop_summary.csv.
"""
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from _common import RESULTS, apipop_replication, load_apipop
from uai.procedures import (ShrinkTable, conformal_threshold, fay_herriot_boot_pac,
                            fay_herriot_reml, hetldc_halfwidth, noise_lower_bound, pac_rank)

warnings.filterwarnings('ignore')
CONDITIONS = [('api99', 3), ('api99', 2), ('socio', 3), ('socio', 2)]
LEV = 0.90


def one(args):
    fset, n, seed = args
    d, feats = load_apipop(); table = ShrinkTable('0.1'); rng = np.random.default_rng(seed)
    V, D, Wc, We = apipop_replication(d, feats[fset], n, rng)
    K = len(V); kp = pac_rank(K, LEV, .05); T = conformal_threshold(V, kp)
    mu, A, vmu = fay_herriot_reml(V, D); se = np.sqrt(A + vmu)
    _, lam = fay_herriot_boot_pac(V, D, mu, A, rng, level=LEV)
    D_low = noise_lower_bound(D, K * (n - 1))
    ints = {'FH_PAC': (mu, lam * se), 'CP_PAC': (0, T),
            'LDC_PAC': (0, T * table(D_low / T**2)), 'HetLDC': (0, hetldc_halfwidth(V, D, kp))}
    return [dict(condition=f'{fset}_n{n}', seed=seed, method=m, half=h,
                 cov=float(np.mean(np.abs(We - c) <= h))) for m, (c, h) in ints.items()]


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    jobs = [(f, n, 70000 + 1000 * i + r) for i, (f, n) in enumerate(CONDITIONS) for r in range(reps)]
    with Pool(procs) as pool:
        r = pd.DataFrame([row for rows in pool.imap_unordered(one, jobs, chunksize=2) for row in rows])
    r.to_csv(RESULTS / 'hetldc_apipop.csv', index=False)
    s = r.groupby(['condition', 'method']).agg(reps=('cov', 'size'), mean_cov=('cov', 'mean'),
                                              pr_cov_ge_90=('cov', lambda c: np.mean(c >= LEV)),
                                              width=('half', lambda h: 2 * h.mean())).reset_index()
    s.to_csv(RESULTS / 'hetldc_apipop_summary.csv', index=False)
    print(s.round(4).to_string(index=False))
