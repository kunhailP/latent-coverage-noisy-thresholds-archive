"""E28: split the HetLDC width gain in E20 into the order-statistic level effect (p_k at delta = .04
vs .05) and the exact-kernel effect at a matched level (Supplementary Material, Section S6). Grid values, ~70 min on 14 cores.
  python experiments/e28_level_decomposition.py
Prints the per-shape ratios reported in the Supplementary Material.
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from _common import RESULTS
from e12_conditional_synth import draw
from uai.procedures import CertifiedShrinkTable, hetldc_parts, pac_rank

K, DBAR = 110, 0.577


def one(a):
    shape, seed = a
    rng = np.random.default_rng(seed)
    w = rng.lognormal(0, .7, K); D = DBAR * w / np.exp(.7**2 / 2)
    V = draw(shape, K, rng) + rng.normal(0, np.sqrt(D))
    kp = pac_rank(K, .9, .05)
    a5 = hetldc_parts(V, D, kp, delta=.05); a4 = hetldc_parts(V, D, kp, delta=.04)
    T = a5[0]
    ldc4 = T * CertifiedShrinkTable(0.9036)(D.mean() / T**2)
    return dict(shape=shape, seed=seed, pk5=a5[1], pk4=a4[1], h5=T * a5[4], h4=T * a4[4], ldc4=ldc4)


if __name__ == '__main__':
    d0 = pd.read_csv(RESULTS/'hetldc_synth.csv')
    jobs = sorted(set(zip(d0['shape'], d0['seed'])))
    with Pool(14) as p: r = pd.DataFrame(p.map(one, jobs))
    m = d0.pivot_table(index=['shape','seed'], columns='method', values='half').reset_index()
    r = r.merge(m, on=['shape','seed'])
    assert np.allclose(r.h5, r.HetLDC, rtol=1e-9), (r.h5/r.HetLDC).describe()
    print(r[['pk5','pk4']].drop_duplicates())
    g = r.groupby('shape')
    print(pd.DataFrame({'level_effect': g.apply(lambda s: s.h4.mean()/s.h5.mean()-1),
                        'kernel_effect_at_.9036': g.apply(lambda s: s.h4.mean()/s.ldc4.mean()-1),
                        'cert/grid': g.apply(lambda s: s.HetLDC_cert.mean()/s.HetLDC.mean()-1)}).round(4))
