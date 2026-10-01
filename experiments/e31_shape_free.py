"""E31. Shape-free baseline in the spirit of LatentCP (Zheng, Zhou & Zhu 2026) on the E20 data.

For each E20 data set (same seeds), `procedures.shape_free_markov` with the known D_i and ranks
k = 106..110 (delta = .05): radius T r with gbar_T(r) = (p_k - q gbar_T(0))/(1 - q). It needs no shape
assumption on the latent law (Markov's inequality), only independent Gaussian noise with known
variances. The rank must be fixed in advance; the table reports every k.
  python experiments/e31_shape_free.py
Writes results/shape_free.csv and results/shape_free_summary.csv.
"""
import numpy as np
import pandas as pd

from _common import RESULTS
from e12_conditional_synth import draw
from uai.latent_laws import cdf
from uai.procedures import shape_free_markov

K, DBAR, LEV = 110, 0.577, 0.90
KS = range(106, 111)

if __name__ == '__main__':
    d0 = pd.read_csv(RESULTS / 'hetldc_synth.csv')
    rows = []
    for shape, seed in sorted(set(zip(d0['shape'], d0.seed))):
        rng = np.random.default_rng(seed)                     # same draws as e20_hetldc_synth.one
        w = rng.lognormal(0, .7, K); D = DBAR * w / np.exp(.7**2 / 2)
        V = draw(shape, K, rng) + rng.normal(0, np.sqrt(D))
        for k in KS:
            h, p_k, _ = shape_free_markov(V, D, k, q=LEV, delta=.05)
            rows.append(dict(shape=shape, seed=seed, method=f'SF_k{k}', k=k, p_k=p_k, half=h,
                             cov=float(cdf(shape, h) - cdf(shape, -h))))
    r = pd.DataFrame(rows)
    r.to_csv(RESULTS / 'shape_free.csv', index=False)
    s = r.groupby(['shape', 'method']).agg(reps=('cov', 'size'),
                                          pr_cov_ge_90=('cov', lambda c: np.mean(c >= LEV)),
                                          width=('half', lambda h: 2 * h.mean())).reset_index()
    s.to_csv(RESULTS / 'shape_free_summary.csv', index=False)
    print(s.round(4).to_string(index=False))
