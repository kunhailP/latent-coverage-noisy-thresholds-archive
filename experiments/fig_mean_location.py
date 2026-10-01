"""Supplementary figure: radius correction against the noise ratio for the laws of E35.

  python experiments/fig_mean_location.py      -> paper/fig_mean_location.pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import RESULTS, ROOT

d = pd.read_csv(RESULTS / 'mean_location.csv')
fig, ax = plt.subplots(figsize=(4.6, 4.0))
style = {'boundary': ('o', 'mean at $20.2\\,x^{1/2}$ from the boundary'),
         'kappa50': ('s', 'mean at $50\\,x^{1/2}$'),
         'centred': ('^', 'mean zero')}
for law, (m, lab) in style.items():
    g = d[(d.law == law) & (d.delta > 0)]
    ax.plot(g.x, g.delta, 'k' + m, ms=4, mfc='none', label=lab)
xs = np.geomspace(1e-8, 1e-3, 50)
ax.plot(xs, 0.019062 * np.sqrt(xs), color='0.5', lw=0.8)
ax.plot(xs, 0.008945 * np.sqrt(xs), color='0.5', lw=0.8, ls='--')
ax.plot(xs, 0.024005 * xs, color='0.5', lw=0.8, ls=':')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Noise-to-threshold variance ratio, $x$')
ax.set_ylabel('Radius correction, $Q_{0.9}(|W|) - 1$')
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / 'paper' / 'fig_mean_location.pdf')
