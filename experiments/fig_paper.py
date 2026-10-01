"""Figures for the paper (PDF, white background, no titles).
  python experiments/fig_paper.py
Writes paper/fig_boundary.pdf (from results/boundary_map.csv) and paper/fig_certified.pdf
(from results/certified_R.csv).
"""
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

from _common import RESULTS, ROOT

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / 'paper'
OUT.mkdir(exist_ok=True)
INK, MUTED, AXIS, GRID = '#1f1e1b', '#5c5b55', '#b9b8b1', '#ecebe6'
BLUES = {.80: '#86b6ef', .85: '#3987e5', .90: '#1c5cab', .95: '#0d366b'}
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': AXIS, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.labelcolor': INK})


def tidy(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(True, which='major', color=GRID, lw=.5)


# Figure 1: R_{q,q}(x) - 1 in percent
d = pd.read_csv(RESULTS / 'boundary_map.csv')
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.axhline(0, color='#8a8983', lw=.8)
for q, g in d.groupby('q'):
    g = g.sort_values('x'); y = 100 * (g.R - 1)
    ax.plot(g.x, y, color=BLUES[q], lw=1.6, label=f'$q = {q:.2f}$')
ax.set_xscale('log'); ax.set_xlim(1e-4, 0.045); ax.set_ylim(-1.0, 0.45)
ax.set_xlabel(r'$x = D/t^2$'); ax.set_ylabel(r'$100\{R_{q,q}(x) - 1\}$')
tidy(ax); ax.legend(frameon=False, loc='lower left')
fig.tight_layout(); fig.savefig(OUT / 'fig_boundary.pdf')

# Figure 2: certified R over the whole range
c = pd.read_csv(RESULTS / 'certified_R.csv')
fig, ax = plt.subplots(figsize=(5.2, 3.0))
x = np.linspace(1e-4, 0.37, 400)
cq = 0.0190618
ax.plot(x, 1 + cq * np.sqrt(x), color=MUTED, lw=1, ls='--', label=r'small-noise bound $1 + c_{0.9}x^{1/2}$')
ub = 1 + np.sqrt(x) * stats.norm.ppf(1 - (0.9036 - 0.9) / 2)
ax.plot(x, ub, color=MUTED, lw=1, ls=':', label=r'shape-free bound, $p = 0.9036$')
for p, col in ((0.9, '#1c5cab'), (0.9036, '#86b6ef')):
    edge = (1 / stats.norm.ppf((1 + p) / 2)) ** 2          # beyond it no law is feasible
    g = c[np.isclose(c.p, p) & np.isfinite(c.U) & (c.x < edge)].sort_values('x')
    ax.plot(g.x, g.U, color=col, lw=1.6, label=rf'certified $R_{{{p:g},0.9}}(x)$')
ax.set_xlim(0, 0.37); ax.set_ylim(0, 1.3)
ax.set_xlabel(r'$x = D/t^2$'); ax.set_ylabel('latent radius / noisy threshold')
tidy(ax); ax.legend(frameon=False, loc='lower left')
fig.tight_layout(); fig.savefig(OUT / 'fig_certified.pdf'); fig.savefig(RESULTS / 'fig_certified.png', dpi=150)
print('wrote', OUT / 'fig_boundary.pdf', OUT / 'fig_certified.pdf')
