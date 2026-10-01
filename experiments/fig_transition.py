"""Figure 1 of the manuscript: the transition coefficient L_q(kappa) (Theorem 3) and the upper
bound min{c_q, (kappa^2 + 1)^{1/2} - kappa}, q = 0.8, 0.9, 0.95.

  python experiments/fig_transition.py      -> paper/fig_transition.pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from _common import ROOT
from uai.extremal import transition_constant, transition_upper

kap = np.geomspace(1, 1000, 90)
styles = {0.8: '-', 0.9: '--', 0.95: ':'}
fig, ax = plt.subplots(figsize=(6.4, 2.2))
for q, ls in styles.items():
    L = np.array([transition_constant(q, k) for k in kap])
    U = np.array([transition_upper(q, k) for k in kap])
    ax.plot(kap, 100 * L, 'k', ls=ls, lw=1.4, label=f'$q={q}$')
    ax.plot(kap, 100 * U, color='0.6', ls=ls, lw=0.9)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'Distance of the mean from the boundary, $\kappa$')
ax.set_ylabel(r'Leading coefficient $\times 100$')
ax.legend(frameon=False, loc='lower left')
fig.tight_layout()
fig.savefig(ROOT / 'paper' / 'fig_transition.pdf')
