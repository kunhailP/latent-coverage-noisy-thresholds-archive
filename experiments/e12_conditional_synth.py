"""E12. Conditional coverage without evaluation noise (synthetic, law of new areas known).

Calibration areas: W_c iid G, V_c = W_c + e_c, e_c ~ N(0, D_c), K = 110, Var W = 1,
mean D = 0.577 (apipop-like D/A). D_c common or heterogeneous (lognormal, sd 0.7, mean 1).
Every rule sees D_hat_c = D_c chi2_10 / 10 (FH treats it as known; LDC pools it).
For a fixed calibration set D the conditional coverage Pr(W_new in C | D) = G(hi) - G(lo) is
computed from a 2e6-draw reference sample of G, so the only randomness left is D.

Rules
  FH_normal, FH_boot       : Fay-Herriot, REML; bootstrap pivot, B = 300
  noisy_CP (k=100), CP_k102, LDC_k102   : marginal rules
  CP_PAC  (k=105)          : Beta law -> Pr_D[noisy cond. coverage >= .9] >= .95
  LDC_PAC (k=105, eta=.01) : same t, shrunk by r(D_low / t^2)
  FH_PAC                   : parametric-bootstrap tolerance multiplier (delta = .05)
Summaries per rule: mean conditional coverage, its sd, Pr_D[cov >= .90], 5% quantile, width.
Equal-reliability widths: each rule's half-width is rescaled by one common factor lam per
(shape, D regime, rule) so that (a) mean coverage = .90 or (b) Pr_D[cov >= .90] = .95; the
resulting mean widths compare centre/scale estimators at the same target (oracle calibration,
a diagnostic, not a procedure).
  python experiments/e12_conditional_synth.py [reps] [procs]
Writes results/conditional_synth_summary.csv, results/conditional_synth_matched.csv.
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

from _common import DATA, RESULTS
from uai.procedures import (ShrinkTable, conformal_threshold, fay_herriot_boot_pac,
                            fay_herriot_reml, noise_lower_bound, pac_rank)

K, DBAR, NU, LEV = 110, 0.577, 10, 0.90
SHAPES = ['normal', 'laplace', 'gamma2_skew', 'shifted_normal', 'trunc_laplace', 'trunc_exp',
          't3_not_LC']
REGIMES = ['common', 'hetero']


def draw(kind, n, rng):
    if kind == 'normal': return rng.normal(size=n)
    if kind == 'laplace': return rng.laplace(size=n) / np.sqrt(2)
    if kind == 'gamma2_skew': return (rng.gamma(2, size=n) - 2) / np.sqrt(2)
    if kind == 'shifted_normal': return rng.normal(size=n) + .3
    if kind == 'trunc_laplace':
        c = 2.705; u = rng.uniform(size=n)
        w = -np.log(1 - u * (1 - np.exp(-c))) * rng.choice([-1, 1], n)
        return w / np.sqrt((2 - np.exp(-c) * (c * c + 2 * c + 2)) / (1 - np.exp(-c)))
    if kind == 'trunc_exp':           # log-affine on a segment (the LDC extremal family), skewed
        b = 4.0; u = rng.uniform(size=n)
        y = np.log1p(u * np.expm1(b)) / b                          # density prop. exp(b y) on [0,1]
        m = 1 / (1 - np.exp(-b)) - 1 / b
        v = 1 / b**2 - np.exp(b) / np.expm1(b)**2
        return (y - m) / np.sqrt(v)
    if kind == 't3_not_LC': return rng.standard_t(3, n) / np.sqrt(3)
    if kind == 'bimodal_blc':         # bi-log-concave, bimodal, not log-concave (latent_laws.BIMODAL)
        from uai.latent_laws import BIMODAL
        p, m1, m2, s = BIMODAL
        return np.where(rng.uniform(size=n) < p, m1, m2) + s * rng.normal(size=n)


def worker(args):
    kind, regime, seed, reps = args
    rng = np.random.default_rng(seed); table = ShrinkTable('0.1')
    ref = np.sort(draw(kind, 2_000_000, np.random.default_rng(seed + 99)))
    G = lambda x: np.searchsorted(ref, x, side='right') / len(ref)
    k0, kp = int(np.ceil((K + 1) * LEV)), pac_rank(K, LEV, .04)
    z = stats.norm.ppf(.5 + LEV / 2)
    rows = []
    for rep in range(reps):
        if regime == 'common':
            Dc = np.full(K, DBAR)
        else:
            w = rng.lognormal(0, .7, K); Dc = DBAR * w / np.exp(.7**2 / 2)
        W = draw(kind, K, rng); V = W + rng.normal(0, np.sqrt(Dc))
        Dh = Dc * rng.chisquare(NU, K) / NU
        D_low = noise_lower_bound(Dh, K * NU)
        mu, A, vmu = fay_herriot_reml(V, Dh); se = np.sqrt(A + vmu)
        piv, lam = fay_herriot_boot_pac(V, Dh, mu, A, rng, level=LEV)
        ql, qu = np.quantile(piv, [.05, .95])
        t0, t102, tp = (conformal_threshold(V, k) for k in (k0, 102, kp))
        ints = {'FH_normal': (mu, -z * se, z * se), 'FH_boot': (mu, ql * se, qu * se),
                'FH_PAC': (mu, -lam * se, lam * se),
                'noisy_CP': (0, -t0, t0), 'CP_k102': (0, -t102, t102), 'CP_PAC': (0, -tp, tp)}
        for name, t in (('LDC_k102', t102), ('LDC_PAC', tp)):
            s = t * table(D_low / t**2); ints[name] = (0, -s, s)
        for m, (c, lo, hi) in ints.items():
            rows.append(dict(shape=kind, regime=regime, rep=rep, method=m, centre=c, lo=lo, hi=hi,
                             cov=G(c + hi) - G(c + lo)))
    return rows, (kind, regime, ref[::200])


def matched(g, ref):
    """Rescale offsets by one lam to hit (a) mean cov .90, (b) Pr[cov >= .90] = .95."""
    G = lambda x: np.searchsorted(ref, x, side='right') / len(ref)
    c, lo, hi = g.centre.values, g.lo.values, g.hi.values
    cov = lambda l: G(c + l * hi) - G(c + l * lo)
    la = brentq(lambda l: cov(l).mean() - LEV, 1e-3, 20)
    lb = brentq(lambda l: np.mean(cov(l) >= LEV) - .95 + 1e-9, 1e-3, 20, xtol=1e-6)
    ca, cb = cov(la), cov(lb)
    return dict(lam_mean=la, width_mean_matched=np.mean(la * (hi - lo)), sd_cov_mean_matched=ca.std(),
                share_below_mean_matched=np.mean(ca < LEV),
                lam_pac=lb, width_pac_matched=np.mean(lb * (hi - lo)))


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    chunks = 10
    jobs = [(s, r, 20000 + 1000 * i + 100 * j + c, reps // chunks)
            for i, s in enumerate(SHAPES) for j, r in enumerate(REGIMES) for c in range(chunks)]
    with Pool(procs) as pool:
        out = pool.map(worker, jobs)
    r = pd.DataFrame([row for rows, _ in out for row in rows])
    refs = {(k, g): ref for _, (k, g, ref) in out}
    r.to_pickle(DATA / 'conditional_synth_reps.pkl')
    summ, mat = [], []
    for (s, reg, m), g in r.groupby(['shape', 'regime', 'method']):
        summ.append(dict(shape=s, regime=reg, method=m, reps=len(g), mean_cov=g['cov'].mean(),
                         sd_cov=g['cov'].std(), pr_cov_ge_90=np.mean(g['cov'] >= LEV),
                         q05_cov=g['cov'].quantile(.05), width=(g.hi - g.lo).mean()))
        mat.append(dict(shape=s, regime=reg, method=m, **matched(g, refs[(s, reg)])))
    S = pd.DataFrame(summ); M = pd.DataFrame(mat)
    S.to_csv(RESULTS / 'conditional_synth_summary.csv', index=False)
    M.to_csv(RESULTS / 'conditional_synth_matched.csv', index=False)
    pd.set_option('display.width', 200)
    print(S.round(3).to_string(index=False)); print(M.round(3).to_string(index=False))
