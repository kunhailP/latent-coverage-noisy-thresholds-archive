"""E34: head-to-head with the closest methods on the data of Table 1 (E20 seeds).

Setting: K = 110, var W = 1, heterogeneous known D_i = 0.577 lognormal(0, 0.7)/mean, four latent
laws, 150 data sets each. Every rule outputs a symmetric latent interval [-s, s]; conditional
coverage given the data is exact from closed-form latent distribution functions.

Rules added here (the certified log-concave rule, noisy conformal and the shape-free rule are
read from results/hetldc_synth.csv and results/shape_free.csv):
  cohen        deconvolution threshold of Cohen, Goldberger & Tirer (arXiv:2509.15120v2,
               Algorithm 1): masked ridge deconvolution of the histogram of V with the mixture
               kernel (1/K) sum N(0, D_i), bin width 0.01 (their default), lambda = 0.01, threshold scanned down
               from the marginal noisy conformal threshold in steps of 0.01 while the estimated
               clean coverage stays >= 0.9. No latent guarantee. With K = 110 most bins are
               empty and the masked fit collapses, so cohen_stable uses bin width 0.2 and fits
               the empty bins as zeros (no mask); both are reported. Their method is
               homoscedastic; the mixture kernel is our adaptation.
  latentcp     LatentCP single level (Zheng, Zhou & Zhu, arXiv:2608.03607, eqs 3-5), gamma =
               alpha/2: C = [-q, q] with q the noisy conformal threshold at level 1 - gamma, and
               U(d) = {w : p_gamma(w, d) >= 1 - gamma/alpha}. The latent set depends on the new
               unit's kernel d; an unsampled area has none, so d_new is drawn afresh from the
               distribution that generated the D_i (400 draws per data set; exchangeability as in
               their Assumption 1) and coverage and width average over that draw. Marginal
               guarantee.
  latentcp_tuned, latentcp_multi
               the tuned single level and the two-level (Multi-gamma, eq 6) versions, tuned on a
               random half of the data (their Algorithms 1-2, criterion: mean width on the tuning
               half) and calibrated on the other half. Marginal guarantee.
  latentcp_pac single level with the order statistic chosen so that the noisy coverage of C is at
               least 1 - gamma with probability 0.95 (Beta quantile, as for the other PAC rules);
               then pr(W_new not in U | data) <= alpha on that event (our adaptation, not in the
               paper). gamma = 0.05.
  noisy_threshold, simple_shrink
               the analytic rules of Proposition 3 and Corollary 2 (uai.procedures): the noisy
               threshold at rank 105, valid for unknown heterogeneous variances, and
               max{0, T - 0.114 D_min^{1/2}}.

  python experiments/e34_competitors.py [reps] [procs] [shapes=a,b,...]
With `shapes=a,b` only those laws are recomputed; rows of the other laws are kept. With
`analytic-shapes=a,b` only the two analytic rules are computed for those laws (seconds); the
LatentCP and deconvolution comparison is reported for the first four laws only.
Writes results/competitors.csv and results/competitors_summary.csv.
"""
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq, lsq_linear
from scipy.special import ndtr

from _common import RESULTS
from e12_conditional_synth import draw
from uai.latent_laws import cdf
from uai.procedures import noisy_threshold_halfwidth, simple_shrink_halfwidth

warnings.filterwarnings('ignore')
K, DBAR, ALPHA, DELTA = 110, 0.577, 0.10, 0.05
SHAPES = ['normal', 'laplace', 'gamma2_skew', 'trunc_laplace',
          'trunc_exp', 'bimodal_blc', 't3_not_LC']   # same order and seeds as e20


def data(shape, seed):
    rng = np.random.default_rng(seed)
    w = rng.lognormal(0, .7, K); D = DBAR * w / np.exp(.7**2 / 2)
    V = draw(shape, K, rng) + rng.normal(0, np.sqrt(D))
    return V, D


def conformal_q(scores, level):
    """Split-conformal threshold at level `level` (marginal): the ceil((n+1) level)-th order statistic."""
    n = len(scores); k = int(np.ceil((n + 1) * level))
    return np.inf if k > n else np.sort(scores)[k - 1]


def pac_q(scores, level, delta=DELTA):
    """Smallest order statistic whose Beta(k, n+1-k) delta-quantile is >= level."""
    n = len(scores); s = np.sort(scores)
    for k in range(1, n + 1):
        if stats.beta.ppf(delta, k, n + 1 - k) >= level:
            return s[k - 1]
    return np.inf


# --- Cohen, Goldberger & Tirer -------------------------------------------------------------

def cohen(V, D, h=0.05, lam=0.01, step=0.01, masked=True):
    sd = np.sqrt(D.max())
    lo, hi = V.min() - 4 * sd, V.max() + 4 * sd
    edges = np.arange(lo, hi + h, h); mids = (edges[:-1] + edges[1:]) / 2; L = len(mids)
    hist = np.histogram(V, edges)[0] / (len(V) * h)                   # density of V
    # mixture kernel on the grid offsets
    m = int(np.ceil(4 * sd / h)); off = np.arange(-m, m + 1) * h
    kern = np.mean(stats.norm.pdf(off[:, None], scale=np.sqrt(D)[None, :]), axis=1) * h
    A = np.zeros((L, L))                                              # (A p)_j = sum_l p_l k(j - l)
    for i, o in enumerate(range(-m, m + 1)):
        A += np.eye(L, k=-o) * kern[i]
    mask = hist > 1e-12 if masked else np.ones(L, bool)              # occupied bins (their eps)
    Am = np.vstack([A[mask], np.sqrt(lam) * np.eye(L)])
    bm = np.concatenate([hist[mask], np.zeros(L)])
    pW = lsq_linear(Am, bm, bounds=(0, 1 / h)).x
    cov = lambda q: h * pW[np.abs(mids) <= q].sum()
    q = conformal_q(np.abs(V), 1 - ALPHA)
    while q - step > 0 and cov(q - step) >= 1 - ALPHA:
        q -= step
    return q


# --- LatentCP ------------------------------------------------------------------------------

def p_gamma(w, d, q):
    return ndtr((q - w) / np.sqrt(d)) - ndtr((-q - w) / np.sqrt(d))


def radius_single(d, q, gamma):
    """U(d) = {w : p_gamma(w, d) >= 1 - gamma/alpha}; symmetric, decreasing in |w|."""
    target = 1 - gamma / ALPHA
    if not np.isfinite(q):
        return np.inf
    if p_gamma(0.0, d, q) < target:
        return 0.0
    return brentq(lambda w: p_gamma(w, d, q) - target, 0, q + 10 * np.sqrt(d))


def radius_multi(d, qs, gammas, weights):
    """U(d) = {w : sum_k w_k (1 - p_{gamma_k}(w, d))/gamma_k <= 1/alpha} (eq 6)."""
    if not all(np.isfinite(qs)):
        return np.inf
    e = lambda w: sum(wk * (1 - p_gamma(w, d, qk)) / gk for wk, qk, gk in zip(weights, qs, gammas))
    if e(0.0) > 1 / ALPHA:
        return 0.0
    return brentq(lambda w: e(w) - 1 / ALPHA, 0, max(qs) + 10 * np.sqrt(d))


def fresh_d(rng, m=400):
    """New units' noise variances from the generating distribution of the D_i."""
    return DBAR * rng.lognormal(0, .7, m) / np.exp(.7**2 / 2)


def latentcp_tuned(V, D, rng, multi, Dnew):
    idx = rng.permutation(K); tun, cal = idx[:K // 2], idx[K // 2:]
    St, Sc = np.abs(V[tun]), np.abs(V[cal]); m = len(tun)
    grid = [j / (m + 1) for j in range(1, m + 1) if 1 / (m + 1) <= j / (m + 1) <= ALPHA * 0.999]
    def width(gs, ws, S, dset):
        qs = [conformal_q(S, 1 - g) for g in gs]
        return np.mean([radius_multi(d, qs, gs, ws) for d in dset])
    if not multi:
        cands = [((g,), (1.0,)) for g in grid]
    else:
        cands = [((g1, g2), (w, 1 - w)) for i, g1 in enumerate(grid) for g2 in grid[i + 1:]
                 for w in np.arange(0.1, 1.0, 0.1)] + [((g,), (1.0,)) for g in grid]
    best = min(cands, key=lambda c: (width(c[0], c[1], St, D[tun]), c[0][0]))
    gs, ws = best
    qs = [conformal_q(Sc, 1 - g) for g in gs]
    return np.array([radius_multi(d, qs, gs, ws) for d in Dnew])   # one radius per d_new


def cov_of(shape, s):
    s = np.atleast_1d(s)
    c = np.where(np.isfinite(s), cdf(shape, s) - cdf(shape, -s), 1.0)
    return float(np.mean(c)), float(np.mean(2 * s))


def one_analytic(args):
    shape, seed = args
    V, D = data(shape, seed)
    out = {'noisy_threshold': cov_of(shape, noisy_threshold_halfwidth(V)),
           'simple_shrink': cov_of(shape, simple_shrink_halfwidth(V, D.min()))}
    return [dict(shape=shape, seed=seed, method=k, cov=c, width=w) for k, (c, w) in out.items()]


def one(args):
    shape, seed = args
    V, D = data(shape, seed)
    rng = np.random.default_rng(seed + 7)
    S = np.abs(V)
    out = {}
    out['cohen'] = cov_of(shape, cohen(V, D, h=0.01))
    out['cohen_stable'] = cov_of(shape, cohen(V, D, h=0.2, masked=False))
    Dnew = fresh_d(rng)
    g = ALPHA / 2
    q = conformal_q(S, 1 - g)
    out['latentcp'] = cov_of(shape, np.array([radius_single(d, q, g) for d in Dnew]))
    out['latentcp_tuned'] = cov_of(shape, latentcp_tuned(V, D, rng, False, Dnew))
    out['latentcp_multi'] = cov_of(shape, latentcp_tuned(V, D, rng, True, Dnew))
    qp = pac_q(S, 1 - g)
    out['latentcp_pac'] = cov_of(shape, np.array([radius_single(d, qp, g) for d in Dnew]))
    out['noisy_threshold'] = cov_of(shape, noisy_threshold_halfwidth(V))
    out['simple_shrink'] = cov_of(shape, simple_shrink_halfwidth(V, D.min()))
    return [dict(shape=shape, seed=seed, method=k, cov=c, width=w) for k, (c, w) in out.items()]


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    jobs = [(s, 40000 + 1000 * i + r) for i, s in enumerate(SHAPES) for r in range(reps)]
    arg = sys.argv[3] if len(sys.argv) > 3 else ''
    analytic = arg.startswith('analytic-shapes=')
    only = arg.split('=', 1)[1].split(',') if arg.startswith(('shapes=', 'analytic-shapes=')) else None
    keep_laws = only if only is not None else SHAPES[:4]           # full comparison: first four laws
    run = [j for j in jobs if j[0] in keep_laws]
    with Pool(procs) as pool:
        rows = [r for rs in pool.map(one_analytic if analytic else one, run, chunksize=2) for r in rs]
    d = pd.DataFrame(rows)
    if only is not None:
        prev = pd.read_csv(RESULTS / 'competitors.csv')
        keep = prev[~prev['shape'].isin(only) & ~prev.method.isin(['HetLDC_cert', 'CP_PAC', 'FH_PAC',
                                                                    'shape_free_k107'])]
        d = pd.concat([keep, d])
    old = pd.read_csv(RESULTS / 'hetldc_synth.csv')
    old = old[old.method.isin(['HetLDC_cert', 'CP_PAC', 'FH_PAC'])].assign(width=lambda x: 2 * x.half)
    sf = pd.read_csv(RESULTS / 'shape_free.csv')
    sf = sf[sf.k == 107].assign(method='shape_free_k107', width=lambda x: 2 * x.half)
    d = pd.concat([d, old[['shape', 'seed', 'method', 'cov', 'width']],
                   sf[['shape', 'seed', 'method', 'cov', 'width']]])
    d = d[d.seed.isin([j[1] for j in jobs])]
    d.to_csv(RESULTS / 'competitors.csv', index=False)
    summ = d.groupby(['shape', 'method']).agg(reps=('cov', 'size'), mean_cov=('cov', 'mean'),
                                              reliability=('cov', lambda c: np.mean(c >= 0.9)),
                                              width=('width', 'mean')).reset_index()
    summ.to_csv(RESULTS / 'competitors_summary.csv', index=False)
    print(summ.round(4).to_string())
