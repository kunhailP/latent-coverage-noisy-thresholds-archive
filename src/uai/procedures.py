"""Interval rules for an unsampled area, all centred at a separately trained predictor.

Scores on calibration areas: V_c = (survey estimate) - (centre), with design variance D_c.
Target on a new area: W = theta - centre.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
_FEASIBLE_X = 0.3696          # above this, P(|e| <= t) < .9: data contradict the noise level


class ShrinkTable:
    """Legacy. Lookup of grid values of the shrink function r(x) = R_{1-a,1-a}(x). The values are
    converged maxima over the extremal family, hence lower bounds of R: not a valid radius.
    For valid radii use the analytic rules below (proved, Arb-certified constants) or
    CertifiedShrinkTable (numerically computed upper bounds, double precision).

    Other alphas read results/r_table.json (E04, differential evolution with Gauss-Legendre
    quadrature), whose values at x = 0.001 exceed the bound of Theorem 3 because the quadrature
    is inaccurate there; they are not used in the paper and a warning is issued.

    alpha = '0.1' uses the exact two-dimensional reduction (results/r_exact_p0.9_q0.9.csv, E16,
    plus the fine small-x grid of E19). r rises for small x (r > 1: widen) to a peak near
    x = .0023 and then decreases. At or right of the peak the value at the grid point at or
    below x is >= r(x); left of it the overall maximum is returned. Other alphas use the E04
    differential-evolution table (same lookup, first value below the grid).
    """

    def __init__(self, alpha='0.1', path=ROOT / 'results' / 'r_table.json'):
        if alpha == '0.1' and (ROOT / 'results' / 'r_exact_p0.9_q0.9.csv').exists():
            import pandas as pd
            d = pd.read_csv(ROOT / 'results' / 'r_exact_p0.9_q0.9.csv')[['x', 'R']]
            b = ROOT / 'results' / 'boundary_map.csv'
            if b.exists():
                e = pd.read_csv(b)
                d = pd.concat([d, e[np.isclose(e.q, 0.9)][['x', 'R']]])
            d = d[np.isfinite(d.R)].sort_values('x')
            self.x, self.r = d.x.values, d.R.values
        else:
            import warnings
            warnings.warn('ShrinkTable: r_table.json (E04) is inaccurate at small x and exceeds '
                          'Theorem 3 at x = 0.001; not a valid radius', stacklevel=2)
            tab = json.load(open(path))[alpha]
            self.x = np.array(sorted(float(k) for k in tab))
            self.r = np.array([tab[str(k)] for k in self.x])
        self.peak = int(np.argmax(self.r))

    def __call__(self, x):
        if x >= _FEASIBLE_X:        # alpha = .10 only; the data contradict the noise level
            return 1.0
        i = np.searchsorted(self.x, x, side='right') - 1
        if i <= self.peak:
            return float(self.r[self.peak])
        return float(self.r[i])



class CertifiedShrinkTable:
    """Numerically certified upper bound U(x) >= R_{p,.9}(x) for p in {.90, .9036, .9068} (E24):
    double-precision branch and bound, not interval arithmetic (see README, Precision).

    Grid points x_i >= .002 carry branch-and-bound certificates U_i (`uai.certify`). For any x
    the lookup returns min over x_i <= x of U_i + c_q sqrt(x - x_i) (Supplementary Proposition S5),
    together with the closed bound of Theorem 3 (p = q: 1 + c_q sqrt x) or the p >= q bound of Section 4.3
    (p = .9036, .9068: 1 + C sqrt x, C = -.057, -.114). The constants are proved upper bounds in ball
    arithmetic (E27, `uai.interval`). Above the feasibility
    edge (P(|e| <= 1) < p) the data contradict the noise level and 1 is returned, as in
    ShrinkTable. Certificates rest on Supplementary Proposition S3 (extremal family) and double-precision
    closed forms with a 1e-9 margin.
    """
    C_Q = 0.019062                       # c_.9 in [0.01906181, 0.01906188] (E27)
    C_SPLIT = {0.9: 0.019062, 0.9036: -0.057,   # C_{.9036,.9} in [-0.0571845, -0.057] (E27)
               0.9068: -0.114}              # C_{.9068,.9} in [-0.1149813, -0.114] (E27)

    def __init__(self, p=0.9, path=ROOT / 'results' / 'certified_R.csv'):
        import pandas as pd
        d = pd.read_csv(path)
        d = d[np.isclose(d.p, p) & np.isfinite(d.U)].sort_values('x')
        self.p, self.x, self.u = p, d.x.values, d.U.values
        self.c_split = self.C_SPLIT[round(p, 4)]
        self.x_edge = (1 / stats.norm.ppf((1 + p) / 2)) ** 2

    def __call__(self, x):
        if x >= self.x_edge:
            return 1.0
        best = 1 + self.c_split * np.sqrt(x)
        m = self.x <= x
        if m.any():
            best = min(best, float(np.min(self.u[m] + self.C_Q * np.sqrt(x - self.x[m]))))
        return float(best)

    def envelope(self, x_low):
        """Certified upper bound of sup{R(x) : x_low <= x < x_edge}, for use when only a lower
        bound x_low of the scaled noise variance is known. R is not monotone in x (it rises for
        small x when p = q), so the value at x_low alone is not enough. Between grid points
        Supplementary Proposition S5 gives R(x) <= U_j + c_q sqrt(x_{j+1} - x_j) on [x_j, x_{j+1}], and below the
        first grid point the closed bound 1 + C sqrt(x) is monotone, so its ends suffice."""
        if x_low >= self.x_edge:
            return 1.0
        keep = self.x < self.x_edge
        xs, us = self.x[keep], self.u[keep]
        closed = lambda x: 1 + self.c_split * np.sqrt(x)      # monotone in x
        vals = []
        if x_low < xs[0]:
            vals += [closed(x_low), closed(xs[0])]
        right = np.append(xs[1:], self.x_edge)
        m = right > x_low
        left = np.maximum(xs[m], x_low)
        by_grid = us[m] + self.C_Q * np.sqrt(right[m] - xs[m])
        by_closed = np.maximum(closed(left), closed(right[m]))
        vals += list(np.minimum(by_grid, by_closed))
        return float(max(vals))

def conformal_threshold(scores, k):
    """k-th smallest |score| (1-based)."""
    return np.sort(np.abs(scores))[k - 1]


def noise_lower_bound(D_hat, df_total, eta=0.01):
    """Pooled chi-square lower confidence bound for a common noise variance."""
    return np.mean(D_hat) * df_total / stats.chi2.ppf(1 - eta, df_total)


def ldc_halfwidth(scores, D_low, k, table):
    t = conformal_threshold(scores, k)
    return t * table(D_low / t**2)


def fay_herriot_reml(V, D):
    """Intercept-only Fay-Herriot fit by REML with D treated as known. Returns (mu, A, var_mu)."""
    def nll(logA):
        A = np.exp(logA); w = 1 / (A + D); mu = (w * V).sum() / w.sum()
        return 0.5 * (np.log(A + D).sum() + np.log(w.sum()) + (w * (V - mu)**2).sum())
    r = minimize_scalar(nll, bounds=(np.log(1e-6 * np.var(V) + 1e-12),
                                     np.log(10 * np.var(V) + 1e-9)), method='bounded')
    A = np.exp(r.x); w = 1 / (A + D)
    return (w * V).sum() / w.sum(), A, 1 / w.sum()


def fay_herriot_boot_pivot(V, D, mu, A, rng, B=300):
    """Parametric-bootstrap draws of (theta* - mu*) / sqrt(A* + v*) for a new area."""
    piv = np.empty(B)
    for b in range(B):
        Vb = mu + rng.normal(0, np.sqrt(A), len(V)) + rng.normal(0, np.sqrt(D))
        thb = mu + rng.normal(0, np.sqrt(A))
        mub, Ab, vb = fay_herriot_reml(Vb, D)
        piv[b] = (thb - mub) / np.sqrt(Ab + vb)
    return piv


def pac_rank(K, level=0.90, delta=0.05):
    """Smallest k with P(Beta(k, K+1-k) >= level) >= 1 - delta. For K i.i.d. scores with a
    continuous law, the k-th order statistic then has coverage >= level for a new independent
    score, conditionally on the calibration sample, with probability >= 1 - delta. (With
    independent non-identical scores see Supplementary Proposition S2 / `hetldc_parts`, which needs
    k >= K p + 1; exchangeability alone gives only the marginal guarantee.)"""
    for k in range(1, K + 1):
        if stats.beta.sf(level, k, K + 1 - k) >= 1 - delta:
            return k
    return None


def fay_herriot_boot_pac(V, D, mu, A, rng, B=300, level=0.90, delta=0.05):
    """One parametric bootstrap giving (i) pivot draws as in fay_herriot_boot_pivot and
    (ii) a tolerance multiplier lam: the (1 - delta) bootstrap quantile of the multiplier each
    refit needs for its interval mu* +- lam sqrt(A* + v*) to cover N(mu, A) with prob. level."""
    piv, lam = np.empty(B), np.empty(B)
    sA = np.sqrt(max(A, 1e-12))
    for b in range(B):
        Vb = mu + rng.normal(0, np.sqrt(A), len(V)) + rng.normal(0, np.sqrt(D))
        thb = mu + rng.normal(0, np.sqrt(A))
        mub, Ab, vb = fay_herriot_reml(Vb, D)
        se = np.sqrt(Ab + vb)
        piv[b] = (thb - mub) / se
        cov = lambda l: (stats.norm.cdf((mub + l * se - mu) / sA)
                         - stats.norm.cdf((mub - l * se - mu) / sA) - level)
        hi = 1.0
        while cov(hi) < 0 and hi < 1e4:
            hi *= 2
        lam[b] = brentq(cov, 0, hi) if hi < 1e4 else hi
    return piv, np.quantile(lam, 1 - delta)


def union_bound_halfwidth(t, D_up, p, q):
    """Assumption-light baseline: if P(|V| <= t) >= p > q, V = W + e, e ~ N(0, D), D <= D_up,
    then P(|W| <= s) >= q for s = t + sqrt(D_up) z_{1-(p-q)/2}, because |W| <= |V| + |e|.
    Needs no shape assumption; used as a safe reference against which LDC is measured."""
    if p <= q:
        return np.inf
    return t + np.sqrt(D_up) * stats.norm.ppf(1 - (p - q) / 2)


def clopper_pearson_lower(n_success, n, delta):
    """One-sided Clopper-Pearson lower confidence bound for an i.i.d. Bernoulli success
    probability. It is NOT valid in general for the mean of independent non-identical trials:
    n = 2, probabilities (.0505, 0), delta = .05 gives mean .02525 but a bound .02532 after one
    success, an event of probability .0505 > delta. Hoeffding's comparison needs a tail
    condition such as k >= n p + 1 (used in `hetldc_parts`). Not used by any procedure."""
    return 0.0 if n_success == 0 else float(stats.beta.ppf(delta, n_success, n - n_success + 1))


def shrink_mix(p, q, xs, wts=None, betas=None, ells=None, m=100):
    """R_{p,q} for heterogeneous known noise (average kernel over scaled variances xs):
    grid over (slope, length) with the exact endpoint reduction, then a local polish.
    A converging feasible value, not a certified upper bound."""
    from scipy.optimize import minimize
    from uai.extremal import shape_values
    betas = np.r_[0.0, np.exp(np.linspace(np.log(1e-2), np.log(1e3), 16))] if betas is None else betas
    ells = np.exp(np.linspace(np.log(1e-3), np.log(20), 30)) if ells is None else ells
    best, arg = -np.inf, None
    for beta in betas:
        v = shape_values(p, q, beta, ells, xs, wts, m=m)
        i = int(np.argmax(v))
        if v[i] > best:
            best, arg = v[i], (beta, ells[i])
    if arg is None:
        return np.inf                    # no log-concave law fits: fall back to no guarantee
    f = lambda z: -shape_values(p, q, np.exp(z[0]), [np.exp(z[1])], xs, wts, m=m)[0]
    r = minimize(f, [np.log(max(arg[0], 1e-3)), np.log(arg[1])], method='Nelder-Mead',
                 options={'xatol': 1e-6, 'fatol': 1e-10, 'maxiter': 60})
    return max(best, -r.fun) if np.isfinite(r.fun) else best


def quantised_kernel(xs, n_pts=32):
    """Compress scaled variances xs to n_pts group means (weights = group sizes) and return them
    with eps >= sup_w |gbar(w) - gbar_quantised(w)|, so that a constraint int gbar dG >= p can be
    replaced by the valid int gbar_q dG >= p - eps. The sup is taken on a grid of spacing h and
    padded by M h^2 / 8, M a bound on the second derivative of the difference (the maximum of a
    C^2 function between two grid points exceeds the larger endpoint value by at most M h^2/8)."""
    from scipy.special import ndtr
    xs = np.sort(np.asarray(xs, dtype=float))
    uniq, counts = np.unique(xs, return_counts=True)
    if len(uniq) <= n_pts:                            # exact: equal variances merged
        return (uniq, counts / len(xs)), 0.0
    tiny = xs < 1e-3                                  # near-indicator kernels: kept exactly
    groups = [xs[i:i + 1] for i in np.where(tiny)[0]] + \
        [g for g in np.array_split(xs[~tiny], max(n_pts - tiny.sum(), 1)) if len(g)]
    pts = np.array([np.mean(g) for g in groups]); wts = np.array([len(g) for g in groups]) / len(xs)
    span = 1 + 8 * np.sqrt(xs.max())
    wg = np.arange(-span, span + 1e-3, 1e-3)          # beyond +-span both kernels are < 1e-15
    kern = lambda x: ndtr((1 - wg[:, None]) / np.sqrt(x)) - ndtr((-1 - wg[:, None]) / np.sqrt(x))
    diff = np.abs(kern(xs).mean(1) - kern(pts) @ wts)
    # |g_x''| <= 2 phi(1) / x; a group's error has second derivative <= 4 phi(1) / min(group)
    M = sum(w * 4 * 0.2420 / g.min() for g, w in zip(groups, wts) if len(g) > 1)
    return (pts, wts), float(diff.max() + M * 1e-6 / 8)


def hetldc_halfwidth(V, D, k=None, q=0.90, delta=0.05, n_pts=32):
    """Grid value of the heterogeneous-noise radius: T times a lower bound of R^mix, so NOT a
    valid radius. `hetldc_certified` returns a numerically computed upper bound of R^mix
    (double precision, not interval arithmetic); the analytic rules `noisy_threshold_halfwidth`
    and `simple_shrink_halfwidth` rest on proofs and Arb-certified constants only.

    Heterogeneous-noise LDC with an order-statistic threshold T = |V|_(k).

    Let Fbar be the average CDF of |V_i| (independent, non-identical because D_i differ). For
    t_p = Fbar^{-1}(p), {Fbar(T) < p} = {#{|V_i| <= t_p} >= k}; the count is Poisson-binomial
    with mean K p, so by Hoeffding (1956) its upper tail at k >= K p + 1 is at most the
    Bin(K, p) tail. Hence p_k = Beta(k, K + 1 - k) delta-quantile still gives
    P(Fbar(T) >= p_k) >= 1 - delta. On that event int gbar_T dG >= p_k with gbar_T the average
    Gaussian kernel at x_i = D_i / T^2, and s = T * R_{p_k - eps, q}(kernel) covers the latent
    target with probability >= q for every log-concave G (eps: kernel quantisation error).
    Assumes W_i iid log-concave, e_i ~ N(0, D_i) independent of W with D_i known."""
    T, p_k, eps, (pts, wts), R = hetldc_parts(V, D, k, q, delta, n_pts)
    return T * R


def hetldc_parts(V, D, k=None, q=0.90, delta=0.05, n_pts=32):
    """Pieces of HetLDC: threshold T, level p_k, quantisation error eps, compressed kernel
    (pts, wts) and the grid value R of R^mix_{p_k - eps, q} (a lower bound of it)."""
    V, D = np.asarray(V), np.asarray(D)
    K = len(V)
    k = pac_rank(K, q, delta) if k is None else k
    p_k = float(stats.beta.ppf(delta, k, K + 1 - k))
    if k < K * p_k + 1:
        raise ValueError('Hoeffding comparison (Supplementary Proposition S2) needs k >= K p_k + 1')
    T = np.sort(np.abs(V))[k - 1]
    (pts, wts), eps = quantised_kernel(D / T**2, n_pts)
    return T, p_k, eps, (pts, wts), shrink_mix(p_k - eps, q, pts, wts=wts)


def hetldc_certified(V, D, k=None, q=0.90, delta=0.05, n_pts=32, parts=None,
                     tols=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1)):
    """HetLDC with a numerically certified radius (double-precision closed forms, margin 1e-9;
    not interval arithmetic): the smallest R (1 + tol) that the branch and bound of
    `uai.certify` clears for the mixture kernel at level p_k - eps. If none clears, the
    shape-free union bound with the largest D is returned. Returns (half-width, tol);
    tol is nan for the fallback."""
    from uai.certify import certified_upper
    T, p_k, eps, (pts, wts), R = hetldc_parts(V, D, k, q, delta, n_pts) if parts is None else parts
    if not np.isfinite(R):
        return np.inf, np.nan
    s, tol, _ = certified_upper(p_k - eps, q, pts, R, rel_tols=tols, ws=wts, max_boxes=8_000_000)
    if np.isfinite(s):
        return T * s, tol
    return union_bound_halfwidth(T, np.max(D), p_k - eps, q), np.nan


def shape_free_markov(V, D, k, q=0.90, delta=0.05):
    """Shape-free PAC radius in the spirit of LatentCP (Zheng, Zhou & Zhu 2026): no assumption on
    the latent law beyond independence of the Gaussian noise (known D_i). On the order-statistic
    event, int gbar_T dG >= p_k. With m = gbar_T(0) = max gbar_T, Y = m - gbar_T(W) >= 0 has
    E Y <= m - p_k, so by Markov P(gbar_T(W) < t) <= (m - p_k)/(m - t) = 1 - q at
    t = (p_k - q m)/(1 - q) (needs p_k > q m). gbar_T is symmetric and decreasing in |w|, so the
    set {gbar_T >= t} is [-r, r]. Using 1 in place of m (the weaker version) and c = 1/2 recovers
    the 1 - 2 alpha bound of Guille-Escuret & Ndiaye (2024). Returns (half-width T r, p_k, T)."""
    from scipy.optimize import brentq
    from scipy.special import ndtr
    V, D = np.asarray(V), np.asarray(D)
    K = len(V)
    p_k = stats.beta.ppf(delta, k, K + 1 - k)
    if k < K * p_k + 1 or p_k <= q:
        raise ValueError('needs k >= K p_k + 1 (Supplementary Proposition S2) and p_k > q')
    T = np.sort(np.abs(V))[k - 1]
    sd = np.sqrt(D) / T
    gbar = lambda w: np.mean(ndtr((1 - w) / sd) - ndtr((-1 - w) / sd))
    m = gbar(0.0)
    thr = (p_k - q * m) / (1 - q)
    if thr <= 0 or m <= thr:
        return np.inf, p_k, T
    r = brentq(lambda w: gbar(w) - thr, 0.0, 1 + 10 * sd.max(), xtol=1e-12)
    return T * r, p_k, T


# ---------------------------------------------------------------------------------------------
# Analytic rules for heterogeneous, possibly unknown, Gaussian noise variances (Theorem 2 and
# Supplementary Corollary S1 of the paper). Both rest only on the order-statistic event of Supplementary Proposition S2 and on
# ball-arithmetic certificates of the one-sided constants (results/interval_constants.json).
# ---------------------------------------------------------------------------------------------

# levels p with C_{p,q} <= 0 certified in ball arithmetic (E27): noisy coverage >= p at threshold t
# implies latent coverage >= q at t, for every Gaussian noise variance
CERTIFIED_SLACK_LEVEL = {0.8: 0.8050, 0.9: 0.9010, 0.95: 0.9502}
# certified C_{p,q} <= value < 0 (E27): latent radius <= t + value * D^{1/2}
CERTIFIED_SHRINK = {0.9: [(0.9036, -0.057), (0.9068, -0.114)]}


def certified_rank(K, q=0.90, delta=0.05, level=None):
    """Smallest rank k whose order-statistic level p_k (the delta-quantile of Beta(k, K + 1 - k))
    reaches `level` (default: the certified slack level CERTIFIED_SLACK_LEVEL[q]) and satisfies
    the Hoeffding condition k >= K p_k + 1 of Supplementary Proposition S2. None if no rank qualifies (for
    q = .9 and delta = .05 this happens exactly when K <= 28). The usual PAC rank
    (`pac_rank`, level q) can fall short: K = 46 gives rank 45 with p_k = .90098 < .901."""
    if level is None:
        if q not in CERTIFIED_SLACK_LEVEL:
            raise ValueError(f'no certified slack level for q = {q}; available: {sorted(CERTIFIED_SLACK_LEVEL)}')
        level = CERTIFIED_SLACK_LEVEL[q]
    for k in range(1, K + 1):
        p_k = float(stats.beta.ppf(delta, k, K + 1 - k))
        if p_k >= level and k >= K * p_k + 1:
            return k
    return None


def _order_statistic_level(V, k, delta):
    """(T, p_k) for T = |V|_(k), or (T, nan) if k violates the Hoeffding condition
    k >= K p_k + 1 of Supplementary Proposition S2 (then no level is guaranteed)."""
    V = np.asarray(V)
    K = len(V)
    if not 1 <= k <= K:
        raise ValueError(f'rank k = {k} outside 1..{K}')
    p_k = float(stats.beta.ppf(delta, k, K + 1 - k))
    T = np.sort(np.abs(V))[k - 1]
    return (T, p_k) if k >= K * p_k + 1 else (T, np.nan)


def noisy_threshold_halfwidth(V, k=None, q=0.90, delta=0.05):
    """The noisy threshold T = |V|_(k) itself, valid for the latent target when the noise
    variances are heterogeneous and unknown (Theorem 2).

    On the event of Supplementary Proposition S2, the average over i of pr(|W + e_i| <= T) is at least p_k. If
    pr(|W| <= T) < q, Corollary 1 (C_{p,q} <= 0 at p = CERTIFIED_SLACK_LEVEL[q]) forces every
    term below that level, hence their average too. So p_k >= that level gives
    pr_D{pr(|W_new| <= T | D) >= q} >= 1 - delta for every bi-log-concave G, whatever the D_i.
    The default rank is `certified_rank`. Returns inf if no rank reaches the certified level
    (K too small) or if a rank k given by the caller does not."""
    K = len(V)
    k = certified_rank(K, q, delta) if k is None else k
    if k is None:
        return np.inf
    T, p_k = _order_statistic_level(V, k, delta)
    return T if p_k >= CERTIFIED_SLACK_LEVEL[q] else np.inf      # nan (Hoeffding fails) -> inf


def simple_shrink_halfwidth(V, D_min, k=None, q=0.90, delta=0.05):
    """max{0, T + C sqrt(D_min)} (Supplementary Corollary S1) with the certified C = C_{p', q} < 0 for the
    largest certified p' <= p_k. On the event of Supplementary Proposition S2 some area i has
    pr(|W + e_i| <= T) >= p_k >= p', so the transfer bound R_{p',q}(x) <= 1 + C x^{1/2} gives
    Q_q(|W|) <= T + C D_i^{1/2} <= T + C D_min^{1/2}, as C < 0. D_min is a known lower bound on
    the D_i. The default rank is `certified_rank`; falls back to `noisy_threshold_halfwidth`
    (possibly inf) if no negative constant is certified at level p_k."""
    if not D_min >= 0:
        raise ValueError(f'D_min must be a nonnegative lower bound on the noise variances, got {D_min}')
    K = len(V)
    k = certified_rank(K, q, delta) if k is None else k
    if k is None:
        return np.inf
    T, p_k = _order_statistic_level(V, k, delta)
    usable = [c for p, c in CERTIFIED_SHRINK.get(q, []) if p <= p_k]      # empty if p_k is nan
    if not usable:
        return noisy_threshold_halfwidth(V, k, q, delta)
    return max(0.0, T + min(usable) * np.sqrt(D_min))
