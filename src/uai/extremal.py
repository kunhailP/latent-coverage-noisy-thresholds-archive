"""Closed-form computations for the latent-coverage transfer problem.

    R_{p,q}(x) = sup { Q_q(|W|) : W log-concave, P(|W + sqrt(x) Z| <= 1) >= p },  Z ~ N(0,1).

Reduction. For fixed s, "R <= s" means inf{ P(|W| < s) : P(|W + e| <= 1) >= p } >= q: a
linear objective and one linear constraint over log-concave laws. By Fradelizi-Guedon
localisation (support truncated to [-M, M], M -> infinity) the extreme points are Dirac masses
and laws with density prop. to exp(beta w) on a segment [a, b]. The family does not depend on
s, so R computed over it equals R over all log-concave laws. By the symmetry W -> -W we may
take beta >= 0.

For that family both P(W + e <= c) and P(|W| <= s) have closed forms (below), so no quadrature
is needed. Everything is evaluated in log space to stay finite for steep beta.
"""
import numpy as np
from scipy.optimize import NonlinearConstraint, brentq, differential_evolution, minimize, minimize_scalar
from scipy.special import erfinv, log_ndtr, ndtr

_BETA0 = 1e-7


def _log_diff_exp(u, v):
    """log(e^u - e^v) for u >= v; -inf when u = v (a zero difference, not an error)."""
    with np.errstate(divide='ignore'):
        return u + np.log1p(-np.exp(v - u))


def noisy_cdf(c, a, b, beta, x):
    """P(W + e <= c), W with density prop. to exp(beta w) on [a, b], e ~ N(0, x).

    With s = sqrt(x) and integration by parts,
      int_a^b e^{bw} Phi((c-w)/s) dw = (1/beta) [ e^{beta b} Phi((c-b)/s) - e^{beta a} Phi((c-a)/s)
          + e^{beta c + beta^2 x / 2} ( Phi((b-c-beta x)/s) - Phi((a-c-beta x)/s) ) ].
    """
    s = np.sqrt(x)
    if beta <= -_BETA0:                         # mirror: W = -W', W' has slope -beta on [-b, -a]
        return 1 - noisy_cdf(-c, -b, -a, -beta, x)
    if abs(beta) < _BETA0:                      # uniform on [a, b]
        def G(u):                               # int Phi(u) du = u Phi(u) + phi(u)
            return u * ndtr(u) + np.exp(-u * u / 2) / np.sqrt(2 * np.pi)
        return s * (G((c - a) / s) - G((c - b) / s)) / (b - a)
    # divide everything by e^{beta b} (beta > 0 wlog); normaliser e^{beta b}(1 - e^{-beta(b-a)})
    lz = np.log(-np.expm1(-beta * (b - a)))
    t1 = np.exp(log_ndtr((c - b) / s) - lz)
    t2 = np.exp(-beta * (b - a) + log_ndtr((c - a) / s) - lz)
    lo, hi = (a - c - beta * x) / s, (b - c - beta * x) / s
    # Phi(hi) - Phi(lo) in log space
    lphi = np.where(hi < 0, _log_diff_exp(log_ndtr(hi), log_ndtr(lo)),
                    _log_diff_exp(log_ndtr(-lo), log_ndtr(-hi)))
    t3 = np.exp(beta * (c - b) + beta * beta * x / 2 + lphi - lz)
    return t1 - t2 + t3


def noisy_mass(a, b, beta, x):
    """P(|W + e| <= 1)."""
    return noisy_cdf(1.0, a, b, beta, x) - noisy_cdf(-1.0, a, b, beta, x)


def latent_cdf(w, a, b, beta):
    if beta <= -_BETA0:                         # mirror, as in noisy_cdf (no atoms)
        return 1 - latent_cdf(-np.asarray(w, dtype=float), -b, -a, -beta)
    w = np.clip(w, a, b)
    if abs(beta) < _BETA0:
        return (w - a) / (b - a)
    return (np.exp(beta * (w - b)) - np.exp(-beta * (b - a))) / -np.expm1(-beta * (b - a))


def latent_abs_quantile(q, a, b, beta):
    """q-quantile of |W|."""
    mass = lambda s: latent_cdf(s, a, b, beta) - latent_cdf(-s, a, b, beta)
    hi = max(abs(a), abs(b))
    lo = 0.0
    if mass(lo) >= q:
        return 0.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mass(mid) < q else (lo, mid)
    return hi


def dirac_value(p, q, x):
    """Best Dirac: W = d >= 0 with P(|d + e| <= 1) >= p; value d (quantile of |W| is d)."""
    s = np.sqrt(x)
    f = lambda d: ndtr((1 - d) / s) - ndtr((-1 - d) / s) - p
    if f(0) < 0:
        return -np.inf
    return brentq(f, 0, 1 + 10 * s) if f(1 + 10 * s) < 0 else 1 + 10 * s


def solve_R(p, q, x, seeds=(0, 1, 2, 3), beta_max=400.0, span=3.0, polish=True):
    """Numerical maximiser of R_{p,q}(x) over Diracs and log-affine segments.

    Parametrisation: a in [-span, span], log length in [log 1e-4, log 2 span], beta in [0, beta_max].
    Returns (value, (a, b, beta)). This is a search: it gives a feasible point, i.e. a lower
    bound on R, not a certificate.
    """
    def unpack(z):
        a, ll, beta = z
        return a, a + np.exp(ll), beta

    def obj(z):
        a, b, beta = unpack(z)
        return -latent_abs_quantile(q, a, b, beta)

    def con(z):
        a, b, beta = unpack(z)
        return noisy_mass(a, b, beta, x) - p

    bounds = [(-span, span), (np.log(1e-4), np.log(2 * span)), (0.0, beta_max)]
    best, arg = dirac_value(p, q, x), None
    for sd in seeds:
        r = differential_evolution(obj, bounds, constraints=NonlinearConstraint(con, 0, np.inf),
                                   seed=sd, maxiter=600, popsize=30, tol=1e-12, polish=False)
        z = r.x
        if polish:
            m = minimize(obj, z, method='SLSQP', bounds=bounds,
                         constraints=[{'type': 'ineq', 'fun': con}],
                         options={'ftol': 1e-14, 'maxiter': 500})
            if m.success and con(m.x) >= -1e-12:
                z = m.x
        if con(z) >= -1e-12 and -obj(z) > best:
            best, arg = -obj(z), unpack(z)
    return best, arg


def _endpoint_values(p, q, x, beta, ell):
    """For W = c + Y, Y with density prop. to exp(beta y) on [0, ell]: the feasible shifts
    {c : P(|W + e| <= 1) >= p} form an interval (the map is log-concave in c, Prekopa), and
    Q_q(|W|) is quasi-convex in c (same argument), so its max over the interval is at an end.
    Returns that max, or -inf if no shift is feasible."""
    s = np.sqrt(x)
    h = lambda c: noisy_mass(c, c + ell, beta, x) - p
    lo, hi = -1 - ell - 12 * s, 1 + 12 * s
    cs = np.linspace(lo, hi, 64)
    hv = np.array([h(c) for c in cs])
    i = int(np.argmax(hv))
    a_, b_ = cs[max(i - 1, 0)], cs[min(i + 1, len(cs) - 1)]
    for _ in range(60):                                  # golden-section to the mode
        m1, m2 = a_ + 0.382 * (b_ - a_), a_ + 0.618 * (b_ - a_)
        a_, b_ = (m1, b_) if h(m1) < h(m2) else (a_, m2)
    cm = (a_ + b_) / 2
    if h(cm) < 0:
        return -np.inf
    best = -np.inf
    for end in (lo, hi):
        if h(end) >= 0:                                  # interval reaches the scan edge
            c = end
        else:
            c = brentq(h, *sorted((end, cm)), xtol=1e-14)
        best = max(best, latent_abs_quantile(q, c, c + ell, beta))
    return best


def R_grid(p, q, x, betas, ells):
    """max over a (beta, ell) grid of the exact endpoint values, plus the best Dirac.
    A lower bound on R_{p,q}(x) that converges to it as the grid is refined."""
    best = dirac_value(p, q, x)
    arg = None
    for beta in betas:
        for ell in ells:
            v = _endpoint_values(p, q, x, beta, ell)
            if v > best:
                best, arg = v, (beta, ell)
    return best, arg


# ---------------------------------------------------------------------------------------------
# Heterogeneous noise. Calibration area i has noise N(0, D_i); with a fixed threshold the mean
# indicator 1{|V_i| <= t} has expectation int gbar dG, gbar = average of the Gaussian kernels.
# That is still one linear constraint on G, so the extremal family is unchanged; only the
# kernel changes (an average of closed forms). gbar is not log-concave in general, so the
# feasible shifts may be a union of intervals: every component's endpoints are checked.
# ---------------------------------------------------------------------------------------------

def noisy_mass_mix(a, b, beta, xs, w=None):
    """Average over noise variances xs (weights w) of P(|W + e_x| <= 1)."""
    xs = np.atleast_1d(np.asarray(xs, dtype=float))
    w = np.full(len(xs), 1 / len(xs)) if w is None else np.asarray(w, dtype=float) / np.sum(w)
    return float(np.dot(w, noisy_mass(a, b, beta, xs)))


def _endpoint_values_mix(p, q, xs, w, beta, ell, n_scan=400):
    smax = np.sqrt(np.max(xs))
    h = lambda c: noisy_mass_mix(c, c + ell, beta, xs, w) - p
    lo, hi = -1 - ell - 12 * smax, 1 + 12 * smax
    cs = np.linspace(lo, hi, n_scan)
    hv = np.array([h(c) for c in cs])
    if hv.max() < 0:
        # refine around the best scan point before declaring infeasible
        i = int(np.argmax(hv)); a_, b_ = cs[max(i - 1, 0)], cs[min(i + 1, n_scan - 1)]
        for _ in range(60):
            m1, m2 = a_ + 0.382 * (b_ - a_), a_ + 0.618 * (b_ - a_)
            a_, b_ = (m1, b_) if h(m1) < h(m2) else (a_, m2)
        cm = (a_ + b_) / 2
        if h(cm) < 0:
            return -np.inf
        ends = [brentq(h, a_ - (b_ - a_) - 1e-9, cm, xtol=1e-14) if h(a_ - 1e-9) < 0 else cm,
                brentq(h, cm, b_ + 1e-9, xtol=1e-14) if h(b_ + 1e-9) < 0 else cm]
    else:
        ends = []
        pos = hv >= 0
        for i in range(n_scan - 1):
            if pos[i] != pos[i + 1]:
                ends.append(brentq(h, cs[i], cs[i + 1], xtol=1e-14))
        if pos[0]:
            ends.append(lo)
        if pos[-1]:
            ends.append(hi)
    return max(latent_abs_quantile(q, c, c + ell, beta) for c in ends)


def _mass_grid(C, ell, beta, xs, w):
    """Average noisy mass for shifts C (n, m) and lengths ell (n, 1); xs, w length-K."""
    out = np.zeros_like(C)
    for xk, wk in zip(xs, w):
        out += wk * noisy_mass(C, C + ell, beta, xk)
    return out


def shape_values(p, q, beta, ells, xs, w=None, m=256, iters=55):
    """Vectorised endpoint values for one slope beta and many lengths.

    For each length the feasible shifts S = {c : mass >= p} are found on a scan of m points and
    their outermost ends refined by bisection. Q_q(|W + c|) is quasi-convex in c on the whole
    line, so its max over S (a union of intervals when the kernel is a Gaussian mixture) is at
    min S or max S. Rows with no feasible scan point are refined around the best scan point by
    golden-section search before being declared infeasible (value -inf)."""
    xs = np.atleast_1d(np.asarray(xs, dtype=float))
    w = np.full(len(xs), 1 / len(xs)) if w is None else np.asarray(w, dtype=float) / np.sum(w)
    ells = np.asarray(ells, dtype=float)[:, None]
    smax = np.sqrt(xs.max())
    lo, hi = -1 - ells - 12 * smax, np.full_like(ells, 1 + 12 * smax)
    C = lo + (hi - lo) * np.linspace(0, 1, m)[None, :]
    H = _mass_grid(C, ells, beta, xs, w) - p
    n = len(ells)
    left, right = np.full(n, np.nan), np.full(n, np.nan)
    pos = H >= 0
    has = pos.any(1)
    # rows with no feasible scan point: golden search around the best scan point
    for i in np.where(~has)[0]:
        j = int(np.argmax(H[i])); a_, b_ = C[i, max(j - 1, 0)], C[i, min(j + 1, m - 1)]
        h = lambda c: _mass_grid(np.array([[c]]), ells[i:i + 1], beta, xs, w)[0, 0] - p
        for _ in range(60):
            m1, m2 = a_ + 0.382 * (b_ - a_), a_ + 0.618 * (b_ - a_)
            a_, b_ = (m1, b_) if h(m1) < h(m2) else (a_, m2)
        cm = (a_ + b_) / 2
        if h(cm) >= 0:
            l_, r_ = brentq(h, C[i, max(j - 1, 0)], cm, xtol=1e-14) if h(C[i, max(j - 1, 0)]) < 0 else cm, \
                     brentq(h, cm, C[i, min(j + 1, m - 1)], xtol=1e-14) if h(C[i, min(j + 1, m - 1)]) < 0 else cm
            left[i], right[i] = l_, r_
    rows = np.where(has)[0]
    if len(rows):
        P = pos[rows]
        j1 = P.argmax(1); j2 = m - 1 - P[:, ::-1].argmax(1)
        Cr, er = C[rows], ells[rows]
        # left end: between j1-1 (infeasible) and j1 (feasible)
        a_ = np.where(j1 > 0, Cr[np.arange(len(rows)), np.maximum(j1 - 1, 0)], Cr[:, 0])
        b_ = Cr[np.arange(len(rows)), j1]
        exact = j1 == 0
        for _ in range(iters):
            mid = (a_ + b_) / 2
            ok = _mass_grid(mid[:, None], er, beta, xs, w)[:, 0] >= p
            a_, b_ = np.where(ok, a_, mid), np.where(ok, mid, b_)
        left[rows] = np.where(exact, Cr[:, 0], b_)
        a_ = Cr[np.arange(len(rows)), j2]
        b_ = np.where(j2 < m - 1, Cr[np.arange(len(rows)), np.minimum(j2 + 1, m - 1)], Cr[:, -1])
        exact = j2 == m - 1
        for _ in range(iters):
            mid = (a_ + b_) / 2
            ok = _mass_grid(mid[:, None], er, beta, xs, w)[:, 0] >= p
            a_, b_ = np.where(ok, mid, a_), np.where(ok, b_, mid)
        right[rows] = np.where(exact, Cr[:, -1], a_)
    vals = np.full(n, -np.inf)
    e = ells[:, 0]
    for ends in (left, right):
        ok = ~np.isnan(ends)
        if ok.any():
            a, b = ends[ok], ends[ok] + e[ok]
            lo_, hi_ = np.zeros(ok.sum()), np.maximum(np.abs(a), np.abs(b))
            for _ in range(80):
                mid = (lo_ + hi_) / 2
                mass = latent_cdf(mid, a, b, beta) - latent_cdf(-mid, a, b, beta)
                lo_, hi_ = np.where(mass < q, mid, lo_), np.where(mass < q, hi_, mid)
            vals[ok] = np.maximum(vals[ok], hi_)
    return vals


# ---------------------------------------------------------------------------------------------
# Scaled one-sided problem c_{p,q} = sup{ F_Y^{-1}(q) : Y log-concave, P(Y + Z <= 0) >= p }.
# Supplementary Lemma S1: the sup is over exponential tails Y = b - E, E ~ Exp(u). Theorem 3: for every x,
# R_{q,q}(x) <= 1 + c_q sqrt(x); the p >= q bound of Section 4.3: R_{p,q}(x) <= 1 + C_{p,q} sqrt(x).
# ---------------------------------------------------------------------------------------------

def exp_tail_mass(b, u):
    """P(b - E + Z <= 0) for E ~ Exp(u) independent of Z ~ N(0, 1)."""
    return ndtr(-b) + np.exp(-u * b + u * u / 2 + log_ndtr(b - u))


def exp_tail_root(p, u):
    """b_u(p): the location at which the exponential tail of rate u has constraint value p."""
    return brentq(lambda b: exp_tail_mass(b, u) - p, -60, 60 / u + 60, xtol=1e-14)


def exp_tail_value(p, q, u):
    """q-quantile of the exponential tail of rate u that meets the constraint at level p."""
    return exp_tail_root(p, u) - np.log(1 / q) / u


def one_sided_constant(p, q, log_u=np.linspace(np.log(1e-3), np.log(200), 160)):
    """c_{p,q} by Supplementary Lemma S1: a scan over the rate u, polished around the best grid point.
    The u -> infinity end is the point mass, value -Phi^{-1}(p)."""
    if p >= 1:
        return -np.inf
    vals = np.array([exp_tail_value(p, q, np.exp(l)) for l in log_u])
    i = int(np.argmax(vals))
    best = max(vals[i], -np.sqrt(2) * erfinv(2 * p - 1))
    if 0 < i < len(log_u) - 1:
        r = minimize_scalar(lambda l: -exp_tail_value(p, q, np.exp(l)),
                            bounds=(log_u[i - 1], log_u[i + 1]), method='bounded',
                            options={'xatol': 1e-10})
        best = max(best, -r.fun)
    return best


def split_constant(p, q, n=21):
    """C_{p,q} of the p >= q bound of Section 4.3 (Supplementary equation (S2)): worst split of the noisy failure 1 - p between the two ends,
    best split of the latent failure 1 - q. Symmetric in the two ends."""
    worst = -np.inf
    for a_r in np.linspace(0, (1 - p) / 2, n):
        a_l = 1 - p - a_r
        f = lambda b_r: max(one_sided_constant(1 - a_l, 1 - (1 - q - b_r)),
                            one_sided_constant(1 - a_r, 1 - b_r))
        r = minimize_scalar(f, bounds=(a_r + 1e-12, a_r + (p - q) - 1e-12), method='bounded',
                            options={'xatol': 1e-9})
        worst = max(worst, r.fun)
    return worst


# ---------------------------------------------------------------------------------------------
# Centring (Theorems 4 and 5 and Supplementary Lemma S2 of the paper). If |EW| <= 1 - kappa sqrt(x) and q > 1 - 1/e,
#   lim_{x -> 0} {R_q^(kappa)(x) - 1} / sqrt(x) = L_q(kappa) = sup{ v(u) : k(u) >= kappa },
#   v(u) = b_u - log(1/q)/u,  k(u) = 1/u - b_u,  b_u = b_u(q).
# The exponential tail b_u - E, E ~ Exp(u), has mean -k(u); L_q(0) = c_q.
# ---------------------------------------------------------------------------------------------

def _tail_vk(q, u):
    b = exp_tail_root(q, u)
    return b - np.log(1 / q) / u, 1 / u - b


def tail_optimum(q):
    """(u*, c_q, kappa*): the maximiser of v, its value, and the mean distance k(u*)."""
    r = minimize_scalar(lambda l: -_tail_vk(q, np.exp(l))[0], bounds=(np.log(1e-3), np.log(5)),
                        method='bounded', options={'xatol': 1e-12})
    u = float(np.exp(r.x))
    v, k = _tail_vk(q, u)
    return u, v, k


def transition_constant(q, kappa, log_u=np.linspace(np.log(1e-6), np.log(50), 400)):
    """L_q(kappa) = sup{ v(u) : k(u) >= kappa }, by a grid in u refined at every crossing of
    k(u) = kappa. Equals c_q for kappa <= kappa*."""
    u_star, c_q, k_star = tail_optimum(q)
    if k_star >= kappa:
        return c_q
    us = np.exp(log_u)
    vk = np.array([_tail_vk(q, u) for u in us])
    ok = vk[:, 1] >= kappa
    best = vk[ok, 0].max() if ok.any() else -np.inf
    for i in np.flatnonzero(ok[:-1] != ok[1:]):
        u0 = brentq(lambda u: _tail_vk(q, u)[1] - kappa, us[i], us[i + 1], xtol=1e-15, rtol=1e-14)
        best = max(best, _tail_vk(q, u0)[0])
    return best


def transition_upper(q, kappa):
    """U_q(kappa) = min{c_q, sqrt(kappa^2 + 1) - kappa}: Theorem 3 and the mean-offset bound of Theorem 4."""
    return min(tail_optimum(q)[1], np.sqrt(kappa * kappa + 1) - kappa)


def centred_exp_law(beta, ell):
    """(a, b): support of X - EX, X with density prop. to exp(beta X) on [0, ell], beta != 0."""
    mean = ell / -np.expm1(-beta * ell) - 1 / beta
    return -mean, ell - mean


def centred_exp_coefficient(q, beta, ell):
    """Leading coefficient C of {Q_q(|W|)/t - 1}/x for the mean-zero law of centred_exp_law, t the
    noisy q-quantile of |W + sqrt(v) Z| and x = v/t^2 (fixed law, v -> 0). Needs +-r0 inside the
    support, where f' = beta f. C = (beta r0 / 2) tanh(beta r0)."""
    a, b = centred_exp_law(beta, ell)
    r0 = latent_abs_quantile(q, a, b, beta)
    if r0 >= min(-a, b):
        return np.nan
    return beta * r0 / 2 * np.tanh(beta * r0)


def noisy_abs_quantile(q, a, b, beta, x):
    """q-quantile of |W + sqrt(x) Z| for W with density prop. to exp(beta w) on [a, b]."""
    mass = lambda t: noisy_cdf(t, a, b, beta, x) - noisy_cdf(-t, a, b, beta, x)
    hi = max(abs(a), abs(b)) + 12 * np.sqrt(x)
    return brentq(lambda t: mass(t) - q, 0.0, hi, xtol=1e-15, rtol=1e-15)


# ---------------------------------------------------------------------------------------------
# Noise-dominated regime. As x increases to the feasibility
# edge x_p = 1/z^2, z = Phi^{-1}{(1 + p)/2}, R_{p,q}(x) ~ M_q (x_p - x)^{1/2}, where
#   M_q = sup{ Q_q(|W|) / E(W^2)^{1/2} : W log-concave }.
# Heterogeneous noise: the latent variance budget c* solves mean_i g_{x_i + c}(0) = p, and
# M_q c*^{1/2} approximates R^mix (an approximation, not a certified bound).
# ---------------------------------------------------------------------------------------------

def segment_second_moment(a, b, beta):
    """E(W^2) for W with density prop. to exp(beta w) on [a, b]."""
    if abs(beta) < _BETA0:
        return (b**3 - a**3) / (3 * (b - a))
    if beta < 0:
        return segment_second_moment(-b, -a, -beta)
    # divide by e^{beta b}: int w^2 e^{beta w} = e^{beta w}(w^2/beta - 2w/beta^2 + 2/beta^3)
    h = lambda w: w * w / beta - 2 * w / beta**2 + 2 / beta**3
    r = np.exp(-beta * (b - a))
    return (h(b) - r * h(a)) / ((1 - r) / beta)


def quantile_moment_ratio(q, beta, c):
    """Q_q(|W|) / E(W^2)^{1/2} for density prop. to exp(beta w) on [c, c + 1] (scale-free)."""
    a, b = c, c + 1.0
    return latent_abs_quantile(q, a, b, beta) / np.sqrt(segment_second_moment(a, b, beta))


def edge_constant(q, grid=(41, 41)):
    """M_q by localization (one linear constraint E W^2 <= 1): grid over slope and position of a
    unit segment, polished by Nelder-Mead. Point masses give ratio 1."""
    best, arg = 1.0, None
    for lb in np.linspace(-6, 4, grid[0]):
        for c in np.linspace(-1.5, 0.5, grid[1]):
            r = quantile_moment_ratio(q, np.exp(lb), c)
            if r > best:
                best, arg = r, (lb, c)
    res = minimize(lambda t: -quantile_moment_ratio(q, np.exp(t[0]), t[1]), arg,
                   method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-13, 'maxiter': 4000})
    return max(best, -res.fun)


def edge_level(p):
    """x_p: the largest x with pr(|x^{1/2} Z| <= 1) >= p."""
    return 1 / (np.sqrt(2) * erfinv(p)) ** 2


def variance_budget(p, xs, w=None):
    """c* with mean_i pr(|(x_i + c*)^{1/2} Z| <= 1) = p; 0 if even c = 0 is infeasible."""
    xs = np.atleast_1d(np.asarray(xs, float))
    w = np.full(len(xs), 1 / len(xs)) if w is None else np.asarray(w, float) / np.sum(w)
    f = lambda c: float(np.sum(w * (2 * ndtr(1 / np.sqrt(xs + c)) - 1))) - p
    if f(0.0) <= 0:
        return 0.0
    hi = 1.0
    while f(hi) > 0:
        hi *= 2
    return brentq(f, 0.0, hi, xtol=1e-15, rtol=1e-13)
