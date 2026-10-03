"""Certified upper bounds for R_{p,q}(x) by monotone branch and bound.

Family (Supplementary Proposition S3, beta >= 0 by the symmetry W -> -W): W = b - Y, where Y has density
proportional to exp(-beta y) on [0, ell]. The ends are included: ell = 0 or beta = inf is a
point mass at b, ell = inf is an exponential tail, beta = 0 is uniform.

Monotonicity. Y decreases in beta (likelihood ratio) and increases in ell (truncation of one
density), so W is stochastically increasing in beta and b and decreasing in ell. On a box
[beta0, beta1] x [ell0, ell1] x [b0, b1] put W_lo = W(beta0, ell1, b0), W_hi = W(beta1, ell0, b1).
Every W in the box satisfies W_lo <=st W <=st W_hi. The maps W -> P(W + e <= 1) and
W -> P(W + e <= -1) are decreasing in that order, hence for all W in the box

  P(|W + e| <= 1) <= A(W_lo) - B(W_hi),     A(W) = P(W + e <= 1), B(W) = P(W + e <= -1),
  P(|W| <= s)     >= P(W_hi <= s) - P(W_lo <= -s).

A third bound uses the density: P(|W + e| <= 1) <= 2 max f_W = 2 beta1 / (1 - exp(-beta1 ell0)).
A box is cleared when its noisy mass is below p (infeasible) or its latent mass is at least q
(value <= s). If every box of a cover of the parameter space is cleared, R_{p,q}(x) <= s.
With heterogeneous noise the constraint is the weighted average of the same kernels, and
the same bounds hold term by term.

Parameter space. b < b_lo is infeasible because W <= b. b > b_hi is infeasible because the
density of W increases towards b; see `b_range`. beta and ell are compactified as
beta = u / (1 - u), ell = v / (1 - v) with (u, v) in [0, 1]^2.

Floating point: the closed forms are evaluated in double precision in log space. They lose
accuracy by cancellation when ell < 1e-4 or 0 < beta ell < 1e-3, so box corners in those
zones are rounded outward (to a point mass, to ell = 1e-4, to beta = 0 or beta ell = 1e-3).
Outside them the error against 30-digit mpmath quadrature is below 1e-11 on random points with
1e-4 <= x <= 0.37, 1e-4 <= ell <= 20 or ell = inf, 1e-3 <= beta ell <= 1e3
(tests/test_core.py::test_closed_forms_match_high_precision_quadrature), and a box is
cleared only with a margin EPS = 1e-9.
"""
import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri

EPS = 1e-9
ELL_MIN, KAPPA_MIN = 1e-4, 1e-3


def _cdf_noisy(c, b, ell, beta, x):
    """P(W + e <= c) for arrays b, ell, beta (same shape); e ~ N(0, x)."""
    s = np.sqrt(x)
    b, ell, beta = np.broadcast_arrays(np.asarray(b, float), np.asarray(ell, float),
                                       np.asarray(beta, float))
    out = np.empty(b.shape)
    dirac = (ell == 0) | np.isinf(beta)
    degen = (beta == 0) & np.isinf(ell)                    # W = -inf: conservative limit
    unif = (beta == 0) & ~dirac & ~degen
    expo = ~dirac & ~degen & ~unif
    out[dirac] = ndtr((c - b[dirac]) / s)
    out[degen] = 1.0
    if unif.any():
        bb, ll = b[unif], ell[unif]
        G = lambda u: u * ndtr(u) + np.exp(-u * u / 2) / np.sqrt(2 * np.pi)
        out[unif] = s * (G((c - bb + ll) / s) - G((c - bb) / s)) / ll
    if expo.any():
        bb, ll, be = b[expo], ell[expo], beta[expo]
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            fin = np.isfinite(ll)
            lz = np.where(fin, np.log(-np.expm1(-be * np.where(fin, ll, 1.0))), 0.0)
            t1 = np.exp(log_ndtr((c - bb) / s) - lz)
            t2 = np.where(fin, np.exp(-be * np.where(fin, ll, 0) + log_ndtr((c - bb + np.where(fin, ll, 0)) / s) - lz), 0.0)
            hi = (bb - c - be * x) / s
            lo = np.where(fin, (bb - np.where(fin, ll, 0) - c - be * x) / s, -np.inf)
            # log(Phi(hi) - Phi(lo)), stable on both sides of zero
            l_left = log_ndtr(hi) + np.log1p(-np.exp(np.minimum(log_ndtr(lo) - log_ndtr(hi), 0)))
            l_right = log_ndtr(-lo) + np.log1p(-np.exp(np.minimum(log_ndtr(-hi) - log_ndtr(-lo), 0)))
            lphi = np.where(hi < 0, l_left, l_right)
            t3 = np.exp(be * (c - bb) + be * be * x / 2 + lphi - lz)
        out[expo] = t1 - t2 + t3
    return out


def _cdf_latent(y, b, ell, beta):
    """P(W <= y). Point masses use the closed inequality (conservative for both uses)."""
    b, ell, beta = np.broadcast_arrays(np.asarray(b, float), np.asarray(ell, float),
                                       np.asarray(beta, float))
    out = np.empty(b.shape)
    dirac = (ell == 0) | np.isinf(beta)
    degen = (beta == 0) & np.isinf(ell)
    unif = (beta == 0) & ~dirac & ~degen
    expo = ~dirac & ~degen & ~unif
    out[dirac] = (b[dirac] <= y).astype(float)
    out[degen] = 1.0
    d = b - y                                              # distance below the right end
    if unif.any():
        out[unif] = np.clip(1 - d[unif] / ell[unif], 0, 1)
    if expo.any():
        dd, ll, be = np.maximum(d[expo], 0), ell[expo], beta[expo]
        with np.errstate(over='ignore', invalid='ignore'):
            fin = np.isfinite(ll)
            num = np.exp(-be * dd) - np.where(fin, np.exp(-be * np.where(fin, ll, 0)), 0)
            den = np.where(fin, -np.expm1(-be * np.where(fin, ll, 1)), 1.0)
            v = np.where(dd >= ll, 0.0, num / den)
        out[expo] = np.clip(v, 0, 1)
    return out


def _fmax(beta, ell):
    """Maximal density of W (at b); inf for point masses."""
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        f = np.where(beta == 0, 1 / ell, beta / -np.expm1(-beta * ell))
    return np.where((ell == 0) | np.isinf(beta), np.inf, f)


def b_range(p, x):
    """b below b_lo or above b_hi is infeasible for every beta >= 0, ell (x a scalar = the
    smallest noise variance for mixtures, which makes both bounds conservative only for b_lo;
    for mixtures use the largest variance for b_hi, see `certify`).

    b_lo: W <= b, so P(|W+e| <= 1) <= P(b + e >= -1) = Phi((1 + b)/sqrt x).
    b_hi: with I = [-1-k, 1+k] and f_W increasing on the support, P(W in I) <= r/(1+r),
    r = (2+2k)/(b-1-k), and P(|W+e| <= 1) <= P(W in I) + P(|e| > k). k = 7 sqrt x."""
    s = np.sqrt(x)
    b_lo = -1 + s * ndtri(p) - 1e-12
    k = 7 * s
    pp = p - 2 * ndtr(-7.0) - EPS
    b_hi = 1 + k + (2 + 2 * k) * (1 - pp) / pp + 1e-9
    return b_lo, b_hi


def _to_beta(u):
    with np.errstate(divide='ignore'):
        return np.where(u >= 1, np.inf, u / (1 - u))


_to_ell = _to_beta


def box_status(p, q, s, xs, ws, boxes):
    """boxes: (n, 6) array of [u0, u1, v0, v1, b0, b1]. Returns cleared (bool) and a score
    (how far from being cleared; larger is worse)."""
    u0, u1, v0, v1, b0, b1 = boxes.T
    be0, be1, l0, l1 = _to_beta(u0), _to_beta(u1), _to_ell(v0), _to_ell(v1)
    # Outward rounding away from the cancellation zones of the closed forms (tiny ell, tiny
    # beta*ell). Each move keeps W_lo lower and W_hi higher in stochastic order.
    l1 = np.where((l1 > 0) & (l1 < ELL_MIN), ELL_MIN, l1)
    l0 = np.where(l0 < ELL_MIN, 0.0, l0)
    with np.errstate(invalid="ignore"):
        be0 = np.where(be0 * l1 < KAPPA_MIN, 0.0, be0)
    with np.errstate(divide='ignore', invalid='ignore'):
        be1 = np.where((l0 > 0) & (be1 * l0 < KAPPA_MIN), KAPPA_MIN / np.where(l0 > 0, l0, 1), be1)
    N = np.zeros(len(boxes))
    for x, w in zip(xs, ws):
        N += w * (_cdf_noisy(1.0, b0, l1, be0, x) - _cdf_noisy(-1.0, b1, l0, be1, x))
    N = np.minimum(N, 2 * _fmax(be1, l0))
    L = _cdf_latent(s, b1, l0, be1) - _cdf_latent(-s, b0, l1, be0)
    cleared = (N < p - EPS) | (L >= q + EPS)
    score = np.minimum(N - p, q - L)
    return cleared, score


def certify(p, q, s, xs, ws=None, max_boxes=4_000_000, init=(8, 8, 32), min_width=1e-7):
    """True if R_{p,q} <= s is certified for the noise mixture (xs, ws); False if the branch
    and bound hit its limits (then R may still be <= s). Returns (ok, n_evaluated)."""
    xs = np.atleast_1d(np.asarray(xs, float))
    ws = np.full(len(xs), 1 / len(xs)) if ws is None else np.asarray(ws, float) / np.sum(ws)
    b_lo = min(b_range(p, x)[0] for x in xs)   # conservative: union of feasible ranges
    b_hi = max(b_range(p - 0, x)[1] for x in xs)
    nu, nv, nb = init
    U, V, B = np.meshgrid(np.linspace(0, 1, nu + 1)[:-1], np.linspace(0, 1, nv + 1)[:-1],
                          np.linspace(b_lo, b_hi, nb + 1)[:-1], indexing='ij')
    du, dv, db = 1 / nu, 1 / nv, (b_hi - b_lo) / nb
    boxes = np.stack([U.ravel(), U.ravel() + du, V.ravel(), V.ravel() + dv,
                      B.ravel(), B.ravel() + db], axis=1)
    bscale = b_hi - b_lo
    n_eval = 0
    while len(boxes):
        n_eval += len(boxes)
        if n_eval > max_boxes:
            return False, n_eval
        cleared, _ = box_status(p, q, s, xs, ws, boxes)
        boxes = boxes[~cleared]
        if not len(boxes):
            break
        wu = boxes[:, 1] - boxes[:, 0]
        wv = boxes[:, 3] - boxes[:, 2]
        wb = (boxes[:, 5] - boxes[:, 4]) / bscale
        if np.max(np.maximum(np.maximum(wu, wv), wb)) < min_width:
            return False, n_eval
        dim = np.argmax(np.stack([wu, wv, wb]), axis=0)          # split the widest side
        lo, hi = boxes.copy(), boxes.copy()
        for d, (i0, i1) in enumerate(((0, 1), (2, 3), (4, 5))):
            m = dim == d
            mid = (boxes[m, i0] + boxes[m, i1]) / 2
            lo[m, i1] = mid
            hi[m, i0] = mid
        boxes = np.concatenate([lo, hi])
    return True, n_eval


def certified_upper(p, q, x, lower, rel_tols=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2), ws=None, **kw):
    """Smallest s = lower (1 + tol) over the tolerances that `certify` accepts.
    Returns (s, tol, n_evaluated) or (nan, nan, n) if none is certified."""
    total = 0
    for tol in rel_tols:
        s = lower * (1 + tol) if lower > 0 else tol
        ok, n = certify(p, q, s, x, ws, **kw)
        total += n
        if ok:
            return s, tol, total
    return np.nan, np.nan, total
