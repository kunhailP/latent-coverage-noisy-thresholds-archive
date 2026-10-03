"""Rigorous enclosures of the one-sided constants c_{p,q} and of the split constant C_{p,q}.

All inequalities that decide a bound are checked in ball arithmetic (Arb via python-flint);
floating point is used only to propose candidate points. A ball is turned into a double only
with outward rounding (`fup`, `fdown`), sums of bounds are formed as balls, and the u-grid ends
are exactly u_min and u_max. The results therefore do not rest on
the double-precision closed forms used elsewhere.

One-sided problem (Supplementary Lemma S1). For the exponential tail Y = b - E, E ~ Exp(u), Z ~ N(0, 1),

    kappa_u(b) = P(Y + Z <= 0) = Phi(-b) + exp(-u b + u^2/2) Phi(b - u),

and c_{p,q} = sup_{u > 0} v_u, v_u = b_u(p) - log(1/q)/u with kappa_u(b_u(p)) = p.

Monotonicity. kappa_u(b) decreases in b, and in u because E = E_1/u decreases in u. Hence
b_u(p) is nonincreasing in u, and on [u0, u1]

    v_u <= b_u0(p) - log(1/q)/u1.                                        (box bound)

A ball evaluation kappa_u(b) < p proves b_u(p) < b (upper candidate); kappa_u(b) > p proves
b_u(p) > b (lower candidate).

Ends. For u >= u_max, v_u <= b_{u_max}(p). For small u take eps = u^3 and
b(u) = (L_p + u^2/2 - log(1 - eps))/u, L_p = log(1/p). Then exp(-u b + u^2/2) = p (1 - eps) and
b >= L_p/u, so Phi(-b) <= phi(L_p/u) u / L_p by the Mills ratio. If phi(L_p/u)/u^2 <= p L_p the
two terms of kappa_u(b) sum to at most p, so b_u(p) <= b(u). The function phi(L/u)/u^2
increases on (0, L/sqrt(2)), so checking the inequality at u_min <= L_p/sqrt(2) covers
(0, u_min]. For p >= q the resulting bound on v_u is increasing in u, so its value at u_min
bounds the whole end.

Split constant (the p >= q bound of Section 4.3; Supplementary equation (S2)). c_{p',q'} <= g iff q' <= exp(-u (b_u(p') - g)) for every u > 0,
iff 1 - q' >= phi_g(1 - p') where

    phi_g(a) = max(0, sup_u 1 - exp(-G_u)),   G_u = u (b_u(1 - a) - g).

Hence C_{p,q} <= g iff  Psi_g = sup_{a_L + a_R = 1 - p} phi_g(a_L) + phi_g(a_R) <= 1 - q
(choose beta_L = phi_g(a_L), beta_R = 1 - q - beta_L; a strict inequality leaves room for a
margin). phi_g(a) is nondecreasing in a because b_u(1 - a) is. G_u is bounded on [u0, u1] by
u1 (b_u0 - g) when b_u0 - g >= 0 and by 0 otherwise, and the small-u end by
L + u^2/2 - log(1 - u^3) + u max(-g, 0) at u_min, as above.
"""
import heapq
import math

import numpy as np
from flint import arb, ctx
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr

ctx.prec = 128
SQRT2 = arb(2).sqrt()


def A(v):
    """Exact ball for a float or a decimal string."""
    return arb(v) if isinstance(v, str) else arb(float(v))


def fup(x):
    """Smallest double found with double >= every point of the ball x (outward rounding)."""
    f = float(x.upper())
    while not arb(f) >= x:
        f = math.nextafter(f, math.inf)
    return f


def fdown(x):
    """Double <= every point of the ball x."""
    f = float(x.lower())
    while not arb(f) <= x:
        f = math.nextafter(f, -math.inf)
    return f


def log_grid(u_min, u_max, n):
    """Geometric grid of doubles whose ends are exactly u_min and u_max, so the boxes
    [grid[i], grid[i+1]] cover [u_min, u_max] without gaps."""
    g = np.exp(np.linspace(math.log(u_min), math.log(u_max), n + 1))
    g[0], g[-1] = u_min, u_max
    return np.maximum.accumulate(g)


def Phi(x):
    return (-x / SQRT2).erfc() / 2


def phi(x):
    return (-x * x / 2).exp() / (2 * arb.pi()).sqrt()


def kappa(u, b):
    """Ball for kappa_u(b); u, b are floats (exact binary numbers)."""
    u, b = A(u), A(b)
    return Phi(-b) + (-u * b + u * u / 2).exp() * Phi(b - u)


def _kappa_float(b, u):
    return ndtr(-b) + np.exp(-u * b + u * u / 2 + log_ndtr(b - u))


def _root(u, p):
    return brentq(lambda b: _kappa_float(b, u) - p, -40.0, 40.0 / u + 40.0, xtol=1e-15, rtol=1e-15,
                  maxiter=400)


def b_bounds(u, p):
    """Floats (lo, hi) with lo < b_u(p) < hi proved in ball arithmetic. p is a ball."""
    b = _root(u, float(p.mid()))
    d = 1e-13 * max(1.0, abs(b))
    hi = lo = None
    for _ in range(60):
        if hi is None and kappa(u, b + d) < p:
            hi = b + d
        if lo is None and kappa(u, b - d) > p:
            lo = b - d
        if hi is not None and lo is not None:
            return lo, hi
        d *= 4
    raise RuntimeError(f'no enclosure of b_u(p) at u = {u}')


def small_u_end(p, candidates=None):
    """Largest u_min from the candidates for which the small-u argument applies at level p."""
    L = -p.log()
    Lf = float(L.mid())
    if candidates is None:
        candidates = Lf * np.array([0.70, 0.5, 0.35, 0.25, 0.18, 0.14, 0.11, 0.09, 0.075, 0.06, 0.05])
    for u in candidates:
        u = float(u)
        ua = A(u)
        if ua < L / SQRT2 and phi(L / ua) / (ua * ua) < p * L:
            return u
    raise RuntimeError('small-u end not established')


def _eps_term(u):
    ua = A(u)
    return ua * ua / 2 - (1 - ua ** 3).log()


def c_enclosure(p, q, u_max=64.0, n0=400, tol=1e-9, max_iter=200000):
    """Rigorous (lo, hi) with lo <= c_{p,q} <= hi, for p >= q (decimal strings or floats)."""
    p, q = A(p), A(q)
    if p < q:
        raise ValueError('needs p >= q')
    Lq = -q.log()
    # lower bound from the best of the proposal points
    u_min = small_u_end(p)
    grid = log_grid(u_min, u_max, n0)
    bs = [b_bounds(float(u), p) for u in grid]
    lo_f = max(fdown(A(bl) - Lq / A(u)) for u, (bl, _) in zip(grid, bs))
    # ends
    Lp = -p.log()
    ua = A(u_min)
    end_small = fup((Lp - Lq + _eps_term(u_min)) / ua)
    end_large = bs[-1][1]
    # box bounds on [u_i, u_{i+1}]
    def box(u0, bh0, u1):
        return fup(A(bh0) - Lq / A(u1))
    heap = [(-box(grid[i], bs[i][1], grid[i + 1]), float(grid[i]), float(grid[i + 1]), bs[i][1])
            for i in range(n0)]
    heapq.heapify(heap)
    it = 0
    while heap and -heap[0][0] > lo_f + tol and it < max_iter:
        _, u0, u1, bh0 = heapq.heappop(heap)
        um = math.sqrt(u0 * u1)
        bl, bh = b_bounds(um, p)
        vl = fdown(A(bl) - Lq / A(um))
        if vl > lo_f:
            lo_f = vl
        heapq.heappush(heap, (-box(u0, bh0, um), u0, um, bh0))
        heapq.heappush(heap, (-box(um, bh, u1), um, u1, bh))
        it += 1
    hi = max(-heap[0][0], end_small, end_large)
    return lo_f, hi, dict(u_min=u_min, boxes=len(heap), iters=it, end_small=end_small,
                          end_large=end_large)


def phi_upper(a, g, u_max=64.0, n0=200, tol=1e-7, max_iter=100000):
    """Rigorous upper bound on phi_g(a) (a, g floats or decimal strings)."""
    a, g = A(a), A(g)
    p1 = 1 - a
    L = -p1.log()
    u_min = small_u_end(p1)
    gneg = -g if g < 0 else arb(0)          # g is an exact double, so one branch is certain
    ua = A(u_min)
    G_end = fup(L + _eps_term(u_min) + ua * gneg)
    bh_max = b_bounds(u_max, p1)[1]
    if not (A(bh_max) - g < 0):
        raise RuntimeError('increase u_max')
    grid = log_grid(u_min, u_max, n0)
    bs = [b_bounds(float(u), p1) for u in grid]

    def box(u0, bh0, u1):
        d = A(bh0) - g
        if d < 0:
            return -math.inf
        return fup(A(u1) * d)

    def point(u, bl):
        return fdown(A(u) * (A(bl) - g))

    lo = max(point(u, bl) for u, (bl, _) in zip(grid, bs))
    heap = [(-box(grid[i], bs[i][1], grid[i + 1]), float(grid[i]), float(grid[i + 1]), bs[i][1])
            for i in range(n0)]
    heapq.heapify(heap)
    it = 0
    while heap and -heap[0][0] > lo + tol and it < max_iter:
        _, u0, u1, bh0 = heapq.heappop(heap)
        um = math.sqrt(u0 * u1)
        bl, bh = b_bounds(um, p1)
        lo = max(lo, point(um, bl))
        heapq.heappush(heap, (-box(u0, bh0, um), u0, um, bh0))
        heapq.heappush(heap, (-box(um, bh, u1), um, u1, bh))
        it += 1
    G = max(-heap[0][0], G_end)
    if G <= 0:
        return 0.0
    return fup(1 - (-A(G)).exp())


def split_certificate(p, q, g, n0=16, max_rounds=40, pool=None, tol=1e-7):
    """Checks C_{p,q} <= g. On an interval a_L in [a0, a1] of [0, T/2], T = 1 - p, the bound is
    phi(a1) + phi(T - a0) (phi nondecreasing; a_R = T - a_L). Intervals whose bound is not below
    1 - q are halved, up to max_rounds times. Returns (worst interval bound, budget, ok, info)."""
    pa, qa = A(p), A(q)
    T = 1 - pa
    budget = 1 - qa                       # compared as a ball
    cache = {0.0: 0.0}                    # phi at a = 0 is 0 (no feasible law)

    def need(points):
        new = sorted({x for x in points if x not in cache})
        args = [(x, g, 64.0, 200, tol) for x in new]
        vals = pool.starmap(phi_upper, args) if pool is not None else [phi_upper(*a) for a in args]
        cache.update(zip(new, vals))

    # nodes as exact dyadic fractions of T; phi (nondecreasing) is evaluated at ends rounded up
    def up(fr):
        return fup(T * fr) if fr > 0 else 0.0

    ivs = [(i / (2 * n0), (i + 1) / (2 * n0)) for i in range(n0)]     # fractions of T, a_L <= T/2
    worst = math.inf
    for rnd in range(max_rounds):
        need([up(b) for _, b in ivs] + [up(1 - a) for a, _ in ivs])
        bad, worst = [], -math.inf
        for a, b in ivs:
            v = A(cache[up(b)]) + A(cache[up(1 - a)])    # exact at 128 bits
            worst = max(worst, fup(v))
            if not v < budget:
                bad.append((a, b))
        if not bad:
            return worst, fdown(budget), True, dict(rounds=rnd, evals=len(cache))
        ivs = [iv for iv in ivs if iv not in bad] + \
              [h for a, b in bad for h in ((a, (a + b) / 2), ((a + b) / 2, b))]
    return worst, fdown(budget), False, dict(rounds=max_rounds, evals=len(cache), bad=bad[:5])


def bimodal_blc_certificate(p, m1, m2, s, n0=64, max_boxes=200000):
    """Ball-arithmetic proof that the mixture p N(m1, s^2) + (1 - p) N(m2, s^2), m1 < m2, is
    bi-log-concave (Supplementary Lemma S3).

    With P(w) the posterior weight of the first component, (log f)'' = -1/s^2 + P(1 - P) r^2/s^2,
    r = (m2 - m1)/s, and P decreases in w. So f is log-concave on (-inf, a] and [b, inf), where
    P(a) = 1/2 + (1/4 - 1/r^2)^{1/2} and P(b) = 1/2 - (1/4 - 1/r^2)^{1/2} (r > 2); by Prekopa's
    theorem log F is concave on (-inf, a] and log(1 - F) on [b, inf). Where f' <= 0,
    f'F <= f^2 holds trivially, and f' < 0 on [m2, inf); where f' >= 0, -f'(1 - F) <= f^2 holds
    trivially, and f' > 0 on (-inf, m1]. What remains is f'F < f^2 on [a, m2] and
    -f'(1 - F) < f^2 on [m1, b], checked here on boxes in ball arithmetic. The box ends a_chk <= a
    and b_chk >= b are rounded outward. Returns (ok, worst upper bound of the two ratios
    (f'F - f^2)/f^2 and (-f'(1 - F) - f^2)/f^2, number of boxes)."""
    P, M1, M2, Sg = A(p), A(m1), A(m2), A(s)
    r = (M2 - M1) / Sg
    if not r > 2:                                   # f log-concave: nothing to check
        return True, -math.inf, 0
    h = (arb(1) / 4 - 1 / (r * r)).sqrt()
    mid, lo = (M1 + M2) / 2, (P / (1 - P)).log()
    # log-odds of the first component: lo - (m2 - m1)(2w - m1 - m2)/(2 s^2), decreasing in w
    w_at = lambda Pv: mid + Sg * Sg * (lo - (Pv / (1 - Pv)).log()) / (M2 - M1)
    a_chk, b_chk = fdown(w_at(arb(1) / 2 + h)), fup(w_at(arb(1) / 2 - h))

    def parts(w):
        z1, z2 = (w - M1) / Sg, (w - M2) / Sg
        d1, d2 = P * phi(z1) / Sg, (1 - P) * phi(z2) / Sg
        f = d1 + d2
        fp = -(d1 * z1 + d2 * z2) / Sg
        F = P * Phi(z1) + (1 - P) * Phi(z2)
        S = P * Phi(-z1) + (1 - P) * Phi(-z2)
        return f, fp, F, S

    tests = [(a_chk, float(m2), lambda f, fp, F, S: (fp * F - f * f) / (f * f)),
             (float(m1), b_chk, lambda f, fp, F, S: (-fp * S - f * f) / (f * f))]
    worst, boxes = -math.inf, 0
    for x0, x1, ratio in tests:
        stack = [(x0 + (x1 - x0) * i / n0, x0 + (x1 - x0) * (i + 1) / n0) for i in range(n0)]
        stack[0], stack[-1] = (x0, stack[0][1]), (stack[-1][0], x1)
        while stack:
            u, v = stack.pop()
            boxes += 1
            if boxes > max_boxes:
                return False, worst, boxes
            val = ratio(*parts(A(u).union(A(v))))
            if val < 0:
                worst = max(worst, fup(val))
            else:
                c = (u + v) / 2
                stack += [(u, c), (c, v)]
    return True, worst, boxes
