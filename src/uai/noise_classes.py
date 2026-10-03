"""Coverage transfer for classes of noise laws (Section on general noise in the paper).

For a class E of noise laws closed under scaling, a latent residual W in B, and noise e ~ some law
in E independent of W, write Psi_E(q) = sup_{eps in E} E min(1, q exp(-eps)). By Proposition 1 (the one-sided argument of
Supplementary Lemma S1, whose proof does not use the noise law) and scaling, the one-sided constant c_{p,q}(eps) is <= 0 for
every eps in E and every scale if Psi_E(q) < p, and fails for some law and scale if Psi_E(q) > p.

Symmetric unimodal noise. By Khintchine's theorem eps = R U with U uniform on [-1, 1] independent
of R >= 0, so E min(1, q e^{-eps}) = E g(R) with g(r) = E min(1, q e^{-rU}) and
Psi_SU(q) = sup_r g(r). With c = log q < 0,
    g(r) = q sinh(r)/r                          for r <= -c,
    g(r) = (c + r + 1 - q e^{-r}) / (2 r)       for r >= -c.
The first branch increases. On the second, the sign of g' is that of
D(r) = q e^{-r}(1 + r) - (1 + c), which decreases in r; so if D(-c) > 0 the maximum is at the
root r* of D, where g(r*) = (1 + q e^{-r*})/2, and otherwise at r = -c.

All deciding inequalities below are evaluated in Arb ball arithmetic (uai.interval).
"""
import math

import numpy as np

from flint import arb

from uai.interval import A, fdown, fup


def _D(r, q):
    c = q.log()
    return q * (-r).exp() * (1 + r) - (1 + c)


def psi_su_enclosure(q, tol=1e-15):
    """Rigorous [lo, hi] for Psi_SU(q), q a float or decimal string in (1/e, 1)."""
    qa = A(q)
    c = qa.log()
    if not (1 + c > 0):
        raise ValueError('q must exceed 1/e')
    r0 = -c                                     # left end of the second branch
    if not (_D(r0, qa) > 0):
        v = qa * r0.sinh() / r0
        return fdown(v), fup(v)
    lo, hi = fup(r0), 1.0
    while not (_D(A(hi), qa) < 0):
        hi *= 2
    # bisection on doubles; signs decided in ball arithmetic
    while hi - lo > tol * max(1.0, hi):
        mid = (lo + hi) / 2
        d = _D(A(mid), qa)
        if d > 0:
            lo = mid
        elif d < 0:
            hi = mid
        else:
            break
    # r* in [lo, hi] (D(lo) >= 0 >= D(hi)); Psi = (1 + q e^{-r*})/2 is decreasing in r*
    return fdown((1 + qa * (-A(hi)).exp()) / 2), fup((1 + qa * (-A(lo)).exp()) / 2)


def psi_su_upper(q):
    return psi_su_enclosure(q)[1]


def latent_level(p, tol=1e-12):
    """Largest decimal-ish q' (as a double) with Psi_SU(q') <= p certified: the one-sided latent
    level guaranteed by noisy level p for every symmetric unimodal noise."""
    lo, hi = 0.5, float(p)
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if psi_su_upper(mid) <= p and A(psi_su_upper(mid)) <= A(p):
            lo = mid
        else:
            hi = mid
    assert A(psi_su_upper(lo)) <= A(p)
    return lo


def split_certificate_su(p, q, n0=64, max_rounds=30):
    """Certifies that every symmetric unimodal noise law, at every scale, satisfies the two-sided
    split bound: for every a_L + a_R = 1 - p there are latent budgets with
    phi(a_L) + phi(a_R) <= 1 - q, where phi(a) = 1 - latent_level(1 - a) bounds the smallest
    latent budget at noisy budget a. phi is nondecreasing, so on [a0, a1] of a_L in [0, T/2]
    the bound phi(a1) + phi(T - a0) suffices. Returns (ok, worst bound, budget)."""
    pa, qa = A(p), A(q)
    T = 1 - pa
    budget = 1 - qa
    cache = {}

    def phi(a):                                  # a a double; upper bound on the latent budget
        if a <= 0:
            return 0.0
        if a not in cache:
            cache[a] = fup(1 - A(latent_level(fdown(1 - A(a)))))
        return cache[a]

    ivs = [(i / (2 * n0), (i + 1) / (2 * n0)) for i in range(n0)]
    for _ in range(max_rounds):
        bad, worst = [], -math.inf
        for f0, f1 in ivs:
            a1 = fup(T * f1)
            a0 = fdown(T * f0) if f0 > 0 else 0.0
            v = A(phi(a1)) + A(phi(fup(T - A(a0))))
            worst = max(worst, fup(v))
            if not v <= budget:
                bad.append((f0, f1))
        if not bad:
            return True, worst, fdown(budget)
        ivs = [iv for iv in ivs if iv not in bad] + \
              [h for f0, f1 in bad for h in ((f0, (f0 + f1) / 2), ((f0 + f1) / 2, f1))]
    return False, worst, fdown(budget)


# ---------------------------------------------------------------------------------------------
# Mean-zero log-concave noise.
#
# By the localization theorem with the single constraint E(eps) >= 0 (h(e) = min(1, q e^{-e}) is
# decreasing and 1-Lipschitz, so truncation to compacts costs nothing in the limit), Psi_LC(q)
# is the supremum of E h over point masses at x >= 0 (value <= q) and over laws with density
# proportional to exp(beta e) on a segment, centred at their mean. Write such a law as
# eps = L (V - mu_k), V on [0, 1] with density proportional to e^{k v}, k = beta L. The
# certificate below covers
#   region I    |k| <= 2,  parameters (k, L);
#   region II   k <= -2,   eps = sigma (tau - m), tau ~ Exp(1) truncated to [0, |k|], sigma = L/|k|,
#                          parameters (1/|k|, sigma), including |k| = inf (centred exponential, long
#                          right tail);
#   region III  k >= 2,    eps = sigma (m - tau), the reflection of region II (long left tail);
# and the ends of the scale parameter by the analytic bounds in `lc_certificate`.
# ---------------------------------------------------------------------------------------------
from flint import arb as _arb, ctx as _ctx


def _hull(lo, hi):
    """Ball containing the reals lo..hi (floats or arbs, lo <= hi)."""
    l, h = A(lo) if not isinstance(lo, _arb) else lo, A(hi) if not isinstance(hi, _arb) else hi
    l_, h_ = fdown(l), fup(h)
    mid = A(l_) / 2 + A(h_) / 2
    rad = A(h_) / 2 - A(l_) / 2
    return mid + _arb(0, fup(rad))


_INF = _arb(0, float('inf'))


def _Z(x):
    """int_0^1 e^{x v} dv for a ball x."""
    if x.abs_upper() <= 0.5:
        s, term = _arb(0), _arb(1)
        N = 30
        for n in range(N + 1):
            s += term / (n + 1)
            term = term * x / (n + 1)
        # tail sum_{n > N} |x|^n/(n+1)! <= |x|^{N+1}/(N+2)! e^{|x|}
        tail = fup(A(x.abs_upper()) ** (N + 1) / _arb(N + 2).fac() * A(x.abs_upper()).exp())
        return s + _arb(0, tail)
    if x.contains(0):
        return _INF
    return (x.exp() - 1) / x


def _M(x):
    """int_0^1 v e^{x v} dv."""
    if x.abs_upper() <= 0.5:
        s, term = _arb(0), _arb(1)          # term = x^n / n!
        N = 30
        for n in range(N + 1):
            s += term / (n + 2)
            term = term * x / (n + 1)
        tail = fup(A(x.abs_upper()) ** (N + 1) / _arb(N + 1).fac() * A(x.abs_upper()).exp())
        return s + _arb(0, tail)
    if x.contains(0):
        return _INF
    return (x.exp() * (x - 1) + 1) / x ** 2


def _I(g, a, b):
    """int_a^b e^{g v} dv = (b - a) e^{g a} Z(g (b - a))."""
    return (b - a) * (g * a).exp() * _Z(g * (b - a))


def eh_region1(q, k0, k1, L0, L1):
    """Upper enclosure of E h over the box [k0,k1] x [L0,L1] of region I.

    For any cut v in [0, 1], min(1, q e^{-x}) <= 1{V <= v} + q e^{-x} 1{V > v}, so
    Phi(v) = {int_0^v e^{kv} + q e^{L mu} int_v^1 e^{(k-L)v}} / Z_k bounds E h, with equality at
    v = v_c. The cut is fixed at v_c of the box centre (a fixed cut), which removes the
    dependence of the cut on the parameters; Phi'(v_c) = 0, so the bound is tight to second
    order."""
    qa, c = A(q), A(q).log()
    k, L = _hull(k0, k1), _hull(L0, L1)
    Zk = _Z(k)
    mu = _M(k) / Zk
    km, Lm = (k0 + k1) / 2, (L0 + L1) / 2
    mum = _M(A(km)) / _Z(A(km))
    v = min(max(float((mum + c / A(Lm)).mid()), 0.0), 1.0)
    vv = A(v)
    num = _I(k, A(0), vv) + qa * (L * mu).exp() * _I(k - L, vv, A(1))
    return num / Zk


def _ek_bounds(t0, t1):
    """For k in [1/t1, 1/t0] (t0 = 0 means k up to inf): balls for e^{-k}, k e^{-k}, and the
    interval of k as floats."""
    k_lo = fdown(1 / A(t1))                     # outward: k_lo <= 1/t1, k_hi >= 1/t0
    k_hi = math.inf if t0 == 0 else fup(1 / A(t0))
    ek = _hull(0.0 if k_hi == math.inf else fdown((-A(k_hi)).exp()), fup((-A(k_lo)).exp()))
    kek = _hull(0.0 if k_hi == math.inf else fdown(A(k_hi) * (-A(k_hi)).exp()),
                fup(A(k_lo) * (-A(k_lo)).exp()))
    return ek, kek, k_lo, k_hi


def _exp_neg(rate_lo_ball_factor, k_lo, k_hi):
    """Ball for e^{-a k}, a > 0 a ball, k in [k_lo, k_hi]."""
    a = rate_lo_ball_factor
    hi = fup((-(A(fdown(a)) * A(k_lo))).exp())
    lo = 0.0 if k_hi == math.inf else fdown((-(A(fup(a)) * A(k_hi))).exp())
    return _hull(lo, hi)


def _m_mid(t0, t1):
    tm = (t0 + t1) / 2
    if tm == 0:
        return 1.0
    km = 1 / tm
    return float((1 - A(km) * (-A(km)).exp() / (1 - (-A(km)).exp())).mid())


def eh_region2(q, t0, t1, s0, s1):
    """k <= -2 (long right tail): eps = sigma (tau - m), t = 1/|k| in [t0, t1], sigma in [s0, s1].
    Upper bound Phi(u) = P(tau <= u) + q E[e^{-eps}; tau > u] at a fixed cut u (see region I)."""
    qa, c = A(q), A(q).log()
    sg = _hull(s0, s1)
    ek, kek, k_lo, k_hi = _ek_bounds(t0, t1)
    D = 1 - ek
    m = 1 - kek / D
    u = min(max(_m_mid(t0, t1) + float(c.mid()) / ((s0 + s1) / 2), 0.0), k_lo)
    uu = A(u)
    P = (1 - (-uu).exp()) / D
    e_k = _exp_neg(1 + sg, k_lo, k_hi)
    A2 = qa * (sg * m).exp() * ((-(1 + sg) * uu).exp() - e_k) / ((1 + sg) * D)
    return P + A2


def eh_region3(q, t0, t1, s0, s1):
    """k >= 2 (long left tail): eps = sigma (m - tau), k = 1/t in [1/t1, 1/t0]. Upper bound
    Phi(u) = P(tau >= u) + q E[e^{-eps}; tau < u] at a fixed cut u in [0, k_lo]."""
    qa, c = A(q), A(q).log()
    sg = _hull(s0, s1)
    ek, kek, k_lo, k_hi = _ek_bounds(t0, t1)
    D = 1 - ek
    m = 1 - kek / D
    u = min(max(_m_mid(t0, t1) - float(c.mid()) / ((s0 + s1) / 2), 0.0), k_lo)
    uu = A(u)
    P = ((-uu).exp() - ek) / D
    integ = _I(sg - 1, A(0), uu)
    return P + qa * (-sg * m).exp() * integ / D


def lc_point(q, sigma):
    """Enclosure of E h for the centred exponential noise sigma (E - 1), exact cut
    tau_c = 1 + c / sigma: a rigorous lower bound for Psi_LC(q) through its lower end."""
    qa, c = A(q), A(q).log()
    sg = A(sigma)
    tc = 1 + c / sg
    if not (tc > 0):
        return qa * sg.exp() / (1 + sg)
    return 1 - (-tc).exp() + qa * sg.exp() * (-(1 + sg) * tc).exp() / (1 + sg)


def _bb(f, box, target, max_boxes, min_width):
    """Branch and bound: True if f(box) < target on all sub-boxes."""
    import heapq
    stack = [box]
    n, worst = 0, -math.inf
    while stack:
        b = stack.pop()
        n += 1
        if n > max_boxes:
            return False, worst, n, b
        v = f(*b)
        if v < target:
            worst = max(worst, fup(v)) if v.is_finite() else worst
            continue
        a0, a1, c0, c1 = b
        # split the relatively wider side (log scale for the scale parameter)
        wa = (a1 - a0)
        wc = math.log(c1 / c0) if c0 > 0 else (c1 - c0)
        if max(wa, wc) < min_width:
            return False, fup(v) if v.is_finite() else math.inf, n, b
        if wa * 3 >= wc:
            am = (a0 + a1) / 2
            stack += [(a0, am, c0, c1), (am, a1, c0, c1)]
        else:
            cm = math.sqrt(c0 * c1) if c0 > 0 else (c0 + c1) / 2
            stack += [(a0, a1, c0, cm), (a0, a1, cm, c1)]
    return True, worst, n, None


def lc_certificate(q, p, max_boxes=2_000_000, min_width=1e-7, verbose=False):
    """Certify Psi_LC(q) < p (one-sided, every mean-zero log-concave noise law, every scale).
    Analytic ends (see the paper's supplement):
      region I,  L <= L_small: E h <= q e^{L^2/24} (variance of a log-affine law on [0, 1] at most 1/12;
                 sharper than Hoeffding's e^{L^2/8}); L >= L_big: E h <= P(V <= mu + z/L) + q e^{-z} with
                 P(V <= mu) <= 1 - 1/e (Grunbaum) and density of V <= 2/(1 - e^{-2});
      region II, sigma <= s_small: E h <= q e^{sigma m} <= q e^{sigma}; sigma >= s_big:
                 E h <= (1 - e^{-1 - z/sigma})/(1 - e^{-2}) + q e^{-z};
      region III, sigma <= s_small3: E h <= q e^{-sigma m}/(1 - sigma) with m >= m(2);
                 sigma >= s_big3: E h <= e^{-m(2) + z/sigma}/(1 - e^{-2}) + q e^{-z}.
    """
    qa, pa = A(q), A(p)
    z = 4.0
    out = {}
    # region I ends
    # E h <= q E e^{-eps} = q exp{int_{k-L}^k (mu(k) - mu(s)) ds} <= q e^{L^2/24}, since
    # mu' = var <= 1/12 for log-affine laws on [0, 1]
    L_small = fdown((24 * (pa / qa).log()).sqrt())
    assert qa * (A(L_small) ** 2 / 24).exp() < pa
    dmax = 2 / (1 - A(-2).exp())
    L_big = 1.0
    while not ((1 - (-A(1)).exp()) + A(z) * dmax / A(L_big) + qa * (-A(z)).exp() < pa):
        L_big *= 1.25
    # region II ends
    s_small = fdown((pa / qa).log())
    s_big = 1.0
    while not ((1 - (-1 - A(z) / A(s_big)).exp()) / (1 - A(-2).exp()) + qa * (-A(z)).exp() < pa):
        s_big *= 1.25
    # region III ends
    m2 = 1 - 2 * A(-2).exp() / (1 - A(-2).exp())
    s_small3 = 0.5
    while not (qa * (-A(s_small3) * m2).exp() / (1 - A(s_small3)) < pa):
        s_small3 *= 0.8
    s_big3 = 1.0
    while not ((-m2 + A(z) / A(s_big3)).exp() / (1 - A(-2).exp()) + qa * (-A(z)).exp() < pa):
        s_big3 *= 1.25
    out['ends'] = dict(L_small=L_small, L_big=L_big, s_small=s_small, s_big=s_big,
                       s_small3=s_small3, s_big3=s_big3)
    f1 = lambda a0, a1, c0, c1: eh_region1(q, a0, a1, c0, c1)
    f2 = lambda a0, a1, c0, c1: eh_region2(q, a0, a1, c0, c1)
    f3 = lambda a0, a1, c0, c1: eh_region3(q, a0, a1, c0, c1)
    ok = True
    for name, f, box in (('II', f2, (0.0, 0.5, s_small, s_big)),
                         ('III', f3, (0.0, 0.5, s_small3, s_big3)),
                         ('I', f1, (-2.0, 2.0, L_small, L_big))):
        r = _bb(f, box, pa, max_boxes, min_width)
        out[name] = dict(ok=r[0], worst=r[1], boxes=r[2], failed_box=r[3])
        if verbose:
            print(name, out[name], flush=True)
        ok &= r[0]
    out['ok'] = ok
    return out


def _bb_job(args):
    region, q, box, target, max_boxes, min_width = args
    f = {'I': eh_region1, 'II': eh_region2, 'III': eh_region3}[region]
    r = _bb(lambda a0, a1, c0, c1: f(q, a0, a1, c0, c1), box, A(target), max_boxes, min_width)
    return region, box, r[0], r[1], r[2], r[3]


def lc_certificate_parallel(q, p, procs=8, pieces=48, max_boxes=3_000_000, min_width=1e-9):
    """As lc_certificate, with each region split into `pieces` sub-boxes along the scale
    parameter (geometric) and run in parallel. Returns (ok, details)."""
    from multiprocessing import Pool
    qa, pa = A(q), A(p)
    z = 4.0
    L_small = fdown((24 * (pa / qa).log()).sqrt())          # E h <= q e^{L^2/24}
    dmax = 2 / (1 - A(-2).exp())
    L_big = 1.0
    while not ((1 - (-A(1)).exp()) + A(z) * dmax / A(L_big) + qa * (-A(z)).exp() < pa):
        L_big *= 1.25
    s_small = fdown((pa / qa).log())
    s_big = 1.0
    while not ((1 - (-1 - A(z) / A(s_big)).exp()) / (1 - A(-2).exp()) + qa * (-A(z)).exp() < pa):
        s_big *= 1.25
    m2 = 1 - 2 * A(-2).exp() / (1 - A(-2).exp())
    s_small3 = 0.5
    while not (qa * (-A(s_small3) * m2).exp() / (1 - A(s_small3)) < pa):
        s_small3 *= 0.8
    s_big3 = 1.0
    while not ((-m2 + A(z) / A(s_big3)).exp() / (1 - A(-2).exp()) + qa * (-A(z)).exp() < pa):
        s_big3 *= 1.25
    ends = dict(L_small=L_small, L_big=L_big, s_small=s_small, s_big=s_big, s_small3=s_small3,
                s_big3=s_big3)
    jobs = []
    for region, (a0, a1), (c0, c1) in (('II', (0.0, 0.5), (s_small, s_big)),
                                       ('III', (0.0, 0.5), (s_small3, s_big3)),
                                       ('I', (-2.0, 2.0), (L_small, L_big))):
        g = [c0 * (c1 / c0) ** (i / pieces) for i in range(pieces + 1)]
        g[0], g[-1] = c0, c1
        for i in range(pieces):
            jobs.append((region, q, (a0, a1, g[i], g[i + 1]), p, max_boxes, min_width))
    with Pool(procs) as pool:
        res = pool.map(_bb_job, jobs, chunksize=1)
    ok = all(r[2] for r in res)
    worst = max((r[3] for r in res if r[3] != -math.inf), default=-math.inf)
    bad = [r for r in res if not r[2]]
    return ok, dict(ends=ends, worst=worst, boxes=sum(r[4] for r in res), failed=bad[:3])


def psi_exp_float(q):
    """Double-precision sup over sigma of E h for the centred exponential noise (used only to
    propose levels; every deciding inequality is certified separately)."""
    from scipy.optimize import minimize_scalar
    c = math.log(q)

    def G(s):
        if s <= -c:
            return q * math.exp(s) / (1 + s)
        a = 1 + c / s
        return 1 - math.exp(-a) + q * math.exp(s - (1 + s) * a) / (1 + s)
    grid = max(G(s) for s in np.geomspace(1e-5, 50, 600))
    r = minimize_scalar(lambda ls: -G(math.exp(ls)), bounds=(math.log(1e-5), math.log(50)),
                        method='bounded', options={'xatol': 1e-13})
    return max(grid, -r.fun)


def lambda_exp_float(p):
    """Largest q' with psi_exp_float(q') <= p (double precision); q' = p for p within 1e-9 of 1."""
    from scipy.optimize import brentq
    if p >= 1 - 1e-9:
        return p
    return brentq(lambda qq: psi_exp_float(qq) - p, 0.5, p + 1e-15, xtol=1e-14)


def greedy_split_grid(p, q, keep=0.5, max_nodes=2000):
    """Partition of a_L in [0, T/2], T = 1 - p, from the centred exponential values: each
    interval [a0, a1] uses at most `keep` of the room phi(a1) + phi(T - a0) <= 1 - q, leaving
    the rest to the certificates. Double precision only proposes the grid."""
    from scipy.optimize import brentq
    T, B = 1 - float(p), 1 - float(q)
    ph = lambda a: 1 - lambda_exp_float(1 - a) if a > 0 else 0.0
    grid, a0 = [0.0], 0.0
    while True:
        room = B - ph(T - a0)
        if room <= ph(a0):
            raise ValueError('no room: the one-sided level is not met')
        level = ph(a0) + keep * (room - ph(a0))
        if ph(T / 2) <= level:
            grid.append(T / 2)
            return grid
        a1 = brentq(lambda a: ph(a) - level, max(a0, 1e-15), T / 2, xtol=1e-15)
        grid.append(a1)
        a0 = a1
        if len(grid) > max_nodes:
            raise ValueError('grid too fine')


def split_certificate_lc(p, q, keep=0.5, share=0.45, procs=60, pieces=24, verbose=False):
    """Two-sided certificate for every mean-zero log-concave noise law at every scale: noisy
    level p transfers to latent level q. On a greedy partition of a_L in [0, T/2]
    (greedy_split_grid), the latent budgets over [a0, a1] are bounded by phi(a1) + phi(T - a0),
    phi nondecreasing. Each node a gets phi(a) <= 1 - q_a, q_a = lambda_exp(1 - a) - delta_a, with
    Psi_LC(q_a) < 1 - a certified by branch and bound; delta_a is `share` of the smallest slack
    of the intervals using a. The sums are checked in ball arithmetic. phi(0) = 0 in the limit."""
    # T bounds the noisy budget 1 - p from above (outward), and the grid is closed at T/2, so
    # the intervals cover every split a_L + a_R <= 1 - p; partners T - a0 are rounded up.
    T, B = fup(1 - A(p)), 1 - float(q)
    grid = greedy_split_grid(p, q, keep=keep)
    grid[-1] = max(grid[-1], T / 2)
    partner = lambda a0: fup(A(T) - A(a0)) if a0 > 0 else T
    ph = {}
    for a in grid[1:] + [partner(a0) for a0 in grid[:-1]]:
        ph[a] = 1 - lambda_exp_float(1 - a) if a > 0 else 0.0
    delta = {a: math.inf for a in ph if a > 0}
    for a0, a1 in zip(grid[:-1], grid[1:]):
        slack = B - ph[a1] - ph[partner(a0)]
        for a in (a1, partner(a0)):
            if a > 0:
                delta[a] = min(delta[a], share * slack)
    phi, certs = {0.0: 0.0}, {}
    for i, a in enumerate(sorted(delta)):
        qa = math.floor((1 - ph[a] - delta[a]) * 1e9) / 1e9
        ok, info = lc_certificate_parallel(qa, fdown(1 - A(a)), procs=procs, pieces=pieces)
        certs[a] = dict(q_a=qa, p_a=fdown(1 - A(a)), delta=delta[a], ok=ok, boxes=info['boxes'])
        if verbose:
            print(f'node {i + 1}/{len(delta)}', a, certs[a], flush=True)
        phi[a] = fup(1 - A(qa)) if ok else 1.0
    worst = -math.inf
    for a0, a1 in zip(grid[:-1], grid[1:]):
        s = A(phi[a1]) + A(phi[partner(a0)])
        worst = max(worst, fup(s))
    budget = fdown(1 - A(q))
    ok = worst <= budget and all(c['ok'] for c in certs.values())
    return ok, worst, budget, dict(grid=grid, nodes={str(a): c for a, c in certs.items()})
