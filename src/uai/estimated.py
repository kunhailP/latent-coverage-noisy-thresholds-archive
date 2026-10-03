"""Latent-coverage radii when the noise variances are estimated (Supplementary Material, Section S6).

Structured model S: D_i = sigma^2 a_i with known a_i > 0 and one unknown scale sigma^2, estimated
by a pooled variance with nu degrees of freedom (nu sigma_hat^2 / sigma^2 ~ chi2_nu, independent
of the noisy scores). On the event {sigma^2 in [lo, hi]} (probability >= 1 - eta) and the
order-statistic event (>= 1 - delta), the radius

    s = T sup_{sigma^2 in [lo, hi]} R^mix_{p_k - eps, q}(sigma^2 a / T^2)

covers a new latent residual with probability >= q (Supplementary Lemma S4); reliability >= 1 - delta - eta.

The supremum is covered by finitely many points with Supplementary Lemma S5, R(lambda x) <= sqrt(lambda) R(x)
for lambda >= 1 (all variances scaled together):

    sup_{[s_j, s_{j+1}]} R <= sqrt(s_{j+1} / s_j) U(s_j),

with U(s) a value of R^mix at sigma^2 = s (a certified upper bound in `certify=True` mode, the
converged grid value otherwise). No monotonicity of R in sigma^2 is assumed.
"""
import numpy as np
from scipy import stats

from uai.procedures import pac_rank, quantised_kernel, shrink_mix


def scale_interval(s2_hat, nu, eta_lo=0.009, eta_hi=0.001):
    """[lo, hi] with P(sigma^2 < lo) = eta_lo and P(sigma^2 > hi) = eta_hi. The lower end gets
    most of the budget because R usually decreases in sigma^2, so the supremum sits near lo."""
    return (nu * s2_hat / stats.chi2.ppf(1 - eta_lo, nu), nu * s2_hat / stats.chi2.ppf(eta_hi, nu))


def radius_at(p, q, xs, n_pts=32, certify=False, tols=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1)):
    """R^mix_{p - eps, q}(xs): (value, eps, certified?)."""
    (pts, wts), eps = quantised_kernel(xs, n_pts)
    R = shrink_mix(p - eps, q, pts, wts=wts)
    if not certify or not np.isfinite(R):
        return R, eps, False
    from uai.certify import certified_upper
    U, tol, _ = certified_upper(p - eps, q, pts, R, rel_tols=tols, ws=wts, max_boxes=8_000_000)
    return (U, eps, True) if np.isfinite(U) else (np.inf, eps, False)


def sup_scale_radius(p, q, a, T, lo, hi, r0=1.004, max_cells=60, certify=False):
    """Bound on sup_{s in [lo, hi]} R^mix_{p - eps, q}(s a / T^2) by Supplementary Lemma S5 on an adaptive grid.
    Returns (bound, cells) with cells = [(s_j, s_{j+1}, U(s_j))]."""
    a = np.asarray(a, dtype=float)
    U0 = radius_at(p, q, lo * a / T**2, certify=certify)[0]
    if not np.isfinite(U0):
        return np.inf, []
    target = np.sqrt(r0) * U0
    s, U, cells, bound = lo, U0, [], 0.0
    while s < hi and len(cells) < max_cells:
        ratio = max(r0, (target / U) ** 2) if U > 0 else np.inf
        s_next = min(hi, s * ratio)
        cells.append((s, s_next, U))
        bound = max(bound, np.sqrt(s_next / s) * U)
        s = s_next
        if s < hi:
            U = radius_at(p, q, s * a / T**2, certify=certify)[0]
            if not np.isfinite(U):
                return np.inf, cells
    if s < hi:                               # ran out of cells: close with Supplementary Lemma S5 in one step
        bound = max(bound, np.sqrt(hi / s) * U)
    return bound, cells


def s_rule(V, a, s2_hat, nu, q=0.90, delta=0.04, eta_lo=0.009, eta_hi=0.001, k=None,
           certify=False):
    """Half-width of rule S: order-statistic threshold T = |V|_(k) at level p_k (delta), variance
    scale interval at level eta = eta_lo + eta_hi; reliability >= 1 - delta - eta under model S."""
    V = np.asarray(V)
    K = len(V)
    k = pac_rank(K, q, delta) if k is None else k
    p_k = stats.beta.ppf(delta, k, K + 1 - k)
    if k < K * p_k + 1:
        raise ValueError('Hoeffding comparison (Supplementary Proposition S2) needs k >= K p_k + 1')
    T = np.sort(np.abs(V))[k - 1]
    lo, hi = scale_interval(s2_hat, nu, eta_lo, eta_hi)
    bound, cells = sup_scale_radius(p_k, q, a, T, lo, hi, certify=certify)
    return T * bound, dict(T=T, p_k=p_k, lo=lo, hi=hi, cells=len(cells))


# ---------------------------------------------------------------------------------------------
# Model H: area-wise variances, no pooling (Supplementary Material, Section S6). Exploratory numerics only:
# the envelope kernel is tabulated on a grid and R[u] is maximised over a (slope, length, shift)
# grid, so the radius is a converging lower value of R[u], not a certified bound.
# ---------------------------------------------------------------------------------------------

def kernel_envelope(w, h0=0.05):
    """Continuous majorant of M(w) = sup_{x > 0} g_x(w), vanishing at +-infinity. For |w| > 1,
    g_x(w) = P(a/s <= Z <= b/s) with a = |w| - 1, b = |w| + 1, s = sqrt(x); its only critical
    point is s^2 = (b^2 - a^2) / (2 log(b/a)) = 2|w| / log((|w|+1)/(|w|-1)), and M(w) <= 1/2.
    M jumps from 1 to 1/2 at |w| = 1, so a ramp of width h0 is added to keep the majorant
    continuous (Supplementary Lemma S4 asks for a continuous kernel)."""
    from scipy.special import ndtr
    w = np.abs(np.asarray(w, dtype=float))
    out = np.ones_like(w)
    o = w > 1
    s = np.sqrt(2 * w[o] / np.log((w[o] + 1) / (w[o] - 1)))
    M = ndtr((1 - w[o]) / s) - ndtr((-1 - w[o]) / s)
    out[o] = np.minimum(1.0, M + 0.5 * np.maximum(0.0, 1 - (w[o] - 1) / h0))
    return out


def h_kernel(D_hat, nu, T, alpha=0.002, eta=0.01, w_max=8.0, h=0.002):
    """Upper envelope u >= gbar_{D,T} on the event {at most N* area-wise intervals miss}, where
    the area-wise intervals have level 1 - alpha (chi-square, nu_i d.f.) and N* is the
    (1 - eta)-quantile of Bin(K, alpha). A missed area contributes at most the kernel envelope
    M(w) (`kernel_envelope`), so u is continuous and vanishes at +-infinity. Returns (w, u, N*)."""
    from scipy.special import ndtr
    D_hat, nu = np.asarray(D_hat, float), np.broadcast_to(np.asarray(nu, float), np.shape(D_hat))
    K = len(D_hat)
    L = nu * D_hat / stats.chi2.ppf(1 - alpha / 2, nu) / T**2
    U = nu * D_hat / stats.chi2.ppf(alpha / 2, nu) / T**2
    n_star = int(stats.binom.ppf(1 - eta, K, alpha))
    w = np.arange(-w_max, w_max + h / 2, h)
    A = lambda x: ndtr((1 - w[None, :]) / np.sqrt(x[:, None]))
    B = lambda x: ndtr((-1 - w[None, :]) / np.sqrt(x[:, None]))
    env = np.maximum(A(L), A(U)) - np.minimum(B(L), B(U))
    u = env.sum(0)
    if n_star > 0:
        u += np.sort(kernel_envelope(w)[None, :] - env, axis=0)[-n_star:].sum(0)
    return w, np.minimum(u / K, 1.0), n_star


def radius_generic(p, q, w, u, betas=None, ells=None):
    """max Q_q(|W|) over point masses and log-affine segments W = c + Y (density prop. to
    exp(beta y) on [0, ell]) with int u dG >= p, u tabulated on the uniform grid w. For each
    (beta, ell) the latent quantile is quasi-convex in the shift, so only the smallest and the
    largest feasible shifts matter (u need not be log-concave)."""
    from uai.extremal import latent_abs_quantile
    h = w[1] - w[0]
    betas = np.r_[0.0, np.geomspace(0.02, 50, 14)] if betas is None else betas
    ells = np.geomspace(0.01, 6, 24) if ells is None else ells
    feas = u >= p
    best = float(np.max(np.abs(w[feas]))) if feas.any() else -np.inf
    for beta in betas:
        for ell in ells:
            y = np.arange(0, ell + h / 2, h)
            f = np.exp(beta * (y - ell)); f /= f.sum()
            m = np.convolve(u, f[::-1], mode='valid')
            idx = np.nonzero(m >= p)[0]
            if len(idx) == 0:
                continue
            for j in (idx[0], idx[-1]):
                c = w[j]
                best = max(best, latent_abs_quantile(q, c, c + y[-1], max(beta, 1e-7)))
    return best
