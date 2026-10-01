"""Log-concave extremal computations.

Notation. W is the latent score of an area (theta - centre), e ~ N(0, x) independent survey
noise, V = W + e the observed score. Scale is fixed so that the noisy threshold is t = 1,
hence x = D / t^2.

Localization (Fradelizi-Guedon): with one linear constraint on the law of W, the extreme
points of the relevant set of log-concave laws are Dirac masses and log-affine laws on a
segment. So both problems below are searched over
    W = loc + sc * Y,   Y on [0, 1] with density proportional to exp(beta * y).
With p constraints the extremals are piecewise log-affine with at most p pieces; the
quantile/sd constant (two moment constraints) is attained by a truncated Laplace law.

These are numerical searches (differential evolution). A search can under-estimate a
supremum, which is the non-conservative direction.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq, differential_evolution, minimize_scalar
from scipy.stats import norm

_GY, _GW = leggauss(200)
_GY = (_GY + 1) / 2
_GW = _GW / 2


def _weights(beta):
    lw = beta * _GY
    lw = lw - lw.max()
    w = np.exp(lw) * _GW
    return w / w.sum()


def _cdf_y(beta, y):
    y = np.clip(y, 0, 1)
    if abs(beta) < 1e-8:
        return y
    if beta > 50:
        return np.where(y > 0, np.exp(np.minimum(beta * (y - 1), 0)), 0.0)
    return np.expm1(beta * y) / np.expm1(beta)


def noisy_mass(beta, loc, sc, x, t=1.0):
    """P(|W + e| <= t) for W = loc + sc*Y, e ~ N(0, x)."""
    w = loc + sc * _GY
    s = np.sqrt(x)
    return float((_weights(beta) * (norm.cdf((t - w) / s) - norm.cdf((-t - w) / s))).sum())


def latent_mass(beta, loc, sc, q):
    """P(|W| <= q)."""
    return float(_cdf_y(beta, (q - loc) / sc) - _cdf_y(beta, (-q - loc) / sc))


def latent_quantile(beta, loc, sc, level):
    """level-quantile of |W|."""
    lo, hi = 0.0, abs(loc) + sc
    for _ in range(60):
        mid = (lo + hi) / 2
        if latent_mass(beta, loc, sc, mid) < level:
            lo = mid
        else:
            hi = mid
    return hi


def shrink_factor(x, alpha=0.10, seeds=(0, 1, 2)):
    """r_alpha(x) = sup over log-concave W of the (1-alpha)-quantile of |W|
    subject to P(|W+e| <= 1) >= 1 - alpha, e ~ N(0, x)."""
    lev = 1 - alpha

    def neg(p):
        beta, loc = p
        f = lambda sc: noisy_mass(beta, loc, sc, x) - lev
        if f(1e-7) < 0:
            return 0.0
        hi = 0.5
        while f(hi) > 0 and hi < 50:
            hi *= 2
        if f(hi) > 0:
            return 0.0
        sc = brentq(f, 1e-7, hi, xtol=1e-12)
        return -latent_quantile(beta, loc, sc, lev)

    best = 0.0
    for seed in seeds:
        r = differential_evolution(neg, [(-80, 80), (-1.5, 1.5)], seed=seed, maxiter=1000,
                                   popsize=40, tol=1e-12, polish=True)
        best = max(best, -r.fun)
    locs = np.linspace(0, 1.5, 30001)                      # Dirac candidates
    ok = norm.cdf((1 - locs) / np.sqrt(x)) - norm.cdf((-1 - locs) / np.sqrt(x)) >= lev
    if ok.any():
        best = max(best, locs[ok].max())
    return best


def worst_latent_coverage(x, p, s0, seeds=(0, 1, 2)):
    """H(p) = inf over log-concave W of P(|W| <= s0) subject to P(|W+e| <= 1) >= p."""
    def obj(z):
        beta, loc, lsc = z
        sc = np.exp(lsc)
        c = noisy_mass(beta, loc, sc, x)
        return latent_mass(beta, loc, sc, s0) + 20 * max(0, p - c)

    best = 1.0
    for seed in seeds:
        r = differential_evolution(obj, [(-80, 80), (-1.5, 1.5), (np.log(1e-4), np.log(8))],
                                   seed=seed, maxiter=1500, popsize=40, tol=1e-12, polish=True)
        best = min(best, r.fun)
    return best


def truncated_laplace_ratio(c, alpha):
    """(1-alpha)-quantile of |X - EX| / sd(X) for X with density prop. to exp(-|x|) on [-c, c]."""
    Z = 1 - np.exp(-c)
    u = -np.log(1 - (1 - alpha) * Z)
    m2 = (2 - np.exp(-c) * (c * c + 2 * c + 2)) / Z
    return u / np.sqrt(m2)


def quantile_sd_constant(alpha):
    """sup over log-concave laws of the (1-alpha)-quantile of |X - EX| / sd(X);
    attained in the truncated Laplace family. Returns (constant, truncation point)."""
    r = minimize_scalar(lambda c: -truncated_laplace_ratio(c, alpha), bounds=(0.01, 50),
                        method='bounded', options={'xatol': 1e-12})
    return -r.fun, r.x


def worst_coverage_normal_interval(alpha, grid=np.linspace(0.01, 30, 30000)):
    """inf over the truncated Laplace family of P(|X - EX| <= z_{1-alpha/2} sd)."""
    z = norm.ppf(1 - alpha / 2)
    Z = 1 - np.exp(-grid)
    m2 = (2 - np.exp(-grid) * (grid**2 + 2 * grid + 2)) / Z
    u = np.minimum(z * np.sqrt(m2), grid)
    cov = (1 - np.exp(-u)) / Z
    i = np.argmin(cov)
    return cov[i], grid[i]
