"""E35: the radius correction under different locations of the mean (Theorems 1-3, Supplementary Proposition S1).

For q = 0.9 and noise-to-threshold variance ratios x from 1e-7 to 1e-2, each law is scaled so
that the noisy coverage pr(|W + x^{1/2} Z| <= 1) is exactly q, and the radius correction
delta = Q_q(|W|) - 1 is computed from closed forms.
  boundary      W = 1 + x^{1/2}(b - E_u) at the maximizer u* of v_q: mean 1 - k(u*) x^{1/2};
                delta / x^{1/2} -> c_q (Theorem 1)
  kappa50       the same family at the rate u with k(u) = 50: mean at distance 50 x^{1/2};
                delta / x^{1/2} -> L_q(50) (Theorem 3)
  centred       a fixed mean-zero law of Supplementary Proposition S1 (beta l = 0.5 on a unit segment);
                delta / x -> (beta r0 / 2) tanh(beta r0) = 0.0240. Larger coefficients (up to
                0.0747 at beta l = 0.9375) sit closer to the end of the support, and their order-x
                widening appears only once the noise is far smaller than that gap: at beta l = 0.9
                (coefficient 0.070, gap 0.0011) only for x below about 1e-7
  gaussian      W ~ N(1 - 2 x^{1/2}, (0.1 x^{1/2})^2) before scaling: mean in the boundary layer,
                but delta < 0, so the layer is necessary for a large correction, not sufficient
The boundary laws keep the exact two-sided noisy mass, including leakage through the far end.
  python experiments/e35_mean_location.py      -> results/mean_location.csv
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr

from _common import RESULTS
from uai.extremal import (_tail_vk, centred_exp_coefficient, centred_exp_law, latent_abs_quantile,
                          noisy_abs_quantile, tail_optimum)

Q = 0.9
XS = np.geomspace(1e-7, 1e-2, 11)


def tail_cdf(y, b, u, sigma):
    """pr(b - E_u + sigma Z <= y)."""
    return ndtr((y - b) / sigma) + np.exp(-u * (b - y) + u * u * sigma * sigma / 2) * ndtr((b - y) / sigma - u * sigma)


def boundary_law(x, u):
    """W = 1 + h(b - E_u), h = x^{1/2}, b chosen so that pr(|W + h Z| <= 1) = Q exactly."""
    h = np.sqrt(x)
    cov = lambda b: tail_cdf(0.0, b, u, 1.0) - tail_cdf(-2 / h, b, u, 1.0)
    b = brentq(lambda b: cov(b) - Q, -20, 20 + 5 / u)
    FW = lambda w: min(1.0, np.exp(u * ((w - 1) / h - b)))
    r = brentq(lambda r: FW(r) - FW(-r) - Q, 1e-9, 1 + h * b)
    return r - 1, 1 + h * (b - 1 / u)


def gaussian_law(x):
    """W ~ N(m, s^2) with m = 1 - 2 x^{1/2}, s = 0.1 x^{1/2}, then scaled by the noisy quantile."""
    m, s = 1 - 2 * np.sqrt(x), 0.1 * np.sqrt(x)
    def abs_q(sd):
        return brentq(lambda r: ndtr((r - m) / sd) - ndtr((-r - m) / sd) - Q, 0, m + 20 * sd)
    t = abs_q(np.sqrt(s * s + x))                               # noisy q-quantile of |W + e|
    xx = x / t**2                                               # normalized ratio after scaling
    return abs_q(s) / t - 1, m / t, xx


def centred_law(beta_ell):
    a, b = centred_exp_law(beta_ell, 1.0)
    r0 = latent_abs_quantile(Q, a, b, beta_ell)
    rows = []
    for v in np.geomspace(1e-9, 1e-3, 25):
        t = noisy_abs_quantile(Q, a, b, beta_ell, v)
        rows.append((v / t**2, r0 / t - 1))
    return np.array(rows), centred_exp_coefficient(Q, beta_ell, 1.0)


if __name__ == '__main__':
    u_star, c_q, k_star = tail_optimum(Q)
    u50 = brentq(lambda u: _tail_vk(Q, u)[1] - 50.0, 1e-4, u_star)
    rows = []
    for x in XS:
        for law, u in (('boundary', u_star), ('kappa50', u50)):
            try:                    # infeasible once the mean distance k(u) x^{1/2} is of order one
                d, mean = boundary_law(x, u)
            except ValueError:
                continue
            rows.append(dict(law=law, x=x, delta=d, mean_distance=(1 - mean) / np.sqrt(x)))
    for x in XS:
        d, m, xx = gaussian_law(x)
        rows.append(dict(law='gaussian', x=xx, delta=d, mean_distance=(1 - m) / np.sqrt(xx)))
    cen, coef = centred_law(0.5)
    for xx, d in cen:
        rows.append(dict(law='centred', x=xx, delta=d, mean_distance=1 / np.sqrt(xx)))
    df = pd.DataFrame(rows)
    df['delta_over_sqrt_x'] = df.delta / np.sqrt(df.x)
    df['delta_over_x'] = df.delta / df.x
    df.to_csv(RESULTS / 'mean_location.csv', index=False)
    print(f'c_q = {c_q:.6f}, L_q(50) = {_tail_vk(Q, u50)[0]:.6f}, centred coefficient = {coef:.6f}')
    with pd.option_context('display.width', 150):
        for law, g in df.groupby('law', sort=False):
            print(law); print(g.iloc[::max(1, len(g) // 6)].round(8).to_string(index=False))
