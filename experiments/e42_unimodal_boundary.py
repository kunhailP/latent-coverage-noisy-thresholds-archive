"""Latent-side boundary (Proposition 2): unimodal mean-zero latent laws.

W has density tau/(A - 1) on [-A, -1), m/2 on [-1, 1] and s/delta on (1, 1 + delta], with
m = 1 - s - tau; the density is nondecreasing up to 1 + delta, so W is unimodal, and A makes
E(W) = 0. The noisy coverage P(|W + e| <= 1) is computed in closed form: for a uniform piece on
[a, b], int_a^b F(c - w) dw = G(c - a) - G(c - b) with G an antiderivative of the noise
distribution function F. Noise laws: Gaussian, uniform (median zero) and the centred
exponential sigma (Exp(1) - 1) (mean zero, log-concave, P(e >= 0) = 1/e).

Limits as sigma -> 0, delta/sigma -> 0, tau -> 0: 2p - 1 for median-zero noise and
1 - e (1 - p) for the centred exponential.

  python experiments/e42_unimodal_boundary.py
"""
import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def G_gauss(y, sigma):
    z = y / sigma
    return y * norm.cdf(z) + sigma * norm.pdf(z)


def G_uniform(y, sigma):
    h = sigma * math.sqrt(3)                    # standard deviation sigma
    if y <= -h:
        return 0.0
    if y >= h:
        return y
    return (y + h) ** 2 / (4 * h)


def G_cexp(y, sigma):
    if y <= -sigma:                              # e = sigma (E - 1) >= -sigma
        return 0.0
    return y + sigma * math.exp(-(y / sigma + 1))


NOISE = {'Gaussian': G_gauss, 'uniform': G_uniform, 'centred exponential': G_cexp}


def build(s, tau, delta):
    m = 1 - s - tau
    A = 2 * s * (1 + delta / 2) / tau - 1
    assert tau / (A - 1) <= m / 2, 'left tail density must not exceed the inside density'
    return [(-A, -1.0, tau), (-1.0, 1.0, m), (1.0, 1 + delta, s)], m, A


def noisy_cov(pieces, G, sigma):
    """P(|W + e| <= 1) = sum over pieces of mass/(b - a) int_a^b {F(1 - w) - F(-1 - w)} dw."""
    tot = 0.0
    for a, b, mass in pieces:
        I = (G(1 - a, sigma) - G(1 - b, sigma)) - (G(-1 - a, sigma) - G(-1 - b, sigma))
        tot += mass / (b - a) * I
    return tot


def worst(p, G, sigma, delta, tau):
    g = lambda s: noisy_cov(build(s, tau, delta)[0], G, sigma) - p
    hi = 0.9 - tau
    if g(1e-9) < 0:
        raise ValueError('infeasible')
    s = brentq(g, 1e-9, hi, xtol=1e-15)
    pieces, m, A = build(s, tau, delta)
    return m, s, A, noisy_cov(pieces, G, sigma)


if __name__ == '__main__':
    for p in (0.9, 0.95):
        print(f'noisy coverage {p}: 2p - 1 = {2 * p - 1:.4f}, 1 - e(1 - p) = {1 - math.e * (1 - p):.4f}')
        for name, G in NOISE.items():
            for sigma in (0.1, 0.03, 0.01, 0.003, 0.001):
                try:
                    m, s, A, nc = worst(p, G, sigma, delta=sigma * 1e-3, tau=1e-4)
                    print(f'  {name:20s} sd={sigma:<6} noisy={nc:.10f} latent={m:.4f} '
                          f'spike={s:.4f} tail to -{A:.0f}')
                except ValueError:
                    print(f'  {name:20s} sd={sigma:<6} infeasible for this family')
