"""Latent-side boundary: unimodal (mean-zero) latent laws, Gaussian or uniform noise.
W: uniform density on [-A,-1] (mass tau), uniform on [-1,1] (mass m), uniform on [1,1+delta] (mass s).
Density nondecreasing up to 1+delta -> unimodal. A chosen so E(W)=0. Exact noisy coverage by quadrature."""
import numpy as np
from scipy.stats import norm, uniform
from scipy.integrate import quad
from scipy.optimize import brentq

def noisy_cov(pieces, noise):
    # pieces: list of (a,b,mass) uniform; P(|W+e|<=1)
    tot = 0.0
    for a, b, mass in pieces:
        f = lambda w: noise.cdf(1 - w) - noise.cdf(-1 - w)
        pts = [x for x in (-1, 1) if a < x < b]
        tot += mass / (b - a) * quad(f, a, b, points=pts or None, limit=500)[0]
    return tot

def build(s, tau, delta):
    m = 1 - s - tau
    # mean zero: s*(1+delta/2) + tau*(-(A+1)/2) = 0  (inside part mean 0)
    A = 2 * s * (1 + delta / 2) / tau - 1
    assert tau / (A - 1) <= m / 2 + 1e-15, 'left tail density must not exceed inside density'
    return [(-A, -1, tau), (-1, 1, m), (1, 1 + delta, s)], m, A

def worst(p, noise_family, sigma, delta, tau):
    noise = noise_family(sigma)
    g = lambda s: noisy_cov(build(s, tau, delta)[0], noise) - p
    s = brentq(g, 1e-6, 0.6 - tau)
    pieces, m, A = build(s, tau, delta)
    return m, s, A

fam = {'Gaussian': lambda sg: norm(0, sg), 'uniform': lambda sg: uniform(-sg * 3**.5, 2 * sg * 3**.5)}
for p in (0.9, 0.95):
    print(f'noisy coverage {p}: shape-free bound 2p-1 = {2*p-1:.3f}')
    for name, nf in fam.items():
        for sigma in (0.1, 0.03, 0.01, 0.003):
            m, s, A = worst(p, nf, sigma, delta=sigma * 1e-3, tau=1e-4)
            print(f'  {name:8s} sd/t={sigma:<6} latent coverage={m:.4f}  spike mass={s:.4f}  left tail to -{A:.0f}')
