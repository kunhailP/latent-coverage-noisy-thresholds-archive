"""Fast checks of the core numbers. Run: python -m pytest -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from uai.logconcave import (quantile_sd_constant, worst_coverage_normal_interval,  # noqa: E402
                            noisy_mass, latent_quantile)
from uai.procedures import ShrinkTable                          # noqa: E402


def test_log_concave_quantile_constant():
    c90, _ = quantile_sd_constant(.10); c95, _ = quantile_sd_constant(.05)
    assert abs(c90 - 1.75585) < 1e-4 and abs(c95 - 2.16907) < 1e-4


def test_worst_normal_interval_coverage():
    assert abs(worst_coverage_normal_interval(.10)[0] - .8775) < 5e-4
    assert abs(worst_coverage_normal_interval(.05)[0] - .9297) < 5e-4


def test_shrink_table_monotone_and_conservative_lookup():
    T = json.load(open(ROOT / 'results' / 'r_table.json'))
    for a, tab in T.items():
        xs = sorted(float(k) for k in tab); rs = [tab[str(x)] for x in xs]
        assert all(rs[i + 1] <= rs[i] + 1e-6 for i in range(len(rs) - 1))
    st = ShrinkTable('0.1')
    assert st(0.136) >= st(0.137) and st(0.5) == 1.0


def test_gaussian_is_feasible_and_below_ldc_bound():
    # Gaussian W with P(|V|<=1) = .9 at x = .136: its 90% quantile must not exceed r_.10(.136)
    from scipy.stats import norm
    x = .136; sW = np.sqrt((1 / norm.ppf(.95))**2 - x)
    assert 1.645 * sW <= ShrinkTable('0.1')(0.136) + 1e-9


def test_pac_rank():
    from uai.procedures import pac_rank
    assert pac_rank(110, .90, .05) == 105
    assert pac_rank(110, .90, .04) == 105


def test_asymmetric_small_noise_counterexample():
    """Log-affine W on [0, b]: noisy mass >= .90 at t = 1 but latent mass < .90, so r > 1."""
    from scipy.integrate import quad
    from scipy.stats import norm
    b, x = 1.0681, 0.0005
    f = lambda w: np.exp(w) / np.expm1(b)
    s = np.sqrt(x)
    noisy = quad(lambda w: f(w) * (norm.cdf((1 - w) / s) - norm.cdf((-1 - w) / s)), 0, b,
                 points=[1], epsabs=1e-14, limit=500)[0]
    latent = np.expm1(1) / np.expm1(b)
    assert noisy > 0.90 > latent


def test_shrink_table_small_x_not_one():
    from uai.procedures import ShrinkTable
    t = ShrinkTable('0.1')
    assert t(1e-5) == t.r.max() > 1              # left of the peak: overall maximum
    assert t(0.003) >= t(0.004) >= t(0.2)


def test_union_bound_baseline():
    """Monte Carlo check of the shape-free baseline on a skewed W."""
    from uai.procedures import union_bound_halfwidth
    rng = np.random.default_rng(0)
    W = rng.exponential(1, 400000) - 1; D = 0.3
    V = W + rng.normal(0, np.sqrt(D), W.size)
    t = np.quantile(np.abs(V), 0.95); s = union_bound_halfwidth(t, D, 0.95, 0.90)
    assert np.mean(np.abs(W) <= s) >= 0.90
    assert union_bound_halfwidth(1, D, 0.9, 0.9) == np.inf


def test_latent_laws_match_e12_draws():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
    from e12_conditional_synth import SHAPES, draw
    from uai.latent_laws import cdf, matched_scales
    rng = np.random.default_rng(3)
    grid = np.linspace(-3, 3, 61)
    for k in list(SHAPES) + ['bimodal_blc']:
        w = np.sort(draw(k, 400000, rng))
        emp = np.searchsorted(w, grid, side='right') / w.size
        assert np.max(np.abs(emp - cdf(k, grid))) < 0.004, k
    m = matched_scales('normal', [0, 0, 0], [-1, -2, -3], [1, 2, 3])
    assert abs(m['lam_pac'] - 1.6448536269514722) < 1e-9 and m['achieved_reliability'] == 1


def test_small_noise_upper_bound_all_x():
    """Theorem 1: R_{q,q}(x) <= 1 + c_q sqrt(x); grid values of R are lower bounds of R."""
    from uai.extremal import R_grid, one_sided_constant
    cq = one_sided_constant(.9, .9)
    assert abs(cq - 0.0190618) < 1e-6
    betas = np.concatenate([[0], np.geomspace(.05, 3000, 30)]); ells = np.geomspace(1e-3, 4, 30)
    for x in (1e-5, 1e-3, 1e-2):
        R, _ = R_grid(.9, .9, x, betas, ells)
        assert R <= 1 + cq * np.sqrt(x) + 1e-12


def test_box_bounds_are_sound():
    """Box bounds of Supplementary Section S3: every law inside a box obeys the box's noisy upper / latent lower bound."""
    from uai.certify import _cdf_latent, _cdf_noisy, box_status
    rng = np.random.default_rng(3)
    for _ in range(300):
        x, s = 10 ** rng.uniform(-2, -0.5), rng.uniform(.2, 1.3)
        u0, v0 = rng.uniform(0, .95, 2); u1, v1 = u0 + rng.uniform(0, .05), v0 + rng.uniform(0, .05)
        b0 = rng.uniform(-1.2, 2); b1 = b0 + rng.uniform(0, .2)
        box = np.array([[u0, u1, v0, v1, b0, b1]])
        u, v, b = rng.uniform(u0, u1, 20), rng.uniform(v0, v1, 20), rng.uniform(b0, b1, 20)
        be, el = u / (1 - u), v / (1 - v)
        N = _cdf_noisy(1.0, b, el, be, x) - _cdf_noisy(-1.0, b, el, be, x)
        L = _cdf_latent(s, b, el, be) - _cdf_latent(-s, b, el, be)
        ok_n, _ = box_status(N.max() - 1e-6, 2.0, s, [x], [1.0], box)     # p just below max N
        assert not ok_n[0]                                                  # N_up >= max N
        bad_l, _ = box_status(-1.0, L.min() + 1e-6, s, [x], [1.0], box)
        assert not bad_l[0]                                                 # L_low <= min L


def test_certify_brackets_R():
    from uai.certify import certify
    assert certify(.9, .9, .8984, .116)[0]
    assert not certify(.9, .9, .8970, .116, max_boxes=2_000_000)[0]


def test_certified_lookup_dominates_grid_values():
    """Feasible grid values (E16) are lower bounds of R, so they may never exceed the lookup."""
    import pandas as pd
    from uai.procedures import CertifiedShrinkTable
    tab = CertifiedShrinkTable(0.9)
    e = pd.read_csv(ROOT / 'results' / 'r_exact_p0.9_q0.9.csv')
    for x, r in zip(e.x, e.R):
        if np.isfinite(r) and x < tab.x_edge:
            assert r <= tab(x) + 1e-12


def test_interval_c_enclosure_contains_float_value():
    """Ball-arithmetic enclosure of c_.9 (E27) brackets the double-precision value."""
    from uai.extremal import one_sided_constant
    from uai.interval import c_enclosure
    lo, hi, _ = c_enclosure('0.9', '0.9', n0=200, max_iter=3000)
    assert lo <= one_sided_constant(.9, .9) <= hi and hi - lo < 1e-3


def test_interval_b_bounds_and_monotone_in_u():
    """b_u(p) is enclosed and nonincreasing in u (kappa_u(b) decreases in u)."""
    from uai.interval import A, b_bounds, kappa
    p = A('0.9')
    prev = np.inf
    for u in (0.05, 0.3, 1.0, 4.0, 20.0):
        lo, hi = b_bounds(u, p)
        assert lo < hi and hi <= prev + 1e-12
        prev = hi
        assert kappa(u, lo) > p and kappa(u, hi) < p
    assert kappa(2.0, 0.3) < kappa(1.0, 0.3)


def test_interval_constants_file():
    """E27 certificates: C_{.9036,.9} <= -0.057 and the slack brackets."""
    d = json.loads((ROOT / 'results' / 'interval_constants.json').read_text())
    assert all(s['ok'] for s in d['split'])
    assert any(s['g'] == -0.057 for s in d['split'])
    for q, s in d['slack'].items():
        assert s['enough']['ok'] and s['not_enough']['positive']
    from uai.procedures import CertifiedShrinkTable
    assert CertifiedShrinkTable.C_Q >= d['c']['0.9'][1]
    assert CertifiedShrinkTable.C_SPLIT[0.9036] >= -0.057


def test_min_variance_plugin_not_conservative():
    """Counterexample (checked in 192-bit balls): noisy coverage >= .9 under an
    equal mixture of N(0, 1e-8) and N(0, 5e-4), yet the min-variance radius 1 + c_.9 * 1e-4 covers
    W = b - Exp(2) with probability < .9."""
    from flint import arb, ctx
    old = ctx.prec
    ctx.prec = 192
    try:
        Phi = lambda z: (-z / arb(2).sqrt()).erfc() / 2
        b, rate = arb('1.04356470165499'), arb(2)

        def noisy_cdf(t, D):
            sd = D.sqrt(); z = (t - b) / sd
            return Phi(z) + (rate * (t - b) + rate**2 * D / 2).exp() * Phi(-z - rate * sd)

        nm = lambda D: noisy_cdf(1, D) - noisy_cdf(-1, D)
        D0, D1 = arb('1e-8'), arb('0.0005')
        h = 1 + arb('0.019062') * D0.sqrt()
        assert (nm(D0) + nm(D1)) / 2 > arb('0.9')
        assert (rate * (h - b)).exp() - (rate * (-h - b)).exp() < arb('0.9')
    finally:
        ctx.prec = old


def test_outward_rounding():
    from flint import arb
    from uai.interval import A, fdown, fup, log_grid
    z = arb(1) / 3
    assert A(float(z.upper())) < z                 # plain conversion can round down
    assert A(fup(z)) >= z and A(fdown(z)) <= z
    T = 1 - A('0.9036')
    assert A(fup(T / 32)) >= T / 32
    g = log_grid(0.0189, 64.0, 50)
    assert g[0] == 0.0189 and g[-1] == 64.0 and np.all(np.diff(g) >= 0)


def test_envelope_dominates_lookup_to_the_right():
    """envelope(x_low) >= lookup(x) for every x >= x_low below the feasibility edge."""
    from uai.procedures import CertifiedShrinkTable
    for p in (0.9, 0.9036, 0.9068):
        tab = CertifiedShrinkTable(p)
        xs = np.linspace(1e-5, tab.x_edge * 0.999, 400)
        look = np.array([tab(x) for x in xs])
        for i in range(0, 400, 23):
            assert tab.envelope(xs[i]) >= look[i:].max() - 1e-12


def test_quantised_kernel_merges_equal_variances():
    from uai.procedures import quantised_kernel
    (pts, wts), eps = quantised_kernel(np.r_[np.full(60, .1), np.full(50, .2)])
    assert eps == 0.0 and np.allclose(pts, [.1, .2]) and np.allclose(wts, [60 / 110, 50 / 110])


def test_scale_lemma_on_grid_values():
    """Supplementary Lemma S3: R(lambda x) <= sqrt(lambda) R(x), checked on converged grid values."""
    from uai.procedures import shrink_mix
    xs, w = np.array([.05, .15]), np.array([.5, .5])
    r1 = shrink_mix(.9036, .9, xs, wts=w)
    for lam in (1.1, 1.5, 3.0):
        assert shrink_mix(.9036, .9, lam * xs, wts=w) <= np.sqrt(lam) * r1 + 1e-9


def test_areawise_envelope_dominates_true_kernel():
    """Model H: on the event that every area-wise interval covers, u >= gbar pointwise."""
    from scipy.special import ndtr
    from uai.estimated import h_kernel
    rng = np.random.default_rng(0)
    D = rng.uniform(.1, 1, 40); Dh = D * rng.chisquare(20, 40) / 20; T = 2.0
    w, u, ns = h_kernel(Dh, 20, T, alpha=1e-9, eta=.5)
    from scipy import stats
    L = 20 * Dh / stats.chi2.ppf(1 - 5e-10, 20); U = 20 * Dh / stats.chi2.ppf(5e-10, 20)
    assert np.all((L <= D) & (D <= U))
    x = D[:, None] / T**2
    g = (ndtr((1 - w) / np.sqrt(x)) - ndtr((-1 - w) / np.sqrt(x))).mean(0)
    assert np.all(u >= g - 1e-12)


def test_kernel_envelope_majorises_and_vanishes():
    from scipy.special import ndtr
    from uai.estimated import kernel_envelope
    w = np.linspace(-6, 6, 1201)
    M = kernel_envelope(w)
    for x in np.geomspace(1e-4, 100, 60):
        g = ndtr((1 - w) / np.sqrt(x)) - ndtr((-1 - w) / np.sqrt(x))
        assert np.all(g <= M + 1e-12)
    assert kernel_envelope(np.array([50.0]))[0] < .02
    near = kernel_envelope(np.array([1 - 1e-9, 1 + 1e-9, -1 - 1e-9]))
    assert np.all(np.abs(near - 1) < 1e-6)                  # continuous at |w| = 1 (ramp)


def test_shape_free_markov_is_valid_for_a_bimodal_law():
    """The Markov bound needs no shape: check the population version on a two-point latent law."""
    from scipy.special import ndtr
    from uai.procedures import shape_free_markov
    rng = np.random.default_rng(1)
    K = 110
    D = np.full(K, .3)
    W = rng.choice([-1.5, 1.5], K)
    V = W + rng.normal(0, np.sqrt(D))
    h, p_k, T = shape_free_markov(V, D, 108)
    # on the order-statistic event, P(|V| <= T) >= p_k; the latent two-point law is covered
    # with probability 1 or 0, and the bound says it is covered when the noisy mass allows
    noisy = ndtr((T - 1.5) / np.sqrt(.3)) - ndtr((-T - 1.5) / np.sqrt(.3))
    if noisy >= p_k:
        assert h >= 1.5


def test_transition_constant_q90():
    """L_q(kappa) = sup{v(u) : k(u) >= kappa}; flat at c_q up to kappa*, then below U."""
    from uai.extremal import tail_optimum, transition_constant, transition_upper
    u, c, k = tail_optimum(0.9)
    assert abs(c - 0.0190618170813) < 1e-10 and abs(k - 20.18233) < 1e-4
    assert transition_constant(0.9, 20.0) == c
    assert abs(transition_constant(0.9, 25.0) - 0.0173931846936) < 1e-9
    kx = (1 - c * c) / (2 * c)                              # the two upper bounds cross
    assert abs(transition_constant(0.9, kx) / transition_upper(0.9, kx) - 0.879055) < 1e-5
    A = 1 - np.log(1 / 0.9)                                 # large kappa: L ~ A / (2 kappa)
    assert abs(transition_constant(0.9, 1000.0) * 2000 / A - 1) < 1e-3


def test_centred_sqrt_bound_on_segment_laws():
    """EW = mu, |mu| < t: Q_q(|W|) <= |mu| + sqrt((t - |mu|)^2 + D), t the noisy quantile."""
    from uai.extremal import centred_exp_law, latent_abs_quantile, noisy_abs_quantile
    for q in (0.5, 0.9, 0.97):
        for beta in (-3, -0.3, 0.3, 3):
            for ell in (0.5, 4.0):
                a, b = centred_exp_law(beta, ell)
                for mu in (0.0, 0.2):
                    for sd in (0.05, 0.5):
                        t = noisy_abs_quantile(q, a + mu, b + mu, beta, sd * sd)
                        if t > mu:
                            lat = latent_abs_quantile(q, a + mu, b + mu, beta)
                            assert lat <= mu + np.sqrt((t - mu) ** 2 + sd * sd) + 1e-9


def test_centred_example_needs_order_x_widening():
    """a mean-zero asymmetric law with widening (beta r0/2) tanh(beta r0) x + o(x) > 0."""
    from uai.extremal import (centred_exp_coefficient, centred_exp_law, latent_abs_quantile,
                              noisy_abs_quantile)
    C = centred_exp_coefficient(0.9, 0.2, 2.0)
    assert abs(C - 0.0156540177189) < 1e-10
    a, b = centred_exp_law(0.2, 2.0)
    r0 = latent_abs_quantile(0.9, a, b, 0.2)
    t = noisy_abs_quantile(0.9, a, b, 0.2, 0.005 ** 2)
    x = 0.005 ** 2 / t ** 2
    assert r0 > t and abs((r0 / t - 1) / x - C) < 1e-4


def test_edge_constant_and_variance_budget():
    """M_q and the closed form M_q (x_p - x)^{1/2} against certified values near the edge."""
    import pandas as pd
    from uai.extremal import edge_constant, edge_level, variance_budget
    M = edge_constant(0.9)
    assert abs(M - 1.8532115) < 1e-6 and 1.7558 < M < 3.1623      # symmetric LC < LC < shape-free
    assert abs(variance_budget(0.9, [0.3]) - (edge_level(0.9) - 0.3)) < 1e-12
    c = pd.read_csv(ROOT / 'results' / 'certified_R.csv')
    row = c[(c.p == 0.9) & (c.q == 0.9) & (np.abs(c.x - 0.3) < 1e-9)].iloc[0]
    closed = M * np.sqrt(edge_level(0.9) - 0.3)
    assert row.L <= closed * 1.001 and abs(row.L / closed - 1) < 1e-3


def _periodic_bilc(phase, eps=0.05):
    """Density (1 + eps cos(2 pi w + phase))/2 on (-1, 1), shifted to mean zero. Bi-log-concave
    because sup|f'| = eps pi <= inf f^2 = (1 - eps)^2/4 (Duembgen et al. 2017, Theorem 1), but
    neither log-concave nor unimodal. Returns the distribution function of W + s^{1/2} Z."""
    from scipy.integrate import quad
    from scipy.special import ndtr
    f = lambda t: (1 + eps * np.cos(2 * np.pi * t + phase)) / 2
    mu = quad(lambda t: t * f(t), -1, 1, epsabs=1e-13)[0]
    def cdf(t, s):
        if s == 0:
            u = min(1.0, t + mu)
            return 0.0 if u <= -1 else quad(f, -1, u, epsabs=1e-13)[0]
        sd = np.sqrt(s)
        cuts = sorted({-1.0, 1.0} | {v for v in (t + mu - 8 * sd, t + mu, t + mu + 8 * sd) if -1 < v < 1})
        return sum(quad(lambda z: f(z) * ndtr((t + mu - z) / sd), a, b, epsabs=1e-13)[0]
                   for a, b in zip(cuts, cuts[1:]))
    return cdf


def test_centred_bound_bilc_and_symmetric_counterexample():
    """the mean-zero bound Q_q(|W|)^2 <= Q_q(|W + s^{1/2} Z|)^2 + s on bi-log-concave laws that
    are not log-concave; and a symmetric bi-log-concave law for which noise shrinks the quantile,
    so symmetry alone (without unimodality) does not make noisy calibration conservative."""
    from scipy.optimize import brentq
    for phase in (0.0, np.pi / 2):
        cdf = _periodic_bilc(phase)
        for q in (0.3, 0.7, 0.9):
            r0 = brentq(lambda r: cdf(r, 0) - cdf(-r, 0) - q, 0, 3, xtol=1e-13)
            for s in (1e-3, 0.1, 1.0):
                r = brentq(lambda r: cdf(r, s) - cdf(-r, s) - q, 0, 6, xtol=1e-13)
                assert r * r + s - r0 * r0 >= -1e-10
    cdf = _periodic_bilc(0.0)
    r0 = brentq(lambda r: cdf(r, 0) - cdf(-r, 0) - 0.7, 0, 3, xtol=1e-13)
    r = brentq(lambda r: cdf(r, 1e-5) - cdf(-r, 1e-5) - 0.7, 0, 3, xtol=1e-13)
    assert r < r0 - 1e-7                                     # widening needed although symmetric


def _exp_tail_noisy_cdf(y, b, u, sigma):
    """pr(b - E_u + sigma Z <= y), E_u exponential of rate u."""
    from scipy.special import ndtr
    return ndtr((y - b) / sigma) + np.exp(-u * (b - y) + u * u * sigma * sigma / 2) * ndtr((b - y) / sigma - u * sigma)


def test_noisy_threshold_valid_for_unknown_variances_population():
    """Proposition 3 at population level: take the boundary-layer law that needs widening at
    x = 1e-4 (latent coverage of [-1, 1] below 0.9, noisy coverage 0.9). Its noisy coverage must
    stay below the certified level 0.901 for every Gaussian noise variance, so no mixture of
    variances can reach 0.901 either."""
    from scipy.optimize import brentq
    from uai.extremal import tail_optimum
    u, _, _ = tail_optimum(0.9)
    h = 1e-2                                                     # x = 1e-4
    cov = lambda b, d: (_exp_tail_noisy_cdf(0.0, b, u, np.sqrt(d) / h)
                        - _exp_tail_noisy_cdf(-2 / h, b, u, np.sqrt(d) / h))   # W = 1 + h(b - E_u)
    b = brentq(lambda b: cov(b, h * h) - 0.9, -5, 60)
    latent = np.exp(-u * b) - np.exp(u * (-2 / h - b))           # pr(|W| <= 1)
    assert latent < 0.9
    assert max(cov(b, d) for d in np.geomspace(1e-10, 1.0, 400)) < 0.901


def test_analytic_rules_values():
    from uai.procedures import noisy_threshold_halfwidth, pac_rank, simple_shrink_halfwidth
    rng = np.random.default_rng(3)
    D = rng.uniform(0.2, 1.0, 110)
    V = rng.normal(size=110) + rng.normal(0, np.sqrt(D))
    k = pac_rank(110, 0.9, 0.05)
    T = np.sort(np.abs(V))[k - 1]
    assert k == 105 and noisy_threshold_halfwidth(V) == T
    assert abs(simple_shrink_halfwidth(V, D.min()) - (T - 0.114 * np.sqrt(D.min()))) < 1e-12
    assert noisy_threshold_halfwidth(V, k=100) == np.inf        # level below the certified 0.901


def test_certified_rank_and_analytic_rule_defaults():
    """Default rank of the analytic rules reaches the certified slack level and the Hoeffding
    condition; the usual PAC rank can fall short (K = 46, 1000), and small K has no rank."""
    from scipy import stats
    from uai.procedures import (CERTIFIED_SLACK_LEVEL, certified_rank, noisy_threshold_halfwidth,
                                pac_rank, simple_shrink_halfwidth)
    assert [certified_rank(K) for K in (20, 28, 29, 46, 110, 1000)] == [None, None, 29, 46, 105, 917]
    assert pac_rank(46, .9, .05) == 45 and pac_rank(1000, .9, .05) == 916
    rng = np.random.default_rng(5)
    for K in (20, 28):
        V = rng.normal(size=K)
        assert noisy_threshold_halfwidth(V) == np.inf and simple_shrink_halfwidth(V, .1) == np.inf
    for K in (29, 46, 110, 1000):
        V = rng.normal(size=K)
        k = certified_rank(K)
        p_k = stats.beta.ppf(.05, k, K + 1 - k)
        assert p_k >= CERTIFIED_SLACK_LEVEL[.9] and k >= K * p_k + 1
        T = np.sort(np.abs(V))[k - 1]
        assert noisy_threshold_halfwidth(V) == T and simple_shrink_halfwidth(V, .1) <= T


def test_bimodal_simulation_law_is_bilogconcave_not_logconcave():
    """The Table 1 law `bimodal_blc`: two modes, f' F <= f^2 and -f' (1 - F) <= f^2 (bi-log-
    concavity, Duembgen et al. 2017), and log f not concave."""
    from scipy.stats import norm
    from uai.latent_laws import BIMODAL, cdf
    p, m1, m2, s = BIMODAL
    x = np.linspace(-8, 8, 400001)
    f = p * norm.pdf(x, m1, s) + (1 - p) * norm.pdf(x, m2, s)
    fp = -(p * norm.pdf(x, m1, s) * (x - m1) + (1 - p) * norm.pdf(x, m2, s) * (x - m2)) / s**2
    F = cdf('bimodal_blc', x)
    S = p * norm.sf(x, m1, s) + (1 - p) * norm.sf(x, m2, s)      # 1 - F without cancellation
    assert np.all(fp * F - f * f <= 0) and np.all(-fp * S - f * f <= 0)
    assert np.sum(np.diff(np.sign(np.diff(f))) < 0) == 2
    assert np.max(np.diff(np.log(f), 2)) > 0
    assert abs(p * m1 + (1 - p) * m2) < 1e-12 and abs(s**2 + p * m1**2 + (1 - p) * m2**2 - 1) < 1e-12


def test_bimodal_bilogconcavity_certificate():
    """Lemma S1: the ball-arithmetic certificate proves the Table 1 law bi-log-concave on the whole
    line; beyond the edge of bi-log-concavity near separation 2.59 it does not succeed."""
    from uai.interval import bimodal_blc_certificate
    from uai.latent_laws import BIMODAL, _bimodal_params
    ok, worst, _ = bimodal_blc_certificate(*BIMODAL)
    assert ok and worst < 0
    assert not bimodal_blc_certificate(*_bimodal_params(0.4, 2.65), max_boxes=20000)[0]


def test_closed_forms_match_high_precision_quadrature():
    """Double-precision closed forms of the box bounds (uai.certify._cdf_noisy) against 30-digit
    quadrature, on the ranges used after outward rounding: 1e-4 <= x <= 1.5, which covers the
    largest compressed mixture-kernel ratio D_i/T^2 of Table 1 (about 1.245), 1e-4 <= ell <= 20
    (and ell = inf), 1e-3 <= beta ell <= 1e3."""
    import mpmath as mp
    from uai.certify import _cdf_noisy
    mp.mp.dps = 30
    rng = np.random.default_rng(11)
    for i in range(80):
        x = 10 ** rng.uniform(-4, np.log10(1.5))
        ell = np.inf if i % 8 == 0 else 10 ** rng.uniform(-4, 1.3)
        be = 10 ** rng.uniform(-3, 3) / (1.0 if np.isinf(ell) else ell)
        b, c, s = rng.uniform(-1.5, 2.5), float(rng.choice([1.0, -1.0])), mp.sqrt(x)
        Z = 1 / mp.mpf(be) if np.isinf(ell) else (1 - mp.e ** (-be * ell)) / be
        f = lambda y: mp.e ** (-be * y) / Z * mp.ncdf((c - (b - y)) / s)     # W = b - Y
        top = mp.inf if np.isinf(ell) else ell
        k = b - c                                                           # kink of the integrand
        pts = [0] + [p for p in (k - 8 * float(s), k, k + 8 * float(s)) if 0 < p < float(top)] + [top]
        ref = float(mp.quad(f, pts))
        val = _cdf_noisy(c, np.array([b]), np.array([ell]), np.array([be]), x)[0]
        assert abs(val - ref) < 1e-11, (x, ell, be, b, c, val, ref)


def test_analytic_rules_input_checks():
    """Caller-supplied ranks that violate the Hoeffding condition give inf, not an unjustified T;
    unknown levels and negative D_min raise."""
    from uai.procedures import certified_rank, noisy_threshold_halfwidth, simple_shrink_halfwidth
    V = np.random.default_rng(1).normal(size=110)
    assert noisy_threshold_halfwidth(V, k=1) == np.inf             # k = 1 < K p_k + 1: no guarantee
    assert simple_shrink_halfwidth(V, .1, k=1) == np.inf
    with pytest.raises(ValueError):
        certified_rank(110, q=0.85)
    with pytest.raises(ValueError):
        simple_shrink_halfwidth(V, -1.0)


def test_exact_reliability_matches_stress_simulation():
    """Proposition 4: the exact latent reliability pr{Bin(K, H(r_q)) <= k - 1} (E40) agrees with
    the simulated reliability of E38 within 3.5 Monte Carlo standard errors, and the usual rank
    falls below 1 - delta for large K while the certified rank does not."""
    import pandas as pd
    ex = pd.read_csv(ROOT / 'results' / 'exact_reliability.csv')
    st = pd.read_csv(ROOT / 'results' / 'stress_summary.csv')
    st = st[st.design == 'homogeneous']
    for K in (110, 1000, 10000):
        for rule, name in (('usual', 'usual_PAC'), ('certified', 'certified'), ('marginal', 'marginal')):
            e = ex[(ex.K == K) & (ex.rule == rule)].reliability.item()
            s = st[(st.K == K) & (st.rule == name)].iloc[0]
            assert abs(e - s.reliability) < 3.5 * max(s.se_rel, 1e-3), (K, rule, e, s.reliability)
    big = ex[ex.K == 1000000].set_index('rule').reliability
    assert big['usual'] < 0.2 and big['marginal'] < 0.01 and big['certified'] > 0.95


def _eh_true(q, k, L):
    """E min(1, q e^{-eps}) for eps = L (V - mu), V with density prop. e^{kv} on [0, 1]."""
    from scipy import integrate
    Z = integrate.quad(lambda v: np.exp(k * v), 0, 1)[0]
    mu = integrate.quad(lambda v: v * np.exp(k * v), 0, 1)[0] / Z
    vc = min(max(mu + np.log(q) / L, 0), 1)
    f = lambda v: min(1, q * np.exp(-L * (v - mu))) * np.exp(k * v) / Z
    return integrate.quad(f, 0, vc)[0] + integrate.quad(f, vc, 1)[0]


def test_noise_class_symmetric_unimodal_closed_form():
    """Theorem 4(i): the closed form of Psi_SU matches a direct maximization over the scale of
    uniform noise, and the two-sided split certificate holds at (0.9018, 0.9) and (0.9, 0.8981)."""
    from scipy.optimize import minimize_scalar
    from uai.noise_classes import psi_su_enclosure, split_certificate_su
    for q in (0.8, 0.9, 0.95):
        c = np.log(q)
        g = lambda r: q * np.sinh(r) / r if r <= -c else (c + r + 1 - q * np.exp(-r)) / (2 * r)
        r = minimize_scalar(lambda lr: -g(np.exp(lr)), bounds=(-8, 3), method='bounded',
                            options={'xatol': 1e-12})
        lo, hi = psi_su_enclosure(q)
        assert lo - 1e-12 <= -r.fun <= hi + 1e-12
    assert split_certificate_su('0.9018', '0.9')[0] and split_certificate_su('0.9', '0.8981')[0]


def test_noise_class_log_concave_enclosures_are_sound():
    """The box enclosures of uai.noise_classes bound E h from above at random points inside the
    boxes (region I against quadrature, regions II-III against the truncated exponential law),
    and the centred exponential point value is the closed form."""
    from scipy import integrate
    from uai.noise_classes import eh_region1, eh_region2, eh_region3, lc_point
    rng = np.random.default_rng(4)
    q = 0.9
    for _ in range(40):
        k0 = rng.uniform(-2, 1.9); k1 = k0 + rng.uniform(1e-3, 0.1)
        L0 = np.exp(rng.uniform(np.log(0.2), np.log(30))); L1 = L0 * np.exp(rng.uniform(1e-3, 0.1))
        ub = float(eh_region1(q, k0, k1, L0, L1).upper())
        for _ in range(3):
            assert _eh_true(q, rng.uniform(k0, k1), rng.uniform(L0, L1)) <= ub + 1e-12
    for _ in range(30):
        t0 = rng.uniform(0.02, 0.45); t1 = t0 + rng.uniform(1e-3, 0.05)
        s0 = np.exp(rng.uniform(np.log(0.01), np.log(5))); s1 = s0 * np.exp(rng.uniform(1e-3, 0.1))
        u2 = float(eh_region2(q, t0, t1, s0, s1).upper())
        u3 = float(eh_region3(q, t0, t1, s0, s1).upper())
        k, s = 1 / rng.uniform(t0, t1), rng.uniform(s0, s1)
        Z = -np.expm1(-k)
        m = 1 - k * np.exp(-k) / Z
        dens = lambda tau: np.exp(-tau) / Z
        right = integrate.quad(lambda tau: min(1, q * np.exp(-s * (tau - m))) * dens(tau), 0, k, limit=200)[0]
        left = integrate.quad(lambda tau: min(1, q * np.exp(-s * (m - tau))) * dens(tau), 0, k, limit=200)[0]
        assert right <= u2 + 1e-12 and left <= u3 + 1e-12
    v = lc_point('0.9', 0.1177686817)
    assert abs(float(v.mid()) - 0.9051755359) < 1e-9


def test_noise_classes_results_file():
    """Every certificate behind Theorem 4 and the levels quoted in the paper holds in
    results/noise_classes.json (E41)."""
    import json
    d = json.loads((ROOT / 'results' / 'noise_classes.json').read_text())
    su, lc = d['symmetric_unimodal'], d['log_concave']
    assert all(r['ok'] for r in su['split']) and all(r['fails'] for r in su['fails'])
    assert {(r['p'], r['q']) for r in su['split']} >= {('0.9018', '0.9'), ('0.9', '0.8981'), ('0.9068', '0.9')}
    assert all(r['ok'] for r in lc['one_sided']) and any(r['target'] == '0.9052' for r in lc['one_sided'])
    assert {(r['p'], r['q']) for r in lc['split'] if r['ok']} >= {('0.9068', '0.9'), ('0.906', '0.9'), ('0.9', '0.8935')}
    lows = {r['q']: r['psi_lo'] for r in lc['exp_lower']}
    assert lows['0.9'] > 0.90517 and lows['0.8942'] > 0.9
    assert 0.90079 < d['gaussian']['0.9']['psi'] <= 0.901
