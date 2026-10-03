# Latent coverage of noisy calibration thresholds

Code, results and manuscript for the paper *Latent coverage of noisy calibration thresholds* by Kun Woo Park (in preparation for *Biometrika*). Every number in the paper and its Supplementary Material can be regenerated from this repository.

## The question

A prediction interval for a latent quantity, such as the true mean of an area that a survey did not sample, is often calibrated on noisy proxies. The latent residual `W` is observed only through `V = W + e`, with noise `e` independent of `W`. The threshold `t` at which `|V|` has coverage `p` is observable. What latent coverage `P(|W| <= t)` does it guarantee, uniformly over a class of latent laws and of noise laws of every scale? Calibration on noisy scores is safe when both laws are symmetric and unimodal (Anderson), and without a shape assumption a noisy level `1 - a` gives only `1 - 2a` (Guille-Escuret and Ndiaye).

**In one sentence:** bi-log-concavity of the latent law closes almost all of that gap, at a price computed exactly or within certified bounds for each class of noise laws; neither it nor the noise condition can be weakened to unimodality; and with unknown, unit-specific noise the price is a raise of the conformal rank by a few ranks, whose minimal value is identified.

## Main results (numbering of the revised manuscript)

| latent law | noise | latent coverage at noisy 0.9 | noisy level for latent 0.9 | smallest rank, `K = 10^3` / `10^4` |
|---|---|---|---|---|
| symmetric unimodal | symmetric unimodal | >= 0.9 (Anderson) | 0.9 | 916 / 9050 |
| bi-log-concave | Gaussian | 0.8991 | (0.90079, 0.901] | 917 / 9058-9060 |
| bi-log-concave | symmetric unimodal | 0.8981 (sharp: 0.898143...) | 0.9017873... (exact) | 918 / 9068 |
| bi-log-concave | mean-zero log-concave | 0.8935 | (0.90517, 0.906] | 921-922 / 9101-9109 |
| bi-log-concave | mean-zero unimodal | none | 1 | - |
| unimodal, mean zero | median zero (Gaussian, symmetric unimodal) | 0.8 | 0.95 | 962 / 9537 |
| unimodal, mean zero | mean-zero log-concave | 1 - e/10 = 0.72817... | 1 - 0.1/e | 974 / 9664 |

1. **Coverage transfer** (Proposition 1, Theorem 1). For a noise class closed under scaling and negation, one end of the interval transfers through `Psi_E(q) = sup E min(1, q e^{-eps})`; exponential tails are extremal. Symmetric unimodal noise: closed form, `Psi` strictly convex, so the two-sided level equals `Psi(q)` exactly (Supplementary proposition, analytic). Mean-zero log-concave noise: localization plus a ball-arithmetic branch and bound; the one-sided supremum lies in [0.90517, 0.9052], the lower end being the value at the centred exponential law, whose extremality is conjectured, not proved. Mean-zero unimodal noise: nothing can be guaranteed. `src/uai/noise_classes.py`, `experiments/e41_noise_classes.py`.
2. **Unimodality cannot replace bi-log-concavity** (Proposition 2). With `beta_E = inf P(eps >= 0)` over the noise class: for `beta_E > 0` and `1 - beta_E < p < 1`, every latent law has latent coverage at least `1 - (1 - p)/beta_E`, and unimodal mean-zero latent laws attain it: `2p - 1` for median-zero noise, `1 - e(1 - p)` for mean-zero log-concave noise, nothing for mean-zero unimodal noise, where `beta_E = 0` and latent coverage can be made arbitrarily small. This shows that unimodality cannot replace bi-log-concavity, not that bi-log-concavity is necessary. Noisy coverage is computed in closed form. `experiments/e42_unimodal_boundary.py`.
3. **Smallest valid rank** (Theorem 2). With independent noise whose laws (in the class) and scales differ between units and are unknown, the noisy threshold at rank `k` has latent reliability at least `P{Bin(K, p_E) <= k - 1}`, and ranks below the one given by `Psi_E(q)` fail for some log-concave law; the smallest rank is exact for symmetric unimodal noise and bracketed for Gaussian and log-concave noise. With the same information (no rule told the noise law or variances), over the settings each rule covers, the shape-constrained ranks cost 0.3-2.9% of width over the usual rank at `K = 1000`, the shape-free ranks 20-54% (`experiments/e43_same_information.py`). At `q = 0.9`, `delta = 0.05` a valid rank exists for `K >= 29` (Gaussian, symmetric unimodal), `K >= 31` (log-concave noise), `K >= 59` (no latent shape). `certified_rank` returns the Gaussian rank.
4. **Without the raise** (Proposition 3). For some log-concave laws the latent reliability of the usual and the marginal rank tends to zero as `K` grows: 0.918 at `K = 10^4` and 0.151 at `K = 10^6` against 0.958 and 0.989 for the raised rank, while the latent coverage tends to 0.89918 (`e40_exact_reliability.py`).
5. **Gaussian noise: the radius and the location of the mean** (Proposition 4, Lemma 1, Theorems 3-5, Corollary 1). The sharp radius `t R_{p,q}(D/t^2)` is a supremum over point masses and log-affine laws on a segment. Without a restriction on the mean the widening is at most `c_q D^{1/2}` (`c_0.9` in `[0.0190618, 0.0190619]`), sharp as `D -> 0`; with the mean at distance `d` from the boundary it is at most `D/(2d)`, an order attained by mean-zero asymmetric laws; the order `D^{1/2}` needs the mean within `O(D^{1/2})` of the boundary, with limiting coefficient `L_q(kappa)`. These are worst-case statements. A slack of `10^{-3}` at `q = 0.9` removes the widening at every noise level.
6. **Known variances** (Proposition 5). The average-kernel radius `T R^mix` shortens the noisy threshold by about 10% in Table 2; the implementation returns a numerically computed upper bound (see Precision). A known lower bound `D_min` gives the analytic shrinkage `T - 0.114 D_min^{1/2}`.
7. **A shifted new residual** (Supplement). For a log-concave latent law a shift of `rho` latent standard deviations costs at most `rho` in coverage, and `q(1 - e^{-rho})` for some laws, so the linear order cannot be improved; the shared latent law, not the noise, is the substantive assumption.

`e33_edge.py` studies a regime that is not in the manuscript: near the feasibility edge `x_p`, `R_{p,q}(x) ~ M_q (x_p - x)^{1/2}`, with `M_q = sup Q_q(|W|)/E(W²)^{1/2}` over log-concave laws (`M_0.9 = 1.8532`, computed, not certified).

## Layout

```
paper/          main.tex and supplement.tex (Biometrika class file v1.5), figures, references, PDFs
src/uai/        library: closed forms and constants (extremal), certificates (certify, interval),
                procedures, estimated-variance rules, closed-form latent laws
experiments/    one script per experiment, numbered as in the Supplementary Material
results/        outputs of the experiments (CSV/JSON) used in the paper
tests/          fast checks of constants, counterexamples and bounds
```

## Installation

Python 3.10 or later.

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test,figures]"
make test
```

`python-flint` provides the Arb ball arithmetic. Scripts take the number of worker processes as an argument (`PROCS`, default 8). Set `OMP_NUM_THREADS=1` when running many workers.

## Reproducing the paper

| Item | Command | Output |
|---|---|---|
| Constants `c_q`, `C_{p,q}`, slack intervals (Lemma 1, Corollary 1) | `make constants` | `results/interval_constants.json` |
| Coverage form of Corollary 1 (0.9 → 0.8991) | `make constants` | `results/coverage_transfer.json` |
| Non-Gaussian noise classes (Proposition 1, Theorem 1) | `make constants` | `results/noise_classes.json` |
| Centring and transition values, Figure 1 | `make quick figures` | `results/centering.json`, `paper/fig_transition.pdf` |
| Edge constants `M_q` (not in the manuscript) | `make edge` | `results/edge.json` |
| Radius correction by location of the mean (Supplement S2) | `make quick figures` | `results/mean_location.csv`, `paper/fig_mean_location.pdf` |
| Certified `R_{p,q}`, boundary map (Supplement S2–S3) | `make certified` | `results/certified_R.csv`, `results/boundary_map.csv` |
| Table 2: computed radius, shape-free rule | `make simulations` | `results/hetldc_synth_summary.csv`, `results/shape_free_summary.csv` |
| Table 2: noisy and shrunk thresholds; comparison with LatentCP and deconvolution | `make comparison` | `results/competitors_summary.csv` |
| Reliability of the analytic rules with 2000 data sets per law (Supplement S6) | `make comparison` | `results/analytic_reliability_summary.csv` |
| Stress test at the extremal law, K up to 10^4 (Supplement S6) | `make comparison` | `results/stress_summary.csv` |
| Exact latent reliability without the slack, K up to 10^6 (Proposition 3, Supplement S6) | `make comparison` | `results/exact_reliability.csv`, `paper/fig_reliability.pdf` |
| Sensitivity to a shared latent law (Supplement S6) | `make comparison` | `results/sensitivity_summary.csv`, `results/sensitivity_stress.csv` |
| Table 2, last three laws only (truncated exponential, bimodal bi-log-concave, `t_3`) | `make table1-extra-laws` | merged into the files above |
| Estimated variances (Supplement S4) | `make estimated` | `results/estimated_scale_summary.csv`, `results/areawise_variance_summary.csv` |
| School-district application | `make apipop` | `results/hetldc_apipop_summary.csv` |
| Plug-in counterexamples, oracle-matched widths | `make supplement-extras` | `results/hetero_kernel.csv`, `results/conditional_synth_exact_*.csv` |
| Unimodal latent laws (Proposition 2) | `python experiments/e42_unimodal_boundary.py` | printed |
| Rules with the same information (Supplement S6) | `cd experiments && python e43_same_information.py` | `results/same_information.csv` |
| Manuscript and supplement | `make paper` | `paper/main.pdf`, `paper/supplement.pdf` |

`make certified`, `make simulations` and `make table1-extra-laws` take hours on a few cores (about 2.5 minutes of one core per data set of Table 2 for the branch and bound of the computed radius; the estimated-variance rerun of Supplement S4 needs about 3.6 minutes per data set), `make constants` about 15 minutes on 4 cores and `make edge` more than 25 minutes; the other targets take minutes.

## Precision

Three levels of support are distinguished throughout.

| What | Status |
|---|---|
| Theorem 2 and the shrunk threshold (`noisy_threshold_halfwidth`, `simple_shrink_halfwidth`, `certified_rank`) | Proofs plus constants `c_q`, `C_{p,q}` and slack intervals certified in Arb ball arithmetic with outward rounding (`uai.interval`, `make constants`) |
| Bi-log-concavity of the bimodal law of Table 2 (`bimodal_blc_certificate`) | Proof (Supplementary Lemma S1) plus a ball-arithmetic check on two compact intervals |
| The computed radius `T R^mix` with the exact `R^mix` | A valid procedure for log-concave latent laws (Propositions 4 and 5) |
| Values returned by `hetldc_certified` and `CertifiedShrinkTable` | Numerically computed upper bounds of `R^mix` and `R_{p,q}`: a branch and bound whose closed forms are evaluated in double precision, boxes cleared at a margin of `10^{-9}`; not interval arithmetic |

`hetldc_halfwidth` and `ShrinkTable` return grid values, which are lower bounds of the sharp radius and not valid radii. `results/r_table.json` (E04, differential evolution with quadrature) is a legacy table: at `x = 0.001` its values exceed the bound of Theorem 1 because the quadrature is inaccurate there, and the paper does not use it; likewise the column `R_DE` of `results/r_exact_p0.9_q0.9.csv` at `x = 0.001`. The transition coefficients, the centred-law coefficients and the edge constants `M_q` are computed in double precision and are not certified.

## Citation

```
@unpublished{park2026latent,
  author = {Park, Kun Woo},
  title  = {Latent coverage of noisy calibration thresholds},
  year   = {2026},
  note   = {Manuscript}
}
```
