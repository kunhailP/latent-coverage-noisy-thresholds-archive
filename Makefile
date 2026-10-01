PY ?= python3
PROCS ?= 8

.PHONY: test quick edge figures constants certified simulations table1-extra-laws comparison estimated apipop supplement-extras paper all

test:
	$(PY) -m pytest -q

# seconds to minutes: log-concave constants, centring and transition values, edge constants
quick:
	$(PY) experiments/e03_lc_constants.py
	$(PY) experiments/e32_centering.py
	$(PY) experiments/e35_mean_location.py

# edge constants M_q near the feasibility edge (not in the manuscript; slow)
edge:
	$(PY) experiments/e33_edge.py $(PROCS)

figures:
	$(PY) experiments/fig_transition.py
	$(PY) experiments/fig_paper.py
	$(PY) experiments/fig_mean_location.py

# ball-arithmetic certificates for c_q and C_{p,q} (Corollary 1, Lemma 1)
constants:
	$(PY) experiments/e27_interval_constants.py $(PROCS)
	$(PY) experiments/e37_coverage_transfer.py $(PROCS)

# the exact two-dimensional reduction, the boundary map and certified upper bounds of R (hours)
certified:
	$(PY) experiments/e04_shrink_table.py $(PROCS)
	$(PY) experiments/e16_exact_shrink_table.py $(PROCS)
	$(PY) experiments/e19_boundary_map.py $(PROCS)
	$(PY) experiments/e24_certified_table.py $(PROCS)
	$(PY) experiments/e24_certified_table.py $(PROCS) 5e-4 0.002 0.0095 0.9,0.9036
	$(PY) experiments/e24_certified_table.py $(PROCS) 5e-4 0.002 0.366 0.9068

# Table 1 and the shape-free rule (hours, dominated by per-data-set certificates)
simulations:
	$(PY) experiments/e20_hetldc_synth.py 150 $(PROCS)
	$(PY) experiments/e31_shape_free.py
	$(PY) experiments/e28_level_decomposition.py

# the last three laws of Table 1 (hours; merged into the existing results)
table1-extra-laws:
	$(PY) experiments/e20_hetldc_synth.py 150 $(PROCS) shapes=trunc_exp,bimodal_blc,t3_not_LC
	$(PY) experiments/e31_shape_free.py
	$(PY) experiments/e34_competitors.py 150 $(PROCS) analytic-shapes=trunc_exp,bimodal_blc,t3_not_LC

# LatentCP and deconvolution conformal on the data sets of Table 1 (minutes)
comparison:
	$(PY) experiments/e34_competitors.py 150 $(PROCS)
	$(PY) experiments/e34_competitors.py 150 $(PROCS) analytic-shapes=trunc_exp,bimodal_blc,t3_not_LC
	$(PY) experiments/e36_analytic_reliability.py 2000 $(PROCS)
	$(PY) experiments/e38_stress.py
	$(PY) experiments/e40_exact_reliability.py
	$(PY) experiments/e39_sensitivity.py 2000

# estimated noise variances (Supplementary Material, Section S4)
estimated:
	$(PY) experiments/e29_estimated_scale.py 100 $(PROCS)
	$(PY) experiments/e29_estimated_scale.py 30 $(PROCS) cert
	$(PY) experiments/e30_areawise_variance.py 30 $(PROCS)

# school-district application (downloads the survey package data)
apipop:
	$(PY) experiments/fetch_apipop.py
	$(PY) experiments/e21_hetldc_apipop.py 120 $(PROCS)

# heterogeneous-kernel plug-ins, oracle-matched widths and the certified mean-variance rule
supplement-extras:
	$(PY) experiments/e18_hetero_kernel.py $(PROCS)
	$(PY) experiments/e12_conditional_synth.py 2000 $(PROCS)
	$(PY) experiments/e15_e12_exact.py $(PROCS)
	$(PY) experiments/e25_certified_ldc.py 2000 $(PROCS)

paper:
	cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
	cd paper && pdflatex supplement && bibtex supplement && pdflatex supplement && pdflatex supplement

all: quick edge figures constants certified simulations comparison estimated apipop supplement-extras paper
