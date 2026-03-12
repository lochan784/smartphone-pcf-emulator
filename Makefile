# Makefile — aligned with paper methodology
.PHONY: help all preprocess baseline fit conformal decision paper results env clean fast check

PYTHON ?= .venv311/bin/python
PYTHON3 ?= python3.11
SCRIPTS := scripts

# Sampling parameters (from paper)
FIT_FLAGS ?= --chains 4 --draws 2000 --target_accept 0.95 --seed 42
FAST_FLAGS ?= --chains 2 --draws 500 --target_accept 0.9 --seed 42

# -----------------------------
# Full pipeline (reproduce paper)
# -----------------------------

all: preprocess baseline fit conformal decision paper
	@echo "Full pipeline complete. See results/ for outputs."

# -----------------------------
# Feature engineering + dataset merge
# -----------------------------

preprocess:
	$(PYTHON) $(SCRIPTS)/01_engineer_structured_features.py
	$(PYTHON) $(SCRIPTS)/02_merge_with_benchmark.py

# -----------------------------
# Ridge baseline
# -----------------------------

baseline:
	$(PYTHON) $(SCRIPTS)/03_train_brand_calibrator.py

# -----------------------------
# Bayesian hierarchical emulator
# -----------------------------

fit:
	$(PYTHON) $(SCRIPTS)/04_fit_bayesian_emulator.py $(FIT_FLAGS)

# -----------------------------
# Conformal certification
# -----------------------------

conformal:
	$(PYTHON) $(SCRIPTS)/05_conformal_certification.py

# -----------------------------
# Decision support layer
# -----------------------------

decision:
	$(PYTHON) $(SCRIPTS)/06_counterfactual_simulation.py
	$(PYTHON) $(SCRIPTS)/07_optimization.py

# -----------------------------
# Paper outputs
# -----------------------------

paper:
	$(PYTHON) $(SCRIPTS)/08_ablation.py
	$(PYTHON) $(SCRIPTS)/09_prior_sensitivity.py
	$(PYTHON) $(SCRIPTS)/generate_paper_tables.py
	$(PYTHON) $(SCRIPTS)/10_final_report.py

# -----------------------------
# Results only (skip training)
# -----------------------------

results:
	$(PYTHON) $(SCRIPTS)/06_counterfactual_simulation.py
	$(PYTHON) $(SCRIPTS)/07_optimization.py
	$(PYTHON) $(SCRIPTS)/08_ablation.py
	$(PYTHON) $(SCRIPTS)/09_prior_sensitivity.py
	$(PYTHON) $(SCRIPTS)/generate_paper_tables.py
	$(PYTHON) $(SCRIPTS)/10_final_report.py

# -----------------------------
# Quick convergence check
# -----------------------------

check:
	$(PYTHON) -c "\
import json; \
d = json.load(open('models/bayesian_emulator_summary.json'))['diagnostics']; \
print(f\"R-hat: {d['rhat_max']}  ESS: {d['ess_min']}  Divergences: {d['divergences']}\"); \
print('PASS' if d['rhat_max'] < 1.05 and d['ess_min'] > 200 and d['divergences'] == 0 else 'FAIL')"

# -----------------------------
# Fast test run
# -----------------------------

fast:
	$(PYTHON) $(SCRIPTS)/04_fit_bayesian_emulator.py $(FAST_FLAGS)

# -----------------------------
# Environment setup
# -----------------------------

env:
	$(PYTHON3) -m venv .venv311
	.venv311/bin/pip install --upgrade pip
	.venv311/bin/pip install -r requirements.txt

# -----------------------------
# Clean generated outputs
# -----------------------------

clean:
	rm -rf results/*.json results/*.csv results/*.tex
	rm -rf data/processed/posterior_predictions_gold.csv
	rm -rf data/processed/conformal_predictions_gold.csv
	rm -rf data/processed/counterfactual_per_device.csv
	find . -name "__pycache__" ! -path "./.venv*" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete. Models and raw data preserved."

# -----------------------------
# Help
# -----------------------------

help:
	@echo "Targets:"
	@echo " make env        - create environment"
	@echo " make all        - reproduce full paper pipeline"
	@echo " make preprocess - feature engineering"
	@echo " make fit        - Bayesian emulator training"
	@echo " make conformal  - conformal calibration"
	@echo " make decision   - counterfactual + optimization"
	@echo " make paper      - generate tables and figures"
	@echo " make results    - regenerate outputs without refitting"
	@echo " make fast       - quick test fit"
	@echo " make check      - posterior convergence diagnostics"
	@echo " make clean      - remove generated outputs"