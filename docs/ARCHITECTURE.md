# Architecture

## Physics-Informed Bayesian–Conformal Emulator for Smartphone Lifecycle Carbon Estimation

---

# System Overview

The system is built as a **four-layer modeling pipeline**.

```
Physics Engine
↓
Bayesian Calibration
↓
Conformal Certification
↓
Decision Support
```

Each layer produces artifacts used by the next stage.

---

# Layer 1 — Physics Engine

Scripts:

```
01_engineer_structured_features.py
02_merge_with_benchmark.py
```

Purpose:

Convert raw smartphone specifications into lifecycle features and compute a baseline PCF estimate.

The model decomposes PCF into additive components.

```
PCF =
battery_ef × battery_kwh
+ display_ef × display_area_cm2
+ material_ef × (estimated_mass_g / 1000)
+ grid_ef × lifetime_years × annual_kwh
+ semiconductor_ef × performance_index
+ assembly_ef
+ transport_ef
```

Inputs:

- Raw data from `data/raw/` (Kaggle device specs, EPD PDFs)
- `data/processed/prior_params.csv` (literature priors)

Outputs:

```
data/processed/catalog_devices.csv       # 968 devices with engineered features
data/processed/verified_devices.csv      # 13 manufacturer-verified PCFs (holdout set)
```

The **verified_devices.csv** dataset contains the gold‑standard PCFs used for evaluation and conformal calibration.

All processed data files include a README in `data/README.md` with detailed descriptions and license information.

---

# Layer 2 — Bayesian Calibration

Scripts:

```
03_train_brand_calibrator.py
04_fit_bayesian_emulator.py
```

Purpose:

Calibrate physics-based predictions against manufacturer PCFs while estimating uncertainty.

### Baseline model

Script 03 trains a ridge regression model for comparison.

### Hierarchical Bayesian emulator

Script 04 fits the main model.

Mean function:

```
μ_i =
PCF_physics(θ)
+ β₀
+ b_brand[brand_i]
+ φᵀ s_i
```

The residual term φᵀ s_i uses two normalized features:
- performance index (P̃_idx,i)
- camera megapixels (Cam̃_i)

Brand effects use **non-centred parameterization**:

```
b_brand = z_brand × τ_brand
z_brand ~ N(0,1)
τ_brand ~ HalfNormal
```

Sampling method:

```
NUTS
4 chains
1500 warmup
3000 draws
target_accept = 0.95
```

Convergence checks:

```
R-hat < 1.05
ESS > 200
0 divergences
```

Outputs:

```
models/bayesian_emulator_posterior.nc
models/bayesian_emulator_summary.json
```

---

# Layer 3 — Conformal Certification

Script:

```
05_conformal_certification.py
```

Purpose:

Convert model predictions into prediction intervals with guaranteed coverage.

Two nonconformity scores were evaluated:

1️⃣ **Absolute residuals** (used in final model)

```
s_i = |y_i − ŷ_i|
```

2️⃣ **Normalized residuals** (robustness test)

```
s_i = |y_i − ŷ_i| / σ̂_i
```

### LOOCV Jackknife+ (reported)

Each device in `verified_devices.csv` is calibrated using the remaining 12 gold devices.

Prediction interval:

```
[ŷ_i − q_i , ŷ_i + q_i]
```

Outputs:

```
results/conformal_results.json
results/conformal_score_comparison.csv
```

---

# Layer 4 — Decision Support

Scripts:

```
06_counterfactual_simulation.py
07_optimization.py
08_ablation.py
09_prior_sensitivity.py
10_final_report.py
```

---

## Counterfactual simulation

Posterior samples (N = 8000) are propagated through alternative lifecycle scenarios.

Example scenarios:

| Scenario | Change |
|---|---|
| Grid decarbonization | grid_ef × 0.70 |
| Next-generation battery | battery_ef × 0.75 |
| Extended lifetime | lifetime_years = 5 |

---

## Optimization

Script 07 selects decarbonization levers under a budget constraint.

Five levers considered:

| Lever | Description |
|---|---|
| Grid decarbonization | reduce grid emission factor |
| Recycled materials | reduce material emission factor |
| Display efficiency | reduce display emission factor |
| Semiconductor efficiency | reduce semiconductor emission factor |
| Battery chemistry | reduce battery emission factor |

Optimization problem:

```
maximize   Σ reduction_i × x_i
subject to Σ cost_i × x_i ≤ budget
```

Binary decisions:

```
x_i ∈ {0,1}
```

Solved using **dynamic programming knapsack**.

Advantages:
- exact solution
- no external solver dependency
- reproducible across environments

---

## Ablation Study

Script 08 sequentially removes components to measure impact.

Result (matches paper):

| Model Variant | MAE |
|---|---|
| Raw physics | 24.18 |
| + posterior θ | 22.75 |
| + intercept | 7.08 |
| + brand effects | 5.72 |
| Full model | 5.66 |

The intercept provides the largest error correction.

---

## Prior Sensitivity

Script 09 tests robustness by perturbing prior standard deviations.

```
±50%
```

Posterior shifts are recorded in:

```
results/sensitivity_summary.json
```

---

# Final Reporting

Script:

```
10_final_report.py
```

Produces:

```
results/paper_tables.tex
results/paper_numbers.json
results/reproducibility_manifest.json
```

---

# Reproducibility Gates

The pipeline automatically checks:

1. MCMC convergence
2. conformal coverage
3. MAE improvement
4. counterfactual sign correctness
5. optimization improvement

If any check fails, the pipeline flags the run.

---

# Data Flow

```
raw specs (Kaggle) + EPDs
↓
01_engineer_structured_features.py
↓
catalog_devices.csv
↓
02_merge_with_benchmark.py
↓
verified_devices.csv (holdout)
↓
physics baseline
↓
Bayesian calibration
↓
conformal intervals
↓
decision support
```

---

# Design Decisions

| Decision | Alternative | Reason |
|---|---|---|
| Bayesian calibration | XGBoost | small n |
| Non‑centred hierarchy | centred | avoids funnel divergences |
| LOOCV Jackknife+ conformal | split conformal | maintains exchangeability |
| DP knapsack | CVXPY solver | dependency‑free |
| NUTS sampler | Metropolis | better scaling |

---

# Extension Points

Future improvements could include:

- integrating recycling and end-of-life processes
- expanding to additional electronics categories
- linking the emulator to real-time grid intensity data
- adding a cloud API for automated PCF estimation

See `data/README.md` for details on the datasets and their licenses.