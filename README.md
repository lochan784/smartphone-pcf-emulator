# Physics-Informed Bayesian–Conformal Emulator for Smartphone Lifecycle Carbon Estimation

A reproducible research framework for estimating the **Product Carbon Footprint (PCF)** of smartphones using a hybrid approach combining:

- physics-informed lifecycle modeling
- hierarchical Bayesian calibration
- conformal prediction for uncertainty certification
- counterfactual simulation and optimization for decision support

The system is designed for **label-scarce environments** where only a small number of verified lifecycle assessments exist (n = 13 manufacturer-declared PCFs).

The emulator produces **auditable, uncertainty-calibrated PCF estimates** and enables **scenario analysis and sustainability optimization**.

---

# Key Results

Using a holdout set of **13 manufacturer-declared PCFs**:

| Metric | Result |
|------|------|
| MAE (full model, LOOCV) | **5.66 kg CO₂e** |
| R² | **0.53** |
| Conformal coverage | **92.3%** (nominal 90%) |

---

# Core Idea

The architecture integrates four layers:

```
Physics Engine → Bayesian Calibration → Conformal Certification → Decision Support
```

Each layer addresses a key challenge in lifecycle modeling:

| Challenge | Solution |
|---|---|
| sparse PCF labels | literature-informed physics model |
| calibration uncertainty | hierarchical Bayesian inference |
| unreliable uncertainty | conformal prediction |
| actionable insights | counterfactual simulation + optimization |

---

# Repository Structure

```
SMARTPHONE-LCA-EMULATOR
│
├── data
│   ├── raw
│   └── processed
│
├── docs
│   └── ARCHITECTURE.md
│
├── models
│
├── paper
│   └── figures
│
├── results
│
├── scripts
│
├── webapp
│
├── requirements.txt
├── Makefile
└── README.md
```

---

# Important Files

### Data

```
data/processed/catalog_devices.csv
```

**968 devices** with hardware specifications (source: Kaggle). Used for feature engineering and large-scale scenario analysis.

```
data/processed/verified_devices.csv
```

**13 manufacturer-verified PCFs** from Apple, Samsung, and Google Environmental Product Declarations (EPDs). Used as the gold-standard holdout set for evaluation and conformal calibration.

```
data/processed/prior_params.csv
```

Literature-informed priors for lifecycle emission factors.

See `data/README.md` for detailed descriptions and license information.

---

### Models

```
models/bayesian_emulator_posterior.nc
```

Posterior samples from the Bayesian emulator.

```
models/brand_ridge_calibrator.joblib
```

Baseline ridge model.

---

### Results

```
results/paper_tables.tex
results/paper_numbers.json
results/conformal_results.json
```

Outputs used directly in the research paper.

---

# Pipeline Scripts

| Script | Purpose |
|------|------|
| `01_engineer_structured_features.py` | convert raw specs to structured features |
| `02_merge_with_benchmark.py` | align device names and build gold holdout |
| `03_train_brand_calibrator.py` | ridge baseline and diagnostics |
| `04_fit_bayesian_emulator.py` | hierarchical Bayesian calibration |
| `05_conformal_certification.py` | conformal prediction intervals |
| `06_counterfactual_simulation.py` | lifecycle scenario analysis |
| `07_optimization.py` | cost-constrained decarbonization optimization |
| `08_ablation.py` | component ablation study |
| `09_prior_sensitivity.py` | prior robustness analysis |
| `10_final_report.py` | reproducibility checks and paper artifacts |

---

# Installation

Requirements

* Python **3.11** (recommended; this project was developed and tested on Python 3.11)
* macOS or Linux terminal (Windows works with PowerShell using the equivalent activation command)

Create the virtual environment used by the Makefile:

```bash
python3.11 -m venv .venv311
```

Activate the environment.

macOS / Linux:

```bash
source .venv311/bin/activate
```

Windows (PowerShell):

```powershell
.venv311\Scripts\Activate.ps1
```

Upgrade pip and install the project dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Verify the environment by importing the core packages:

```bash
python -c "import pymc, arviz, numpy, pandas, streamlit, IPython, h5py; print('Environment OK')"
```

Expected output:

```
Environment OK
```

Run the full research pipeline (feature engineering → model training → conformal calibration → counterfactual analysis → optimization → paper artifacts):

```bash
make all
```

You can also run individual stages:

```bash
make preprocess
make fit
make conformal
make decision
make paper
```

Notes:

* The Makefile executes scripts using `.venv311/bin/python`, so ensure the `.venv311` environment exists before running `make`.
* Some runs may show warnings related to `pkg_resources` or BLAS backends; these are expected and do not affect results.
* The repository pins `setuptools<81` so that `pkg_resources` remains available for ArviZ/PyMC compatibility.
* All generated artifacts (tables, JSON results, CSV outputs) will appear in the `results/` directory.



# Reproducibility

All results in the paper can be reproduced using the provided Makefile pipeline.

## 1. Setup Environment

```bash
make env
```

This creates the environment `.venv311` and installs all required packages.

## 2. Run the Full Pipeline

```bash
make all
```

This executes the complete workflow:

- Feature engineering and dataset preparation (using `catalog_devices.csv` and `verified_devices.csv`)
- Ridge regression baseline
- Hierarchical Bayesian emulator (NUTS sampling, 4 chains × 3000 draws, 1500 warmup, target_accept=0.95)
- LOOCV Jackknife+ conformal prediction calibration (using absolute residuals; normalized residuals evaluated for robustness)
- Counterfactual lifecycle simulations (propagating 8000 posterior draws through alternative scenarios)
- Cost-constrained decarbonization optimization (five levers: grid decarbonization, recycled materials, display efficiency, semiconductor efficiency, battery chemistry)
- Model ablation and prior sensitivity analysis
- Generation of paper tables and summary outputs

## 3. Output Files

All generated artifacts are written to `results/`:

```
results/paper_tables.tex
results/paper_numbers.json
results/conformal_results.json
results/optimization_results.json
results/pareto_frontier.csv
results/ablation_table.csv
```

These files correspond directly to the figures and tables used in the paper.

## 4. Verify Bayesian Convergence

After training the Bayesian emulator, convergence diagnostics can be checked with:

```bash
make check
```

Expected conditions:

- R-hat < 1.05
- ESS > 200
- Divergences = 0

These match the convergence criteria reported in the paper.

## 5. Quick Test Run (Optional)

```bash
make fast
```

Runs the Bayesian emulator with reduced draws for debugging.

## 6. Regenerate Results Without Re-training

If the posterior model already exists:

```bash
make results
```

Regenerates decision-support outputs and paper tables without refitting the Bayesian model.

## 7. Cleaning Generated Outputs

```bash
make clean
```

Removes generated outputs while preserving models and raw data.

## One-Command Reproduction

For reviewers and reproducibility checks:

```bash
make env
make all
```

---

# Running the Web App

```
pip install streamlit
streamlit run webapp/app.py
```

The app allows interactive PCF prediction and scenario exploration.

---

# Physics Model

The lifecycle PCF is decomposed into additive components:

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

All emission factors are initialized using literature priors.

---

# Why Bayesian?

Only **13 verified PCF labels** exist.

Traditional ML methods (XGBoost, neural networks) would overfit.

Bayesian inference allows:

* informative priors
* uncertainty propagation
* hierarchical structure across brands

---

# Why Conformal Prediction?

Bayesian credible intervals assume the model is correct.

Conformal prediction guarantees **finite-sample coverage** even if the model is misspecified.

We evaluated two nonconformity scores:

- absolute residuals (used in final model)
- normalized residuals (for robustness check)

The final intervals use LOOCV Jackknife+ to maintain exchangeability with our small gold set.

---

# Extending the Project

Possible extensions:

* add additional manufacturer PCFs
* expand lifecycle boundary to include end-of-life recycling
* extend emulator to laptops or tablets
* integrate regional electricity grid scenarios

---

# License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

# Citation

If you use this work, please cite:

```bibtex
@article{lalithya2026smartphonepcf,
  title={Physics-Informed Bayesian–Conformal Emulator for Smartphone Lifecycle Carbon Estimation},
  author={Lochan Lalithya S},
  year={2026}
}
```

---

# Contact

Open a GitHub issue for questions, reproducibility reports, or contributions.
