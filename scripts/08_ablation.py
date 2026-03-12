#!/usr/bin/env python3
"""
Phase 8 — Ablation Study
Removes one component at a time. Produces Table II for the paper.
"""

import json, os, warnings
import numpy as np
import pandas as pd
import arviz as az
import joblib

warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)

gold = pd.read_csv("data/processed/gold_holdout.csv")
train = pd.read_csv("data/processed/train_modeling.csv")
trace = az.from_netcdf("models/bayesian_emulator_posterior.nc")
scaler = joblib.load("models/resid_feature_scaler.joblib")
brand_map = joblib.load("models/brand_to_idx.joblib")

TARGET = "pcf_kg_co2e_y"
BRAND_COL = "brand"
RESID_COLS = [c for c in ["performance_index", "total_camera_mp"] if c in gold.columns]

y_gold = gold[TARGET].values.astype(float)
model_col = "model_y" if "model_y" in gold.columns else "model_x"

post = trace.posterior
theta = {
    v: float(post[v].mean())
    for v in [
        "battery_ef",
        "display_ef",
        "material_ef",
        "semiconductor_intensity",
        "assembly_ef",
        "transport_ef",
        "lifetime_years",
        "grid_ef",
    ]
}
beta0 = float(post["beta0"].mean())
b_brand = post["b_brand"].mean(dim=("chain", "draw")).values
phi = post["phi"].mean(dim=("chain", "draw")).values

bat_g = gold["battery_kwh"].values.astype(float)
disp_g = gold["display_area_cm2"].values.astype(float)
mass_g = (gold["estimated_mass_g"].values / 1000.0).astype(float)
perf_g = gold["performance_index"].values.astype(float)
ann_g = gold["annual_kwh"].values.astype(float)
bidx_g = np.array([brand_map.get(str(b), 0) for b in gold[BRAND_COL]])
X_gold = scaler.transform(gold[RESID_COLS].fillna(0).values.astype(float))


def metrics(pred, true):
    res = pred - true
    mae = float(np.abs(res).mean())
    rmse = float(np.sqrt((res**2).mean()))
    bias = float(res.mean())
    ss_res = float((res**2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    r2 = float(1 - ss_res / ss_tot)
    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "Bias": round(bias, 3),
        "R2": round(r2, 3),
    }


def physics_only():
    return (
        bat_g * theta["battery_ef"]
        + disp_g * theta["display_ef"]
        + mass_g * theta["material_ef"]
        + ann_g * theta["grid_ef"] * theta["lifetime_years"]
        + perf_g * theta["semiconductor_intensity"]
        + theta["assembly_ef"]
        + theta["transport_ef"]
    )


def full_pred(intercept=True, brand=True, resid=True):
    pred = physics_only()
    if intercept:
        pred = pred + beta0
    if brand:
        bc = np.array([b_brand[i] if 0 <= i < len(b_brand) else 0.0 for i in bidx_g])
        pred = pred + bc
    if resid:
        pred = pred + X_gold @ phi
    return pred


def flat_physics():
    """Raw literature EFs — no posterior, no calibration."""
    return (
        bat_g * 120.0
        + disp_g * 0.03
        + mass_g * 18.0
        + ann_g * 0.45 * 3.0
        + perf_g * 0.80
        + 3.5
        + 2.0
    )


print("=" * 66)
print("PHASE 8 — ABLATION STUDY")
print("=" * 66)

rows = []

# 1. Flat physics (literature priors, no Bayesian calibration)
m = metrics(flat_physics(), y_gold)
rows.append(
    {
        "model": "Raw Physics (literature EFs, no calibration)",
        "components": "Physics only",
        **m,
    }
)
raw_mae = m["MAE"]
print(f"\n1. Raw Physics                         MAE={m['MAE']:.3f}  R²={m['R2']:.3f}")

# 2. Physics + Bayesian posterior theta (no intercept/brand/resid)
m = metrics(physics_only(), y_gold)
rows.append(
    {
        "model": "Physics + Posterior θ (no intercept)",
        "components": "Physics + θ_posterior",
        **m,
    }
)
print(
    f"2. + Posterior θ                       MAE={m['MAE']:.3f}  R²={m['R2']:.3f}  "
    f"ΔMAE={m['MAE']-raw_mae:+.3f}"
)

# 3. + Intercept β₀
m = metrics(full_pred(intercept=True, brand=False, resid=False), y_gold)
rows.append(
    {"model": "Physics + θ + Intercept β₀", "components": "Physics + θ + β₀", **m}
)
print(
    f"3. + Intercept β₀                      MAE={m['MAE']:.3f}  R²={m['R2']:.3f}  "
    f"ΔMAE={m['MAE']-raw_mae:+.3f}"
)

# 4. + Brand effects
m = metrics(full_pred(intercept=True, brand=True, resid=False), y_gold)
rows.append(
    {
        "model": "Physics + θ + β₀ + Brand effects",
        "components": "Physics + θ + β₀ + b_brand",
        **m,
    }
)
print(
    f"4. + Brand effects                     MAE={m['MAE']:.3f}  R²={m['R2']:.3f}  "
    f"ΔMAE={m['MAE']-raw_mae:+.3f}"
)

# 5. Full model
m_full = metrics(full_pred(intercept=True, brand=True, resid=True), y_gold)
rows.append(
    {
        "model": "Full Bayesian (θ + β₀ + Brand + h(s;φ))",
        "components": "All components",
        **m_full,
    }
)
print(
    f"5. Full Bayesian                       MAE={m_full['MAE']:.3f}  R²={m_full['R2']:.3f}  "
    f"ΔMAE={m_full['MAE']-raw_mae:+.3f}"
)

# 6. Ablate: remove brand
m = metrics(full_pred(intercept=True, brand=False, resid=True), y_gold)
rows.append(
    {"model": "Full − Brand effects", "components": "Physics + θ + β₀ + h(s;φ)", **m}
)
print(
    f"6. Full − Brand effects                MAE={m['MAE']:.3f}  R²={m['R2']:.3f}  "
    f"ΔMAE vs full={m['MAE']-m_full['MAE']:+.3f}"
)

# 7. Ablate: remove residual h
m = metrics(full_pred(intercept=True, brand=True, resid=False), y_gold)
rows.append(
    {"model": "Full − Residual h(s;φ)", "components": "Physics + θ + β₀ + Brand", **m}
)
print(
    f"7. Full − Residual h(s;φ)              MAE={m['MAE']:.3f}  R²={m['R2']:.3f}  "
    f"ΔMAE vs full={m['MAE']-m_full['MAE']:+.3f}"
)

# 8. Ridge LOOCV baseline (historical)
rows.append(
    {
        "model": "Ridge Calibration (LOOCV baseline)",
        "components": "sklearn Ridge on physics features",
        "MAE": 5.010,
        "RMSE": 6.830,
        "Bias": -4.860,
        "R2": 0.554,
    }
)
print(f"8. Ridge LOOCV (historical)            MAE=5.010  R²=0.554")

# ── Conformal alpha sensitivity ────────────────────────────────
print("\n" + "─" * 66)
print("CONFORMAL SENSITIVITY — alpha ∈ [0.05, 0.20]")
print("─" * 66)

conf_df = pd.read_csv("data/processed/conformal_predictions_gold.csv")
pred_conf = conf_df["conformal_pred"].values
scores = np.abs(y_gold - pred_conf)
n = len(scores)

alpha_rows = []
print(f"  {'Alpha':>6} {'Target':>8} {'Coverage':>10} {'Avg Width':>11}")
print("  " + "─" * 40)
for alpha in [0.05, 0.10, 0.15, 0.20]:
    covs, widths = [], []
    for i in range(n):
        cal = np.delete(scores, i)
        nc = len(cal)
        ql = min(np.ceil((1 - alpha) * (nc + 1)) / nc, 1.0)
        qi = float(np.quantile(cal, ql))
        covs.append(pred_conf[i] - qi <= y_gold[i] <= pred_conf[i] + qi)
        widths.append(2 * qi)
    cov_emp = float(np.mean(covs))
    avg_w = float(np.mean(widths))
    flag = "✅" if cov_emp >= (1 - alpha - 0.05) else "❌"
    print(f"  {alpha:>6.2f} {1-alpha:>8.2f} {cov_emp:>10.3f} {flag}  {avg_w:>9.2f} kg")
    alpha_rows.append(
        {
            "alpha": alpha,
            "target": 1 - alpha,
            "coverage": round(cov_emp, 3),
            "avg_width_kg": round(avg_w, 3),
        }
    )

# ── Table II print ─────────────────────────────────────────────
print("\n" + "=" * 66)
print("TABLE II — ABLATION (ready for paper)")
print("=" * 66)
print(f"  {'Model':<46} {'MAE':>6} {'R²':>7}")
print("  " + "─" * 60)
for r in rows:
    print(f"  {r['model']:<46} {r['MAE']:>6.3f} {r['R2']:>7.3f}")

# ── Save ──────────────────────────────────────────────────────
abl_df = pd.DataFrame(rows)
abl_df.to_csv("results/ablation_table.csv", index=False)

with open("results/ablation_results.json", "w") as f:
    json.dump(
        {"ablation": rows, "conformal_alpha_sensitivity": alpha_rows}, f, indent=2
    )

print(f"\n✅ results/ablation_table.csv")
print(f"✅ results/ablation_results.json")
print(f"\nNext: python scripts/09_generate_paper_tables.py")
