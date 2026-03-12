#!/usr/bin/env python3
"""
Phase 5 — Conformal Certification (LOOCV-Conformal over gold)
Correct approach for small n: use leave-one-out nonconformity scores
on the gold set itself, avoiding synthetic-data exchangeability violation.
"""

import json, os, warnings
import numpy as np
import pandas as pd
import arviz as az
import joblib

warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)

ALPHA = 0.10  # target 90% coverage

# ── Load ──────────────────────────────────────────────────────
train = pd.read_csv("data/processed/train_modeling.csv")
gold = pd.read_csv("data/processed/gold_holdout.csv")
trace = az.from_netcdf("models/bayesian_emulator_posterior.nc")
scaler = joblib.load("models/resid_feature_scaler.joblib")
brand_map = joblib.load("models/brand_to_idx.joblib")

TARGET = "pcf_kg_co2e_y"
BRAND_COL = "brand"
RESID_COLS = [c for c in ["performance_index", "total_camera_mp"] if c in gold.columns]

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
sigma = float(post["sigma"].mean())


def predict_df(df):
    C = (
        df["battery_kwh"].values * theta["battery_ef"]
        + df["display_area_cm2"].values * theta["display_ef"]
        + (df["estimated_mass_g"].values / 1000) * theta["material_ef"]
        + df["annual_kwh"].values * theta["grid_ef"] * theta["lifetime_years"]
        + df["performance_index"].values * theta["semiconductor_intensity"]
        + theta["assembly_ef"]
        + theta["transport_ef"]
    )
    bidx = np.array([brand_map.get(str(b), 0) for b in df[BRAND_COL]])
    bc = np.array([b_brand[i] if 0 <= i < len(b_brand) else 0.0 for i in bidx])
    X = scaler.transform(df[RESID_COLS].fillna(0).values.astype(float))
    return C + beta0 + bc + (X @ phi)


y_gold = gold[TARGET].values.astype(float)
gold_preds = predict_df(gold)
n_gold = len(gold)

print("=" * 60)
print("PHASE 5 — CONFORMAL CERTIFICATION")
print("=" * 60)

# ── Method A: Split-conformal on synthetic train (for reference) ──
SYNTH = "pcf_kg_co2e_x" if "pcf_kg_co2e_x" in train.columns else "pcf_kg_co2e"
np.random.seed(42)
calib_idx = np.random.choice(len(train), size=int(0.20 * len(train)), replace=False)
calib_df = train.iloc[calib_idx].reset_index(drop=True)
calib_obs = calib_df[SYNTH].values.astype(float)
calib_pred = predict_df(calib_df)
scores_syn = np.abs(calib_obs - calib_pred)
n_calib = len(scores_syn)
q_level = min(np.ceil((1 - ALPHA) * (n_calib + 1)) / n_calib, 1.0)
q_split = float(np.quantile(scores_syn, q_level))

conf_lo_split = gold_preds - q_split
conf_hi_split = gold_preds + q_split
cov_split = float(np.mean((y_gold >= conf_lo_split) & (y_gold <= conf_hi_split)))

print(f"\nMethod A — Split-conformal (synthetic calibration)")
print(f"  n_calib : {n_calib}  |  q = {q_split:.2f} kg CO₂e")
print(
    f"  Coverage: {cov_split:.3f}  (note: synthetic calibration breaks exchangeability)"
)

# ── Method B: LOOCV-conformal over gold (exchangeability-valid) ──
# For each gold device i, score = |y_i - pred_i| using full-model pred
# (posterior mean is fixed — no refit per fold at this stage)
# This is the "jackknife+" / cross-conformal approach for small n
loocv_scores = np.abs(y_gold - gold_preds)

# Conformal interval for each point i uses all OTHER scores as calibration
conf_lo_loocv = np.zeros(n_gold)
conf_hi_loocv = np.zeros(n_gold)
for i in range(n_gold):
    calib_scores_i = np.delete(loocv_scores, i)  # leave-one-out scores
    n_c = len(calib_scores_i)
    ql = min(np.ceil((1 - ALPHA) * (n_c + 1)) / n_c, 1.0)
    q_i = float(np.quantile(calib_scores_i, ql))
    conf_lo_loocv[i] = gold_preds[i] - q_i
    conf_hi_loocv[i] = gold_preds[i] + q_i

covered_loocv = (y_gold >= conf_lo_loocv) & (y_gold <= conf_hi_loocv)
cov_loocv = float(covered_loocv.mean())
avg_width = float((conf_hi_loocv - conf_lo_loocv).mean())
passed = cov_loocv >= (1 - ALPHA - 0.05)

print(f"\nMethod B — LOOCV-conformal over gold (exchangeability-valid)")
print(f"  n_gold  : {n_gold}")
print(
    f"  Coverage: {cov_loocv:.3f}  "
    f"({'✅ PASSED' if passed else '❌'} target >= {1-ALPHA-0.05:.2f})"
)
print(f"  Avg interval width: {avg_width:.2f} kg CO₂e")

model_col = "model_y" if "model_y" in gold.columns else "model_x"
print(
    f"\n  {'Model':<24} {'Declared':>9} {'Pred':>8} "
    f"{'Lo90':>8} {'Hi90':>8} {'CI':>5}"
)
print("  " + "─" * 62)
for i in range(n_gold):
    flag = "✅" if covered_loocv[i] else "❌"
    print(
        f"  {str(gold[model_col].values[i])[:24]:<24} "
        f"{y_gold[i]:>9.1f} {gold_preds[i]:>8.1f} "
        f"{conf_lo_loocv[i]:>8.1f} {conf_hi_loocv[i]:>8.1f} {flag:>5}"
    )

# ── Paper note ────────────────────────────────────────────────
print(f"""
  Paper note (Section IV):
  Method A (split-conformal on synthetic data) violates exchangeability
  because calibration PCFs are synthetic and gold PCFs are declared by
  manufacturers. Method B (LOOCV-conformal on gold) is exchangeability-valid
  and is the reported method. Coverage = {cov_loocv:.2f} vs nominal 0.90.
  Small n (13) means finite-sample guarantee holds loosely; sensitivity
  analysis with alpha in [0.05, 0.20] is reported in ablation (Phase 9).
""")

# ── Save ──────────────────────────────────────────────────────
out = gold[[model_col, BRAND_COL]].rename(columns={model_col: "model"}).copy()
out["declared_pcf"] = y_gold
out["conformal_pred"] = gold_preds
out["conf_lo90_loocv"] = conf_lo_loocv
out["conf_hi90_loocv"] = conf_hi_loocv
out["covered_loocv"] = covered_loocv
out["conf_lo90_split"] = conf_lo_split
out["conf_hi90_split"] = conf_hi_split
out["covered_split"] = (y_gold >= conf_lo_split) & (y_gold <= conf_hi_split)
out.to_csv("data/processed/conformal_predictions_gold.csv", index=False)

result = {
    "method_A_split": {
        "n_calibration": n_calib,
        "q": round(q_split, 3),
        "coverage": round(cov_split, 3),
        "note": "Synthetic calibration — exchangeability violated",
    },
    "method_B_loocv": {
        "n_gold": n_gold,
        "coverage": round(cov_loocv, 3),
        "avg_width_kg": round(avg_width, 3),
        "target": 1 - ALPHA,
        "passed": passed,
        "note": "LOOCV-conformal on gold — exchangeability valid",
    },
    "reported_method": "method_B_loocv",
    "alpha": ALPHA,
}
with open("results/conformal_results.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"✅ data/processed/conformal_predictions_gold.csv")
print(f"✅ results/conformal_results.json")
print(
    f"\nReported coverage (LOOCV): {cov_loocv:.3f}  "
    f"| {'✅ PASSED' if passed else '❌ FAILED'}"
)
print("\nNext: python scripts/06_counterfactual_simulation.py")
