#!/usr/bin/env python3
import argparse, json, os, sys, warnings
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import joblib
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("results", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--chains", type=int, default=4)
parser.add_argument("--draws", type=int, default=3000)
parser.add_argument("--tune", type=int, default=1500)
parser.add_argument("--target-accept", type=float, default=0.95)
args = parser.parse_args()
np.random.seed(args.seed)

# ── Load ──────────────────────────────────────────────────────
gold = pd.read_csv("data/processed/gold_holdout.csv")
train = pd.read_csv("data/processed/train_modeling.csv")
priors_df = pd.read_csv("data/processed/prior_params.csv").set_index("param")

TARGET = "pcf_kg_co2e_y"
BRAND_COL = "brand"
RESID_COLS = [c for c in ["performance_index", "total_camera_mp"] if c in gold.columns]

y_gold = gold[TARGET].values.astype(float)

print("=" * 60)
print("PHASE 3 — BAYESIAN HIERARCHICAL EMULATOR (fixed)")
print(f"  target_accept={args.target_accept}  non-centered brands")
print(f"  n_gold={len(gold)}  target={TARGET}")
print(f"  PCF: mean={y_gold.mean():.1f}  std={y_gold.std():.1f}")
print("=" * 60)


# ── Priors ────────────────────────────────────────────────────
def gp(param, fm, fs):
    if param in priors_df.index:
        r = priors_df.loc[param]
        m = float(r["prior_mean"]) if pd.notnull(r.get("prior_mean")) else fm
        s = float(r["prior_sd"]) if pd.notnull(r.get("prior_sd")) else fs
        return m, s
    return fm, fs


pm_bat, ps_bat = gp("battery_ef", 120.0, 30.0)
pm_disp, ps_disp = gp("display_ef", 0.03, 0.008)
pm_mat, ps_mat = gp("material_ef", 18.0, 5.0)
pm_sem, ps_sem = gp("semiconductor_intensity", 0.8, 0.3)
pm_asm, ps_asm = gp("assembly_ef", 3.5, 1.0)
pm_trn, ps_trn = gp("transport_ef", 2.0, 0.6)
pm_lif, ps_lif = gp("lifetime_years", 3.0, 0.5)
pm_grd, ps_grd = gp("grid_ef", 0.45, 0.07)
_, ps_tau = gp("tau_brand", 0.0, 5.0)
_, ps_sig = gp("sigma", 0.0, 10.0)

# ── Features ──────────────────────────────────────────────────
bat_g = gold["battery_kwh"].values.astype(float)
disp_g = gold["display_area_cm2"].values.astype(float)
mass_g = (gold["estimated_mass_g"].values / 1000.0).astype(float)
perf_g = gold["performance_index"].values.astype(float)
ann_g = gold["annual_kwh"].values.astype(float)

X_gold_raw = gold[RESID_COLS].fillna(0).values.astype(float)
X_train_raw = train[RESID_COLS].fillna(0).values.astype(float)
scaler = StandardScaler().fit(X_train_raw)
X_gold = scaler.transform(X_gold_raw)
n_resid = X_gold.shape[1]

all_brands = sorted(
    set(gold[BRAND_COL].astype(str).tolist() + train[BRAND_COL].astype(str).tolist())
)
brand_to_idx = {b: i for i, b in enumerate(all_brands)}
n_brands = len(brand_to_idx)
brand_idx_gold = np.array([brand_to_idx.get(str(b), 0) for b in gold[BRAND_COL]])

print(f"\nBrands: {n_brands}  |  Resid cols: {RESID_COLS}")
print(f"Gold brand indices: {brand_idx_gold}\n")

# ── PyMC model ─────────────────────────────────────────────────
with pm.Model() as model:

    # Physics θ — truncated Normal from literature
    battery_ef = pm.TruncatedNormal(
        "battery_ef", mu=pm_bat, sigma=ps_bat, lower=60, upper=200
    )
    display_ef = pm.TruncatedNormal(
        "display_ef", mu=pm_disp, sigma=ps_disp, lower=0.005, upper=0.1
    )
    material_ef = pm.TruncatedNormal(
        "material_ef", mu=pm_mat, sigma=ps_mat, lower=8, upper=30
    )
    semiconductor_intensity = pm.TruncatedNormal(
        "semiconductor_intensity", mu=pm_sem, sigma=ps_sem, lower=0.1, upper=3.0
    )
    assembly_ef = pm.TruncatedNormal(
        "assembly_ef", mu=pm_asm, sigma=ps_asm, lower=1.0, upper=8.0
    )
    transport_ef = pm.TruncatedNormal(
        "transport_ef", mu=pm_trn, sigma=ps_trn, lower=0.5, upper=5.0
    )
    lifetime_years = pm.TruncatedNormal(
        "lifetime_years", mu=pm_lif, sigma=ps_lif, lower=2.0, upper=6.0
    )
    grid_ef = pm.TruncatedNormal(
        "grid_ef", mu=pm_grd, sigma=ps_grd, lower=0.05, upper=1.2
    )

    # Physics baseline
    pcf_base = (
        bat_g * battery_ef
        + disp_g * display_ef
        + mass_g * material_ef
        + ann_g * grid_ef * lifetime_years
        + perf_g * semiconductor_intensity
        + assembly_ef
        + transport_ef
    )

    # Intercept
    beta0 = pm.Normal("beta0", mu=0.0, sigma=20.0)

    # NON-CENTERED brand random effects (fixes tau_brand ESS + divergences)
    tau_brand = pm.HalfNormal("tau_brand", sigma=ps_tau)
    b_brand_z = pm.Normal("b_brand_z", mu=0.0, sigma=1.0, shape=n_brands)
    b_brand = pm.Deterministic("b_brand", b_brand_z * tau_brand)

    # Low-capacity residual h(s; φ)
    phi = pm.Normal("phi", mu=0.0, sigma=2.0, shape=n_resid)
    h = pm.math.dot(X_gold, phi)

    sigma = pm.HalfNormal("sigma", sigma=ps_sig)
    mu = pcf_base + beta0 + b_brand[brand_idx_gold] + h

    # Likelihood
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=y_gold)

    # Posterior predictive node — INCLUDES sigma (fixes coverage)
    Y_pred = pm.Normal("Y_pred", mu=mu, sigma=sigma)

# ── Sample ────────────────────────────────────────────────────
with model:
    trace = pm.sample(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        random_seed=args.seed,
        target_accept=args.target_accept,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True},
        progressbar=True,
    )
    ppc = pm.sample_posterior_predictive(
        trace, var_names=["Y_pred"], random_seed=args.seed, progressbar=False
    )

# ── Diagnostics ───────────────────────────────────────────────
diag_vars = [
    "battery_ef",
    "display_ef",
    "material_ef",
    "semiconductor_intensity",
    "assembly_ef",
    "transport_ef",
    "lifetime_years",
    "grid_ef",
    "beta0",
    "tau_brand",
    "sigma",
]
summary = az.summary(trace, var_names=diag_vars)

print("\n" + "=" * 60)
print("DIAGNOSTICS")
print("=" * 60)
print(summary.to_string())

rhat_max = float(summary["r_hat"].max())
ess_min = float(summary["ess_bulk"].min())
divergences = int(trace.sample_stats["diverging"].values.sum())

print(f"\n  R-hat max   : {rhat_max:.4f}  {'✅' if rhat_max  < 1.05 else '❌'}")
print(f"  ESS min     : {ess_min:.0f}    {'✅' if ess_min   > 200  else '❌'}")
print(f"  Divergences : {divergences}      {'✅' if divergences == 0 else '❌'}")

# ── Coverage (Y_pred includes sigma) ─────────────────────────
y_pp = ppc.posterior_predictive["Y_pred"].stack(sample=("chain","draw")).values.T

pred_median = np.median(y_pp, axis=0)

pred_lo90 = np.percentile(y_pp, 5, axis=0)
pred_hi90 = np.percentile(y_pp, 95, axis=0)

pred_lo95 = np.percentile(y_pp, 2.5, axis=0)
pred_hi95 = np.percentile(y_pp, 97.5, axis=0)

residuals = pred_median - y_gold
mae = float(np.abs(residuals).mean())
rmse = float(np.sqrt((residuals**2).mean()))
bias = float(residuals.mean())
r2 = float(1 - (residuals**2).sum() / ((y_gold - y_gold.mean()) ** 2).sum())
cov90 = float(np.mean((y_gold >= pred_lo90) & (y_gold <= pred_hi90)))
cov95 = float(np.mean((y_gold >= pred_lo95) & (y_gold <= pred_hi95)))

print("\n" + "=" * 60)
print("POSTERIOR PREDICTIVE RESULTS")
print("=" * 60)
print(f"  MAE    : {mae:.2f} kg CO₂e")
print(f"  RMSE   : {rmse:.2f} kg CO₂e")
print(f"  Bias   : {bias:.2f} kg CO₂e")
print(f"  R²     : {r2:.3f}")
print(f"  90% PP coverage : {cov90:.2f}  (target ≥ 0.85)")
print(f"  95% PP coverage : {cov95:.2f}  (target ≥ 0.90)")

model_col = "model_y" if "model_y" in gold.columns else "model_x"
print(f"\n  {'Model':<24} {'Declared':>9} {'Median':>9} {'Error':>7} {'90%CI':>7}")
print("  " + "─" * 58)
for i in range(len(gold)):
    in_ci = "✅" if pred_lo90[i] <= y_gold[i] <= pred_hi90[i] else "❌"
    print(
        f"  {str(gold[model_col].values[i])[:24]:<24} "
        f"{y_gold[i]:>9.1f} {pred_median[i]:>9.1f} {residuals[i]:>+7.1f} {in_ci:>7}"
    )

print(f"\n  {'Raw physics':.<30} MAE 7.74   R² -0.014")
print(f"  {'Ridge LOOCV':.<30} MAE 5.01   R²  0.554")
print(f"  {'Bayesian PP':.<30} MAE {mae:.2f}   R²  {r2:.3f}")

# ── Save ──────────────────────────────────────────────────────
az.to_netcdf(trace, "models/bayesian_emulator_posterior.nc")

out = gold[[model_col, "brand"]].rename(columns={model_col: "model"}).copy()
out["declared_pcf"] = y_gold
out["pred_median"] = pred_median
out["pred_lo90"] = pred_lo90
out["pred_hi90"] = pred_hi90
out["pred_lo95"] = pred_lo95
out["pred_hi95"] = pred_hi95
out["error"] = residuals
out["abs_error"] = np.abs(residuals)
out["in_90pct_ci"] = (y_gold >= pred_lo90) & (y_gold <= pred_hi90)
out.to_csv("data/processed/posterior_predictions_gold.csv", index=False)

joblib.dump(scaler, "models/resid_feature_scaler.joblib")
joblib.dump(brand_to_idx, "models/brand_to_idx.joblib")

result = {
    "run": {
        "chains": args.chains,
        "draws": args.draws,
        "tune": args.tune,
        "seed": args.seed,
        "target_accept": args.target_accept,
    },
    "diagnostics": {
        "rhat_max": rhat_max,
        "ess_min": ess_min,
        "divergences": divergences,
        "passed": rhat_max < 1.05 and ess_min > 200 and divergences == 0,
    },
    "metrics": {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "Bias": round(bias, 3),
        "R2": round(r2, 3),
        "coverage_90pct": round(cov90, 3),
        "coverage_95pct": round(cov95, 3),
    },
    "posterior_means": {
        v: round(float(trace.posterior[v].mean()), 4)
        for v in [
            "battery_ef",
            "display_ef",
            "material_ef",
            "semiconductor_intensity",
            "assembly_ef",
            "transport_ef",
            "lifetime_years",
            "grid_ef",
            "beta0",
            "sigma",
        ]
    },
    "baselines": {"raw_physics_mae": 7.74, "ridge_loocv_mae": 5.01},
}
with open("models/bayesian_emulator_summary.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\n✅ models/bayesian_emulator_posterior.nc")
print(f"✅ data/processed/posterior_predictions_gold.csv")
print(f"✅ models/bayesian_emulator_summary.json")
passed = result["diagnostics"]["passed"]
print(f"\n{'✅ All diagnostics passed.' if passed else '⚠️  Check diagnostics above.'}")
print("Next: python scripts/05_conformal_certification.py")
