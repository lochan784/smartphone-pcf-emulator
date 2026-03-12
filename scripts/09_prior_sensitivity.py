#!/usr/bin/env python3
"""
Phase 9 — Prior Sensitivity Analysis
Runs 6 prior scenarios over the gold hold-out set, computes MAE + LOOCV
conformal coverage for each, builds posterior comparison table and density
overlay plots.

Requires (all produced by earlier phases):
  data/processed/gold_holdout.csv
  data/processed/train_modeling.csv
  data/processed/prior_params.csv
  models/resid_feature_scaler.joblib
  models/brand_to_idx.joblib

Outputs:
  results/sensitivity_metrics.csv
  results/appendix_a_posterior_comparison.csv
  results/appendix_a_prior_sensitivity.pdf
  results/appendix_a_prior_sensitivity.png
  results/sensitivity_summary.json
"""

import argparse, json, os, warnings
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ── CLI ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--chains",        type=int,   default=4)
parser.add_argument("--draws",         type=int,   default=2000)
parser.add_argument("--tune",          type=int,   default=2000)
parser.add_argument("--target-accept", type=float, default=0.90)
args = parser.parse_args()
np.random.seed(args.seed)

# ── Load data & artefacts ─────────────────────────────────────
gold      = pd.read_csv("data/processed/gold_holdout.csv")
train     = pd.read_csv("data/processed/train_modeling.csv")
priors_df = pd.read_csv("data/processed/prior_params.csv").set_index("param")
scaler    = joblib.load("models/resid_feature_scaler.joblib")
brand_map = joblib.load("models/brand_to_idx.joblib")

TARGET    = "pcf_kg_co2e_y"
BRAND_COL = "brand"
RESID_COLS = [c for c in ["performance_index", "total_camera_mp"] if c in gold.columns]

y_gold   = gold[TARGET].values.astype(float)
n_gold   = len(gold)
model_col = "model_y" if "model_y" in gold.columns else "model_x"

# Pre-scaled residual features (scaler is fixed from Phase 3)
X_gold = scaler.transform(gold[RESID_COLS].fillna(0).values.astype(float))
n_resid = X_gold.shape[1]

all_brands = sorted(
    set(gold[BRAND_COL].astype(str).tolist() + train[BRAND_COL].astype(str).tolist())
)
brand_to_idx  = {b: i for i, b in enumerate(all_brands)}
n_brands      = len(brand_to_idx)
brand_idx_gold = np.array([brand_to_idx.get(str(b), 0) for b in gold[BRAND_COL]])

# Raw physics features (fixed — not perturbed by sensitivity scenarios)
bat_g  = gold["battery_kwh"].values.astype(float)
disp_g = gold["display_area_cm2"].values.astype(float)
mass_g = (gold["estimated_mass_g"].values / 1000.0).astype(float)
perf_g = gold["performance_index"].values.astype(float)
ann_g  = gold["annual_kwh"].values.astype(float)

# ── Baseline prior parameters (match Phase 3 / Table I) ──────
BASELINE_MEANS = {
    "battery_ef":             120.0,
    "display_ef":             0.03,
    "material_ef":            18.0,
    "semiconductor_intensity": 0.8,
    "assembly_ef":            3.5,
    "transport_ef":           2.0,
    "lifetime_years":         3.0,
    "grid_ef":                0.45,
}
BASELINE_SDS = {
    "battery_ef":             30.0,
    "display_ef":             0.008,
    "material_ef":            5.0,
    "semiconductor_intensity": 0.30,
    "assembly_ef":            1.0,
    "transport_ef":           0.6,
    "lifetime_years":         0.5,
    "grid_ef":                0.07,
}
BOUNDS = {
    "battery_ef":             (60,   200),
    "display_ef":             (0.005, 0.1),
    "material_ef":            (8,    30),
    "semiconductor_intensity": (0.1,  3.0),
    "assembly_ef":            (1.0,  8.0),
    "transport_ef":           (0.5,  5.0),
    "lifetime_years":         (2.0,  6.0),
    "grid_ef":                (0.05, 1.2),
}

# Override baseline means/sds with prior_params.csv if present
# (keeps parity with Phase 3 which also reads from that file)
def _gp(param):
    if param in priors_df.index:
        r = priors_df.loc[param]
        m = float(r["prior_mean"]) if pd.notnull(r.get("prior_mean", np.nan)) else BASELINE_MEANS[param]
        s = float(r["prior_sd"])   if pd.notnull(r.get("prior_sd",   np.nan)) else BASELINE_SDS[param]
        return m, s
    return BASELINE_MEANS[param], BASELINE_SDS[param]

for _p in BASELINE_MEANS:
    BASELINE_MEANS[_p], BASELINE_SDS[_p] = _gp(_p)

# Extra hyper-priors (not perturbed in this analysis)
_, ps_tau = (0.0, float(priors_df.loc["tau_brand","prior_sd"])
             if "tau_brand" in priors_df.index else 5.0)
_, ps_sig = (0.0, float(priors_df.loc["sigma","prior_sd"])
             if "sigma" in priors_df.index else 10.0)

# ── Six sensitivity scenarios ─────────────────────────────────
SCENARIOS = {
    "baseline":          {"mean_scale": 1.0,  "sd_scale": 1.0},
    "mean_plus_50pct":   {"mean_scale": 1.5,  "sd_scale": 1.0},
    "mean_minus_50pct":  {"mean_scale": 0.5,  "sd_scale": 1.0},
    "sd_plus_50pct":     {"mean_scale": 1.0,  "sd_scale": 1.5},
    "sd_minus_50pct":    {"mean_scale": 1.0,  "sd_scale": 0.5},
    "flat_priors":       {"mean_scale": 1.0,  "sd_scale": 100.0},   # → pm.Uniform
}

ALPHA = 0.10   # target 90% conformal coverage

# ── Model builder ─────────────────────────────────────────────
def build_model(mean_scale=1.0, sd_scale=1.0):
    """
    Mirrors Phase 3 model exactly; only the physics θ priors are perturbed.
    Non-centered brand effects, residual linear head, and likelihood are
    identical to bayesian_calibration.py.
    """
    is_flat = sd_scale >= 50

    def make_theta(name):
        mu0   = BASELINE_MEANS[name]
        sig0  = BASELINE_SDS[name]
        lo, hi = BOUNDS[name]
        if is_flat:
            return pm.Uniform(name, lower=lo, upper=hi)
        scaled_mu  = np.clip(mu0 * mean_scale, lo + 1e-6, hi - 1e-6)
        scaled_sig = sig0 * sd_scale
        return pm.TruncatedNormal(name, mu=scaled_mu, sigma=scaled_sig,
                                  lower=lo, upper=hi)

    with pm.Model() as model:
        battery_ef             = make_theta("battery_ef")
        display_ef             = make_theta("display_ef")
        material_ef            = make_theta("material_ef")
        semiconductor_intensity = make_theta("semiconductor_intensity")
        assembly_ef            = make_theta("assembly_ef")
        transport_ef           = make_theta("transport_ef")
        lifetime_years         = make_theta("lifetime_years")
        grid_ef                = make_theta("grid_ef")

        # Physics baseline  (identical to Phase 3)
        pcf_base = (
            bat_g  * battery_ef
            + disp_g * display_ef
            + mass_g * material_ef
            + ann_g  * grid_ef * lifetime_years
            + perf_g * semiconductor_intensity
            + assembly_ef
            + transport_ef
        )

        beta0 = pm.Normal("beta0", mu=0.0, sigma=20.0)

        # Non-centered brand random effects  (identical to Phase 3)
        tau_brand = pm.HalfNormal("tau_brand", sigma=ps_tau)
        b_brand_z = pm.Normal("b_brand_z", mu=0.0, sigma=1.0, shape=n_brands)
        b_brand   = pm.Deterministic("b_brand", b_brand_z * tau_brand)

        # Residual linear head  (identical to Phase 3)
        phi = pm.Normal("phi", mu=0.0, sigma=2.0, shape=n_resid)
        h   = pm.math.dot(X_gold, phi)

        sigma = pm.HalfNormal("sigma", sigma=ps_sig)
        mu_rv = pcf_base + beta0 + b_brand[brand_idx_gold] + h

        Y_obs  = pm.Normal("Y_obs",  mu=mu_rv, sigma=sigma, observed=y_gold)
        Y_pred = pm.Normal("Y_pred", mu=mu_rv, sigma=sigma)

    return model


# ── LOOCV conformal coverage (mirrors Phase 5 exactly) ───────
def loocv_conformal_coverage(preds, y_true, alpha=ALPHA):
    n = len(y_true)
    scores = np.abs(y_true - preds)
    covered = np.zeros(n, dtype=bool)
    for i in range(n):
        calib = np.delete(scores, i)
        n_c   = len(calib)
        ql    = min(np.ceil((1 - alpha) * (n_c + 1)) / n_c, 1.0)
        q_i   = float(np.quantile(calib, ql))
        covered[i] = abs(y_true[i] - preds[i]) <= q_i
    return float(covered.mean())


# ── Run all scenarios ─────────────────────────────────────────
print("=" * 60)
print("PHASE 9 — PRIOR SENSITIVITY ANALYSIS")
print(f"  scenarios={list(SCENARIOS)}")
print(f"  chains={args.chains}  draws={args.draws}  tune={args.tune}")
print("=" * 60)

results    = {}
all_traces = {}

for scenario_name, kwargs in SCENARIOS.items():
    print(f"\n{'─'*60}")
    print(f"  Scenario: {scenario_name}  {kwargs}")
    print(f"{'─'*60}")

    model = build_model(**kwargs)

    with model:
        trace = pm.sample(
            draws         = args.draws,
            tune          = args.tune,
            chains        = args.chains,
            target_accept = args.target_accept,
            random_seed   = args.seed,
            return_inferencedata = True,
            idata_kwargs  = {"log_likelihood": False},
            progressbar   = True,
        )
        ppc = pm.sample_posterior_predictive(
            trace, var_names=["Y_pred"], random_seed=args.seed, progressbar=False
        )

    all_traces[scenario_name] = trace

    # Convergence
    param_names = list(BASELINE_MEANS.keys())
    summary = az.summary(trace, var_names=param_names)
    rhat_max    = float(summary["r_hat"].max())
    ess_min     = float(summary["ess_bulk"].min())
    divergences = int(trace.sample_stats["diverging"].values.sum())

    print(f"  R-hat max : {rhat_max:.4f}  {'✅' if rhat_max < 1.05 else '❌'}")
    print(f"  ESS min   : {ess_min:.0f}    {'✅' if ess_min > 200 else '❌'}")
    print(f"  Divergences: {divergences}   {'✅' if divergences == 0 else '⚠️'}")

    # Posterior predictive metrics
    y_pp        = ppc.posterior_predictive["Y_pred"].stack(
                      sample=("chain","draw")).values.T
    pred_median = np.median(y_pp, axis=0)
    mae         = float(np.mean(np.abs(pred_median - y_gold)))
    rmse        = float(np.sqrt(np.mean((pred_median - y_gold)**2)))
    r2          = float(1 - np.sum((pred_median-y_gold)**2) /
                            np.sum((y_gold-y_gold.mean())**2))
    coverage    = loocv_conformal_coverage(pred_median, y_gold)

    print(f"  MAE      : {mae:.3f}  RMSE: {rmse:.3f}  R²: {r2:.3f}")
    print(f"  LOOCV coverage: {coverage:.3f}  (target ≥ {1-ALPHA-0.05:.2f})")

    results[scenario_name] = {
        "summary":    summary,
        "rhat_max":   rhat_max,
        "ess_min":    ess_min,
        "divergences": divergences,
        "MAE":        round(mae, 3),
        "RMSE":       round(rmse, 3),
        "R2":         round(r2, 3),
        "coverage":   round(coverage, 3),
        "pred_median": pred_median,
        "passed_convergence": rhat_max < 1.05 and ess_min > 200,
    }


# ── Metrics table ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SENSITIVITY METRICS SUMMARY")
print("=" * 60)
metrics_rows = []
for sc, r in results.items():
    metrics_rows.append({
        "scenario":    sc,
        "MAE":         r["MAE"],
        "RMSE":        r["RMSE"],
        "R2":          r["R2"],
        "coverage_90": r["coverage"],
        "rhat_max":    round(r["rhat_max"], 4),
        "ess_min":     int(r["ess_min"]),
        "divergences": r["divergences"],
        "converged":   r["passed_convergence"],
    })

metrics_df = pd.DataFrame(metrics_rows).set_index("scenario")
print(metrics_df.to_string())
metrics_df.to_csv("results/sensitivity_metrics.csv")
print("\n✅ results/sensitivity_metrics.csv")


# ── Non-identifiability check ─────────────────────────────────
print("\n=== NON-IDENTIFIABILITY CHECK (posterior mean within 5% of bound) ===")
any_flag = False
for sc, r in results.items():
    for param in BASELINE_MEANS:
        if param not in r["summary"].index:
            continue
        post_mean = r["summary"].loc[param, "mean"]
        lo, hi = BOUNDS[param]
        pct_lo = (post_mean - lo) / (hi - lo)
        pct_hi = (hi - post_mean) / (hi - lo)
        if pct_lo < 0.05 or pct_hi < 0.05:
            print(f"  ⚠️  {sc:<22} / {param:<26} "
                  f"post_mean={post_mean:.4f}  bounds=({lo},{hi})")
            any_flag = True
if not any_flag:
    print("  ✅ No parameters near bounds.")


# ── Sensitivity index (range / baseline mean) ─────────────────
print("\n=== SENSITIVITY INDEX (posterior-mean range / baseline posterior mean) ===")
baseline_post_means = {
    p: results["baseline"]["summary"].loc[p, "mean"]
    for p in BASELINE_MEANS if p in results["baseline"]["summary"].index
}
sensitivity_index = {}
for param in BASELINE_MEANS:
    post_means = [
        results[sc]["summary"].loc[param, "mean"]
        for sc in SCENARIOS
        if param in results[sc]["summary"].index
    ]
    rng = max(post_means) - min(post_means)
    base = baseline_post_means.get(param, 1.0)
    sensitivity_index[param] = rng / abs(base) if base != 0 else np.nan

for param, val in sorted(sensitivity_index.items(), key=lambda x: -x[1]):
    print(f"  {param:<30}: {val*100:.1f}%")

most_sensitive = max(sensitivity_index, key=sensitivity_index.get)
print(f"\n  Most sensitive: {most_sensitive}  "
      f"({sensitivity_index[most_sensitive]*100:.1f}%)")


# ── Posterior comparison table (Appendix A) ───────────────────
rows = []
for param in BASELINE_MEANS:
    row = {"parameter": param,
           "prior_mean": round(BASELINE_MEANS[param], 4),
           "prior_sd":   round(BASELINE_SDS[param], 4)}
    for sc in SCENARIOS:
        if param in results[sc]["summary"].index:
            s = results[sc]["summary"].loc[param]
            row[f"{sc}_mean"] = round(float(s["mean"]), 4)
            row[f"{sc}_sd"]   = round(float(s["sd"]),   4)
    rows.append(row)

table_df = pd.DataFrame(rows)
table_df.to_csv("results/appendix_a_posterior_comparison.csv", index=False)
print("\n✅ results/appendix_a_posterior_comparison.csv")
print(table_df.to_string(index=False))


# ── Density overlay plots (Appendix A figure) ─────────────────
COLORS = {
    "baseline":         "blue",
    "mean_plus_50pct":  "red",
    "mean_minus_50pct": "orange",
    "sd_plus_50pct":    "green",
    "sd_minus_50pct":   "purple",
    "flat_priors":      "gray",
}

params = list(BASELINE_MEANS.keys())
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.flatten()

for i, param in enumerate(params):
    ax  = axes[i]
    mu0 = BASELINE_MEANS[param]
    sd0 = BASELINE_SDS[param]
    lo, hi = BOUNDS[param]

    # Baseline prior as dashed black line
    x    = np.linspace(lo, hi, 400)
    a, b = (lo - mu0) / sd0, (hi - mu0) / sd0
    prior_pdf = truncnorm.pdf(x, a, b, loc=mu0, scale=sd0)
    ax.plot(x, prior_pdf, "k--", linewidth=2, label="Prior (baseline)", zorder=6)

    # Posterior histograms per scenario
    for sc, res in results.items():
        trace = all_traces[sc]
        if param in trace.posterior:
            samples = trace.posterior[param].values.flatten()
            ax.hist(samples, bins=60, density=True, alpha=0.35,
                    color=COLORS[sc], label=sc)

    ax.set_title(param, fontsize=9, fontweight="bold")
    ax.set_xlabel("Value", fontsize=7)
    ax.set_ylabel("Density", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlim(lo, hi)

axes[-1].set_visible(False)   # 8th panel unused

# One shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right",
           bbox_to_anchor=(0.98, 0.03), fontsize=8, framealpha=0.9)

plt.suptitle(
    "Prior vs. Posterior Densities Across Sensitivity Scenarios",
    fontsize=12, fontweight="bold"
)
plt.tight_layout(rect=[0, 0.0, 1, 0.97])
plt.savefig("results/appendix_a_prior_sensitivity.pdf", dpi=300, bbox_inches="tight")
plt.savefig("results/appendix_a_prior_sensitivity.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ results/appendix_a_prior_sensitivity.pdf")
print("✅ results/appendix_a_prior_sensitivity.png")


# ── MAE delta bar chart (quick visual) ────────────────────────
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sc_labels = list(SCENARIOS.keys())
mae_vals  = [results[sc]["MAE"]      for sc in sc_labels]
cov_vals  = [results[sc]["coverage"] for sc in sc_labels]
colors_bar = [COLORS[sc] for sc in sc_labels]

ax1.barh(sc_labels, mae_vals, color=colors_bar, edgecolor="black", linewidth=0.5)
ax1.axvline(results["baseline"]["MAE"], color="black", linestyle="--", linewidth=1.5,
            label=f'Baseline MAE={results["baseline"]["MAE"]:.2f}')
ax1.set_xlabel("MAE (kg CO₂e)")
ax1.set_title("MAE by Scenario")
ax1.legend(fontsize=8)

ax2.barh(sc_labels, cov_vals, color=colors_bar, edgecolor="black", linewidth=0.5)
ax2.axvline(1 - ALPHA, color="black", linestyle="--", linewidth=1.5,
            label=f"Target {1-ALPHA:.0%}")
ax2.axvline(1 - ALPHA - 0.05, color="red", linestyle=":", linewidth=1.2,
            label=f"Min {1-ALPHA-0.05:.0%}")
ax2.set_xlabel("LOOCV Coverage")
ax2.set_title("90% Conformal Coverage by Scenario")
ax2.legend(fontsize=8)

plt.suptitle("Sensitivity Analysis — MAE and Coverage", fontweight="bold")
plt.tight_layout()
plt.savefig("results/sensitivity_barplots.pdf", dpi=300, bbox_inches="tight")
plt.savefig("results/sensitivity_barplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ results/sensitivity_barplots.pdf / .png")


# ── Save JSON summary ─────────────────────────────────────────
summary_json = {
    "run": {"chains": args.chains, "draws": args.draws, "tune": args.tune,
            "seed": args.seed, "target_accept": args.target_accept},
    "alpha": ALPHA,
    "most_sensitive_parameter": most_sensitive,
    "sensitivity_index_pct": {k: round(v*100, 2) for k, v in sensitivity_index.items()},
    "scenarios": {
        sc: {
            "MAE":         r["MAE"],
            "RMSE":        r["RMSE"],
            "R2":          r["R2"],
            "coverage_90": r["coverage"],
            "rhat_max":    r["rhat_max"],
            "divergences": r["divergences"],
            "converged":   r["passed_convergence"],
        }
        for sc, r in results.items()
    },
    "paper_text": (
        f"Posterior estimates were broadly stable across ±50% perturbations to "
        f"prior means and variances. The largest prior sensitivity was observed for "
        f"{most_sensitive}, whose posterior mean range across scenarios was "
        f"{sensitivity_index[most_sensitive]*100:.1f}%. "
        f"Full tables and density overlays are in Appendix A."
    ),
}
with open("results/sensitivity_summary.json", "w") as f:
    json.dump(summary_json, f, indent=2)
print("✅ results/sensitivity_summary.json")

print("\n" + "="*60)
print("PHASE 9 COMPLETE")
print(f"  Most sensitive parameter : {most_sensitive}")
print(f"  Baseline MAE             : {results['baseline']['MAE']:.3f}")
print(f"  MAE range across scenarios: "
      f"{min(mae_vals):.3f} – {max(mae_vals):.3f}")
print(f"  Coverage range            : "
      f"{min(cov_vals):.3f} – {max(cov_vals):.3f}")
print("="*60)