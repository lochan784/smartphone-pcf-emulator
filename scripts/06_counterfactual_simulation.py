#!/usr/bin/env python3
"""
Phase 6 — Counterfactual Simulation
Propagates full posterior through three decarbonisation scenarios.
Each returns a ΔPCF distribution with 90% credible intervals.
"""

import json, os, warnings
import numpy as np
import pandas as pd
import arviz as az
import joblib

warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)

gold = pd.read_csv("data/processed/gold_holdout.csv")
trace = az.from_netcdf("models/bayesian_emulator_posterior.nc")
scaler = joblib.load("models/resid_feature_scaler.joblib")
brand_map = joblib.load("models/brand_to_idx.joblib")

TARGET = "pcf_kg_co2e_y"
BRAND_COL = "brand"
RESID_COLS = [c for c in ["performance_index", "total_camera_mp"] if c in gold.columns]

y_gold = gold[TARGET].values.astype(float)
model_col = "model_y" if "model_y" in gold.columns else "model_x"

post = trace.posterior


def draws(var):
    return post[var].values.reshape(-1)


bat_d = draws("battery_ef")
disp_d = draws("display_ef")
mat_d = draws("material_ef")
semi_d = draws("semiconductor_intensity")
asm_d = draws("assembly_ef")
trn_d = draws("transport_ef")
lif_d = draws("lifetime_years")
grd_d = draws("grid_ef")
beta0_d = draws("beta0")
b_brand_d = post["b_brand"].values.reshape(-1, post["b_brand"].shape[-1])
phi_d = post["phi"].values.reshape(-1, post["phi"].shape[-1])

N_draws = len(bat_d)
n_gold = len(gold)

bat_g = gold["battery_kwh"].values.astype(float)
disp_g = gold["display_area_cm2"].values.astype(float)
mass_g = (gold["estimated_mass_g"].values / 1000.0).astype(float)
perf_g = gold["performance_index"].values.astype(float)
ann_g = gold["annual_kwh"].values.astype(float)
bidx_g = np.array([brand_map.get(str(b), 0) for b in gold[BRAND_COL]])
X_gold = scaler.transform(gold[RESID_COLS].fillna(0).values.astype(float))


def pcf_draws(grd=None, lif=None, bat=None):
    _grd = grd_d if grd is None else grd
    _lif = lif_d if lif is None else lif
    _bat = bat_d if bat is None else bat
    base = (
        np.outer(_bat, bat_g)
        + np.outer(disp_d, disp_g)
        + np.outer(mat_d, mass_g)
        + np.outer(_grd * _lif, ann_g)
        + np.outer(semi_d, perf_g)
        + asm_d[:, None]
        + trn_d[:, None]
    )
    return base + beta0_d[:, None] + b_brand_d[:, bidx_g] + phi_d @ X_gold.T


print("=" * 62)
print("PHASE 6 — COUNTERFACTUAL SIMULATION")
print("=" * 62)
print(f"  Posterior draws : {N_draws}  |  Gold devices : {n_gold}")

pcf_base = pcf_draws()

SCENARIOS = {
    "grid_decarbonisation": (
        "Grid decarbonisation (−30%)",
        "National grids decarbonise 30% — IEA net-zero pathway",
    ),
    "next_gen_battery": (
        "Next-gen battery chemistry (−25%)",
        "Solid-state/LFP cuts battery manufacturing EF by 25%",
    ),
    "extended_lifetime": (
        "Extended device lifetime (3→5 yr)",
        "EU right-to-repair extends average lifetime to 5 years",
    ),
}

results = {}

for key, (label, desc) in SCENARIOS.items():
    print("\n" + "─" * 62)
    print(f"  {label}")
    print(f"  {desc}")

    if key == "grid_decarbonisation":
        pcf_new = pcf_draws(grd=grd_d * 0.70)
    elif key == "next_gen_battery":
        pcf_new = pcf_draws(bat=bat_d * 0.75)
    elif key == "extended_lifetime":
        pcf_new = pcf_draws(lif=np.full_like(lif_d, 5.0))

    delta = pcf_new - pcf_base  # (N_draws, n_gold)
    fleet = delta.mean(axis=1)  # (N_draws,)

    d50 = float(np.median(fleet))
    d05 = float(np.percentile(fleet, 5))
    d95 = float(np.percentile(fleet, 95))
    base_med = float(np.median(pcf_base.mean(axis=1)))
    pct = d50 / base_med * 100

    dev_med = np.median(delta, axis=0)
    dev_lo90 = np.percentile(delta, 5, axis=0)
    dev_hi90 = np.percentile(delta, 95, axis=0)

    print(f"\n  Fleet mean ΔPCF : {d50:+.2f} kg CO₂e  90%CI [{d05:+.2f}, {d95:+.2f}]")
    print(f"  Relative change : {pct:+.1f}% of baseline")

    print(f"\n  {'Device':<26} {'ΔPCF':>8} {'Lo90':>8} {'Hi90':>8}")
    print("  " + "─" * 52)
    for i in range(n_gold):
        name = str(gold[model_col].values[i])[:26]
        print(
            f"  {name:<26} {dev_med[i]:>+8.2f} {dev_lo90[i]:>+8.2f} {dev_hi90[i]:>+8.2f}"
        )

    results[key] = {
        "label": label,
        "desc": desc,
        "fleet": {
            "delta_median_kg": round(d50, 3),
            "delta_lo90_kg": round(d05, 3),
            "delta_hi90_kg": round(d95, 3),
            "delta_pct": round(pct, 2),
        },
        "per_device": [
            {
                "model": str(gold[model_col].values[i]),
                "brand": str(gold[BRAND_COL].values[i]),
                "declared_pcf": round(float(y_gold[i]), 2),
                "delta_median": round(float(dev_med[i]), 3),
                "delta_lo90": round(float(dev_lo90[i]), 3),
                "delta_hi90": round(float(dev_hi90[i]), 3),
            }
            for i in range(n_gold)
        ],
    }

base_fleet = pcf_base.mean(axis=1)
base_med = float(np.median(base_fleet))
base_lo = float(np.percentile(base_fleet, 5))
base_hi = float(np.percentile(base_fleet, 95))

print("\n" + "=" * 62)
print("SCENARIO COMPARISON")
print("=" * 62)
print(f"  Baseline fleet PCF: {base_med:.1f} kg  90%CI [{base_lo:.1f}, {base_hi:.1f}]")
print(f"\n  {'Scenario':<38} {'ΔPCF':>8} {'90%CI':>22} {'%':>6}")
print("  " + "─" * 76)
for key, r in results.items():
    f = r["fleet"]
    print(
        f"  {r['label']:<38} {f['delta_median_kg']:>+8.2f} "
        f"[{f['delta_lo90_kg']:>+7.2f},{f['delta_hi90_kg']:>+7.2f}] "
        f"{f['delta_pct']:>+6.1f}%"
    )

full = {
    "baseline": {
        "fleet_median_kg": round(base_med, 3),
        "fleet_lo90_kg": round(base_lo, 3),
        "fleet_hi90_kg": round(base_hi, 3),
        "n_draws": N_draws,
        "n_devices": n_gold,
    },
    "scenarios": results,
}
with open("results/counterfactual_results.json", "w") as f:
    json.dump(full, f, indent=2)

pd.DataFrame(
    [
        {"scenario": k, "scenario_label": r["label"], **d}
        for k, r in results.items()
        for d in r["per_device"]
    ]
).to_csv("data/processed/counterfactual_per_device.csv", index=False)

print(f"\n✅ results/counterfactual_results.json")
print(f"✅ data/processed/counterfactual_per_device.csv")
print(f"\nPhysical sense check:")
print(f"  Grid −30%    → expect negative ΔPCF (use-phase reduction)")
print(f"  Battery −25% → expect negative ΔPCF (mfg reduction)")
print(f"  Lifetime 5yr → expect POSITIVE ΔPCF (more use-phase kWh)")
print(f"\nNext: python scripts/07_optimization.py")
