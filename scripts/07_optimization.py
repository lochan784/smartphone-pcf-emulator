#!/usr/bin/env python3
"""
Phase 7 — Pareto Optimization (exact 0/1 knapsack via dynamic programming no GLPK dependency)
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

bat_m = float(gold["battery_kwh"].mean())
disp_m = float(gold["display_area_cm2"].mean())
mass_m = float((gold["estimated_mass_g"] / 1000.0).mean())
perf_m = float(gold["performance_index"].mean())
ann_m = float(gold["annual_kwh"].mean())

baseline_pcf = (
    bat_m * theta["battery_ef"]
    + disp_m * theta["display_ef"]
    + mass_m * theta["material_ef"]
    + ann_m * theta["grid_ef"] * theta["lifetime_years"]
    + perf_m * theta["semiconductor_intensity"]
    + theta["assembly_ef"]
    + theta["transport_ef"]
)

print("=" * 64)
print("PHASE 7 — PARETO OPTIMIZATION")
print("=" * 64)
print(f"\n  Fleet avg baseline PCF : {baseline_pcf:.2f} kg CO₂e")

LEVERS = [
    (
        "grid_decarb",
        "Grid decarbonisation (−30%)",
        20.0,
        ann_m * theta["grid_ef"] * theta["lifetime_years"] * 0.30,
    ),
    (
        "battery_nextgen",
        "Next-gen battery chemistry (−25%)",
        35.0,
        bat_m * theta["battery_ef"] * 0.25,
    ),
    (
        "recycled_mat",
        "30% recycled material content",
        15.0,
        mass_m * theta["material_ef"] * 0.30,
    ),
    (
        "display_opt",
        "Low-energy OLED display (−20%)",
        8.0,
        disp_m * theta["display_ef"] * 0.20,
    ),
    (
        "efficient_chip",
        "Next-node semiconductor (−15%)",
        25.0,
        perf_m * theta["semiconductor_intensity"] * 0.15,
    ),
]
# Note: lifetime_ext excluded — it INCREASES PCF, never optimal to select

lever_ids = [L[0] for L in LEVERS]
lever_costs = np.array([L[2] for L in LEVERS])
lever_reds = np.array([L[3] for L in LEVERS])
n_levers = len(LEVERS)

print(f"\n  {'Lever':<42} {'Cost':>6} {'ΔPCF':>8}  {'Ratio':>6}")
print("  " + "─" * 66)
for L in LEVERS:
    ratio = L[3] / L[2]
    print(f"  {L[1]:<42} ${L[2]:>5.0f} {-L[3]:>+7.3f} kg  {ratio:.4f}")


# ── Exact 0/1 knapsack (dynamic programming — no solver needed) ──
def knapsack_dp(costs_int, reductions, budget_int):
    """Exact 0/1 knapsack via DP. Costs must be integers."""
    n = len(costs_int)
    dp = np.zeros(budget_int + 1)
    sel = [[[] for _ in range(budget_int + 1)] for _ in range(n + 1)]
    for i in range(n):
        for w in range(budget_int, costs_int[i] - 1, -1):
            val_with = dp[w - costs_int[i]] + reductions[i]
            if val_with > dp[w]:
                dp[w] = val_with
                sel[0][w] = sel[0][w - costs_int[i]] + [i]
    # Reconstruct
    best_sel = []
    best_val = 0.0
    for w in range(budget_int + 1):
        if dp[w] > best_val:
            best_val = dp[w]
            best_sel = sel[0][w]
    return best_sel, best_val


# Scale costs to integers (×1 since already integer $)
costs_int = lever_costs.astype(int)

budgets = np.arange(0, 205, 5, dtype=float)
pareto_rows = []

for budget in budgets:
    b_int = int(budget)
    # Only consider levers with positive reduction and cost <= budget
    eligible = [
        i for i in range(n_levers) if lever_reds[i] > 0 and costs_int[i] <= b_int
    ]

    if not eligible:
        sel_idx = []
    else:
        e_costs = costs_int[eligible]
        e_reds = lever_reds[eligible]
        # DP on eligible subset
        dp = np.zeros(b_int + 1)
        track = [[] for _ in range(b_int + 1)]
        for ii, (c, r) in enumerate(zip(e_costs, e_reds)):
            for w in range(b_int, c - 1, -1):
                if dp[w - c] + r > dp[w]:
                    dp[w] = dp[w - c] + r
                    track[w] = track[w - c] + [eligible[ii]]
        sel_idx = track[b_int]

    sel_mask = np.zeros(n_levers, int)
    sel_mask[sel_idx] = 1
    total_red = float(sel_mask @ lever_reds)
    total_cost = float(sel_mask @ lever_costs)

    pareto_rows.append(
        {
            "budget": budget,
            "total_cost": round(total_cost, 2),
            "pcf_reduction_kg": round(total_red, 3),
            "optimised_pcf": round(baseline_pcf - total_red, 3),
            "selected_levers": "|".join(
                lever_ids[i] for i in range(n_levers) if sel_mask[i]
            ),
            "n_levers": int(sel_mask.sum()),
        }
    )

pareto_df = pd.DataFrame(pareto_rows)

print("\n" + "─" * 64)
print("PARETO FRONTIER (key budget points)")
print("─" * 64)
print(f"  {'Budget':>8} {'Cost':>8} {'ΔPCF':>9} {'Opt PCF':>9}  Levers")
print("  " + "─" * 72)
for _, row in pareto_df[
    pareto_df["budget"].isin([0, 25, 50, 75, 100, 150, 200])
].iterrows():
    levers = (
        row["selected_levers"].replace("|", ", ") if row["selected_levers"] else "none"
    )
    print(
        f"  ${row['budget']:>7.0f} ${row['total_cost']:>7.2f} "
        f"{row['pcf_reduction_kg']:>+9.3f} {row['optimised_pcf']:>9.3f}  {levers}"
    )

# ── Posterior uncertainty on recommended $50 solution ─────────
rec = pareto_df[pareto_df["budget"] == 50].iloc[0]
rec_lev = rec["selected_levers"].split("|") if rec["selected_levers"] else []

base_d = (
    bat_m * bat_d
    + disp_m * disp_d
    + mass_m * mat_d
    + ann_m * grd_d * lif_d
    + perf_m * semi_d
    + asm_d
    + trn_d
    + beta0_d
)

opt_d = base_d.copy()
if "grid_decarb" in rec_lev:
    opt_d -= ann_m * grd_d * lif_d * 0.30
if "battery_nextgen" in rec_lev:
    opt_d -= bat_m * bat_d * 0.25
if "recycled_mat" in rec_lev:
    opt_d -= mass_m * mat_d * 0.30
if "display_opt" in rec_lev:
    opt_d -= disp_m * disp_d * 0.20
if "efficient_chip" in rec_lev:
    opt_d -= perf_m * semi_d * 0.15

delta_d = opt_d - base_d

print("\n" + "=" * 64)
print("RECOMMENDED $50 SOLUTION — POSTERIOR UNCERTAINTY")
print("=" * 64)
print(f"  Levers selected : {rec_lev}")
print(f"  Total cost      : ${rec['total_cost']:.2f}/device")
print(f"  ΔPCF median     : {float(np.median(delta_d)):+.3f} kg CO₂e")
print(
    f"  90% CI          : [{float(np.percentile(delta_d,5)):+.3f}, "
    f"{float(np.percentile(delta_d,95)):+.3f}] kg CO₂e"
)
print(f"  P(ΔPCF < 0)     : {float((delta_d < 0).mean()):.3f}")

# ── Save ──────────────────────────────────────────────────────
pareto_df.to_csv("results/pareto_frontier.csv", index=False)

result = {
    "baseline_pcf": round(baseline_pcf, 3),
    "solver": "exact_dp_knapsack",
    "levers": [
        {"id": L[0], "label": L[1], "cost": L[2], "pcf_reduction_kg": round(L[3], 3)}
        for L in LEVERS
    ],
    "recommended_50": {
        "levers": rec_lev,
        "cost": round(float(rec["total_cost"]), 2),
        "delta_pcf_median": round(float(np.median(delta_d)), 3),
        "delta_pcf_lo90": round(float(np.percentile(delta_d, 5)), 3),
        "delta_pcf_hi90": round(float(np.percentile(delta_d, 95)), 3),
        "prob_improvement": round(float((delta_d < 0).mean()), 3),
    },
}
with open("results/optimization_results.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\n✅ results/pareto_frontier.csv")
print(f"✅ results/optimization_results.json")
print(f"\nNext: python scripts/08_ablation.py")
