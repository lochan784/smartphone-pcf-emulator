# generate_appendix_b.py
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)

# Load idata
idata = az.from_netcdf("models/bayesian_emulator_posterior.nc")
print("idata loaded successfully")

# ── TRACE PLOTS ───────────────────────────────────────────────────────────────
az.plot_trace(
    idata,
    var_names=["battery_ef", "grid_ef", "material_ef",
               "semiconductor_intensity", "tau_brand", "sigma"],
    combined=False,
    compact=False,
    figsize=(14, 18)
)
plt.suptitle("Trace Plots — 4 Chains (post-warmup)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("results/appendix_b_traces.pdf", bbox_inches="tight", dpi=300)
plt.close()
print("✅ Saved: results/appendix_b_traces.pdf")

# ── POSTERIOR PREDICTIVE CHECKS ───────────────────────────────────────────────
# Y_pred is stored inside idata.posterior (confirmed from your sampling output)
ppc_draws = idata.posterior["Y_pred"].values      # shape: (chains, draws, 13)
ppc_flat  = ppc_draws.reshape(-1, 13)             # shape: (12000, 13)

# Observed values — from your gold predictions CSV
import pandas as pd
df = pd.read_csv("data/processed/posterior_predictions_gold.csv")
print("CSV columns:", list(df.columns))           # prints so you can verify
observed = df["declared_pcf"].values                  # declared PCF values
print("Observed values:", observed)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
rng = np.random.default_rng(42)

# Left panel: histogram overlay
ax = axes[0]
for i in range(100):
    ax.hist(ppc_flat[rng.integers(len(ppc_flat))],
            bins=15, alpha=0.04, color="steelblue", density=True)
ax.hist(observed, bins=10, alpha=0.9, color="firebrick",
        density=True, label="Observed (n=13)", zorder=5)
ax.set_xlabel("PCF (kg CO₂e)")
ax.set_ylabel("Density")
ax.set_title("Posterior Predictive Check — Histogram")
ax.legend()

# Right panel: QQ plot
q_grid        = np.linspace(0.01, 0.99, 50)
ppc_quantiles = np.quantile(ppc_flat, q_grid, axis=0).mean(axis=1)
obs_quantiles = np.quantile(observed, q_grid)
lo = min(ppc_quantiles.min(), obs_quantiles.min()) - 2
hi = max(ppc_quantiles.max(), obs_quantiles.max()) + 2
axes[1].scatter(ppc_quantiles, obs_quantiles, color="steelblue", zorder=5)
axes[1].plot([lo, hi], [lo, hi], "k--", label="Perfect calibration")
axes[1].set_xlabel("Posterior Predictive Quantiles")
axes[1].set_ylabel("Observed Quantiles")
axes[1].set_title("QQ Plot — Observed vs. Simulated")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/appendix_b_ppc.pdf", bbox_inches="tight", dpi=300)
plt.close()
print("✅ Saved: results/appendix_b_ppc.pdf")

print("\n✅ All done. Check results/ folder for both PDFs.")