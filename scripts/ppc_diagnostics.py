import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Load posterior ─────────────────────────────────────────
idata = az.from_netcdf("models/bayesian_emulator_posterior.nc")

# Load observed PCF values (same ones used in training)
df = pd.read_csv("data/processed/train_modeling.csv")
observed = df["pcf_kg_co2e"].values

# ── 1. R-hat & ESS Table ───────────────────────────────────
summary = az.summary(
    idata,
    var_names=[
        "beta0",
        "battery_ef",
        "display_ef",
        "material_ef",
        "grid_ef",
        "transport_ef",
        "lifetime_years",
        "tau_brand",
        "sigma",
        "phi"
    ],
    round_to=3
)

print(summary[["mean","sd","ess_bulk","ess_tail","r_hat"]].to_latex())


# ── 2. Trace Plots ─────────────────────────────────────────
az.plot_trace(
    idata,
    var_names=[
        "battery_ef",
        "grid_ef",
        "material_ef",
        "tau_brand",
        "sigma",
        "lifetime_years"
    ],
    combined=False,
    compact=False,
    figsize=(14,18)
)

plt.suptitle("Trace Plots — 4 Chains (post-warmup)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("results/appendix_b_traces.pdf", dpi=300)


# ── 3. Posterior Predictive Checks ─────────────────────────
fig, axes = plt.subplots(1,2, figsize=(12,5))

# Posterior predictive samples
ppc_draws = idata.posterior["Y_pred"].values
ppc_flat = ppc_draws.reshape(-1, ppc_draws.shape[-1])

# (a) Histogram overlay
ax = axes[0]

for i in range(100):
    ax.hist(
        ppc_flat[np.random.randint(ppc_flat.shape[0])],
        bins=15,
        alpha=0.04,
        color="steelblue",
        density=True
    )

ax.hist(
    observed,
    bins=10,
    alpha=0.9,
    color="firebrick",
    density=True,
    label="Observed (n=13)",
    zorder=5
)

ax.set_xlabel("PCF (kg CO₂e)")
ax.set_ylabel("Density")
ax.set_title("Posterior Predictive Check — Histogram")
ax.legend()


# (b) QQ plot
ppc_quantiles = np.quantile(ppc_flat, np.linspace(0.01,0.99,50), axis=0).mean(axis=1)
obs_quantiles = np.quantile(observed, np.linspace(0.01,0.99,50))

axes[1].scatter(ppc_quantiles, obs_quantiles)

lims = [
    min(ppc_quantiles.min(), obs_quantiles.min()) - 2,
    max(ppc_quantiles.max(), obs_quantiles.max()) + 2
]

axes[1].plot(lims, lims, "k--", label="Perfect calibration")

axes[1].set_xlabel("Posterior Predictive Quantiles")
axes[1].set_ylabel("Observed Quantiles")
axes[1].set_title("QQ Plot — Observed vs Simulated")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/appendix_b_ppc.pdf", dpi=300)


# ── 4. Divergence count ───────────────────────────────────
divergences = idata.sample_stats["diverging"].values.sum()
print(f"Total divergent transitions: {divergences}")