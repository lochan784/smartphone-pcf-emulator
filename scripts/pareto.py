import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------------------------------
# Load Bayesian posterior results
# -------------------------------
idata = az.from_netcdf("models/bayesian_emulator_posterior.nc")
loo = az.loo(idata)

k_vals = loo.pareto_k.values
x = np.arange(len(k_vals))

phones = [
    "iphone 14 pro max","iphone 13","iphone 15","iphone 14",
    "iphone 14 pro","iphone 12","pixel 8 pro","pixel 7",
    "pixel 6","galaxy s23 ultra","iphone 15 pro max",
    "galaxy s23","iphone 15 pro"
]

# -------------------------------
# Plot Pareto-k diagnostics
# -------------------------------
plt.figure(figsize=(8,5))

# background
plt.gca().set_facecolor("#f2f2f2")

# scatter
plt.scatter(
    x,
    k_vals,
    color="#1f4e79",
    s=90,
    marker="o",
    edgecolor="black",
    zorder=3
)

# threshold line
plt.axhline(
    0.7,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Influence threshold (k = 0.7)"
)

# annotate influential phones
for i, k in enumerate(k_vals):
    if k > 0.7:
        plt.annotate(
            phones[i],
            xy=(i, k),
            xytext=(i+0.5, k+0.08),
            arrowprops=dict(arrowstyle="->", color="darkred"),
            fontsize=9,
            color="darkred"
        )

plt.grid(True, linestyle="--", alpha=0.6)

plt.xticks(x, phones, rotation=45, ha="right")

plt.xlabel("Device Model")
plt.ylabel("Pareto k")

plt.title("LOO Pareto-k Diagnostic")

plt.legend()

plt.tight_layout()

plt.savefig("results/loo_pareto_k_styled.png", dpi=300)

plt.show()

# -------------------------------
# Identify influential observations
# -------------------------------
df = pd.read_csv("data/processed/gold_holdout.csv")

df["pareto_k"] = k_vals

print(df[["model_y", "pareto_k"]].sort_values("pareto_k", ascending=False))

influential = df[df["pareto_k"] > 0.7]

print("\nInfluential observations:")
print(influential[["model_y", "pareto_k"]])

# Save influential points
influential.to_csv("results/influential_observations.csv", index=False)