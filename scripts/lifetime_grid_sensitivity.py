import numpy as np
import matplotlib.pyplot as plt

# Sensitivity data
grid = np.array([0.45,0.315,0.225,0.135,0.05])
delta = np.array([5.61,3.93,2.81,1.68,0.62])

plt.figure(figsize=(10,5))

# scatter + line
plt.plot(grid, delta, marker='o', linewidth=2, markersize=9)

# threshold line
plt.axhline(0, linestyle='--', linewidth=2, color='red',
            label='Break-even threshold')

# grid styling
plt.grid(True, linestyle='--', alpha=0.5)

# labels
plt.title("Lifetime Extension Sensitivity", fontsize=18)
plt.xlabel("Grid carbon intensity (kg CO2e/kWh)", fontsize=14)
plt.ylabel("Δ PCF (kg)", fontsize=14)

# annotate first point
plt.annotate(
    "Current global grid",
    xy=(0.45,5.61),
    xytext=(0.38,5.2),
    arrowprops=dict(arrowstyle="->",color="black"),
    fontsize=12
)

# invert x axis like your plot
plt.gca().invert_xaxis()

plt.legend(fontsize=12)

plt.tight_layout()

plt.savefig("i6 - lifetime_grid_sensitivity.png", dpi=300)
plt.show()