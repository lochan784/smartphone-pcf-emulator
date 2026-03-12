import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Grid Decarb (-30%)', 'Next-Gen Battery (-25%)', 'Ext. Lifetime (3 to 5y)']
values = [-6.07, -1.04, 12.20]
display_labels = ['-6.1%', '-1.0%', '+12.2%']
colors = ['#228B22', '#228B22', '#FF3333']

fig, ax = plt.subplots(figsize=(10, 6))

# Bar plot
bars = ax.bar(labels, values, color=colors, edgecolor='black', width=0.75, linewidth=0.8)

# Adding value labels with dynamic positioning
for bar, label in zip(bars, display_labels):
    height = bar.get_height()
    # Position: 'bottom' for positive (above bar), 'top' for negative (below bar)
    va = 'bottom' if height > 0 else 'top'
    offset = 0.3 if height > 0 else -0.3
    ax.text(bar.get_x() + bar.get_width() / 2, height + offset, label,
            ha='center', va=va, fontweight='bold', fontsize=12)

# Baseline
ax.axhline(0, color='black', linewidth=1)

# Axis Ticks and Grid
ax.set_yticks(np.arange(-8, 16, 2))
ax.grid(axis='y', linestyle='--', color='gray', alpha=0.5)
ax.set_axisbelow(True)

# Labels
ax.set_ylabel('Change in Total PCF (%)', fontsize=13)
ax.tick_params(axis='both', which='major', labelsize=11)

# Style tweaks
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(-9, 15) # Buffer for labels

plt.tight_layout()
plt.show()




