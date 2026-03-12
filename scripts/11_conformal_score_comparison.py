"""
scripts/11_conformal_score_comparison.py
========================================
Compares absolute vs. normalized nonconformity scores for
Jackknife+ LOOCV conformal prediction.

Figure style matches the paper's reference figures:
  - Clean sans-serif, larger readable fonts
  - Horizontal-only dashed grid for bar chart (Fig 1)
  - Full dashed grid for dot plot (Fig 2)
  - No top/right spines, white background
  - Bold value annotations, saturated colours
  - Legend placed below axes to keep bar labels unobscured

Run from repo root
------------------
    python scripts/11_conformal_score_comparison.py
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── 0. Style (matches paper reference figures) ────────────────────────────────
mpl.rcParams.update({
    "font.family"        : "sans-serif",
    "font.sans-serif"    : ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size"          : 11,
    "axes.titlesize"     : 12,
    "axes.labelsize"     : 11,
    "xtick.labelsize"    : 10,
    "ytick.labelsize"    : 10,
    "legend.fontsize"    : 10,
    "axes.linewidth"     : 0.8,
    "xtick.major.width"  : 0.8,
    "ytick.major.width"  : 0.8,
    "xtick.major.size"   : 4,
    "ytick.major.size"   : 4,
    "xtick.direction"    : "out",
    "ytick.direction"    : "out",
    "axes.grid"          : True,
    "grid.linewidth"     : 0.6,
    "grid.color"         : "#cccccc",
    "grid.linestyle"     : "--",
    "axes.axisbelow"     : True,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "figure.facecolor"   : "white",
    "axes.facecolor"     : "white",
    "figure.dpi"         : 150,
    "savefig.dpi"        : 300,
    "savefig.bbox"       : "tight",
    "savefig.pad_inches" : 0.08,
})

C_ABS  = "#2166ac"   # steel blue  — absolute score
C_NORM = "#c0392b"   # brick red   — normalized score
C_MISS = "#aaaaaa"   # grey        — not covered
C_REF  = "#333333"   # near-black  — reference lines & text

BRAND_COLOR = {
    "apple"  : "#2166ac",
    "samsung": "#e67e00",
    "google" : "#27ae60",
}

LABEL_MAP = {
    "iphone 14 pro max": "iPhone 14 Pro Max",
    "iphone 14 pro"    : "iPhone 14 Pro",
    "iphone 14"        : "iPhone 14",
    "iphone 15 pro max": "iPhone 15 Pro Max",
    "iphone 15 pro"    : "iPhone 15 Pro",
    "iphone 15"        : "iPhone 15",
    "iphone 13"        : "iPhone 13",
    "iphone 12"        : "iPhone 12",
    "galaxy s23 ultra" : "Galaxy S23 Ultra",
    "galaxy s23"       : "Galaxy S23",
    "pixel 8 pro"      : "Pixel 8 Pro",
    "pixel 7"          : "Pixel 7",
    "pixel 6"          : "Pixel 6",
}

# ── 1. Paths ───────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "posterior_predictions_gold.csv"
RES_DIR   = ROOT / "results"
FIG_DIR   = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.10

# ── 2. Load & normalise ────────────────────────────────────────────────────────
print(f"Loading: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)
print(f"  Columns found: {list(df.columns)}")

df = df.rename(columns={
    "model"        : "device",
    "declared_pcf" : "y_true",
    "pred_median"  : "y_pred_mean",
})
df["y_pred_sd"] = (df["pred_hi90"] - df["pred_lo90"]) / (2 * 1.645)
df["tier"] = df["device"].apply(
    lambda n: "Mid-range"
    if ("pixel 6" in str(n).lower() and "pro" not in str(n).lower())
    else "Flagship"
)
df["device_label"] = (
    df["device"].str.lower().str.strip()
    .map(LABEL_MAP).fillna(df["device"])
)

print(f"\n  {len(df)} devices loaded.")
print(df[["device_label", "brand", "tier", "y_true",
          "y_pred_mean", "y_pred_sd"]].to_string(index=False))

# ── 3. Scores ──────────────────────────────────────────────────────────────────
df["abs_score"]  = (df["y_true"] - df["y_pred_mean"]).abs()
df["norm_score"] = df["abs_score"] / df["y_pred_sd"]

# ── 4. Jackknife+ LOOCV ────────────────────────────────────────────────────────
def jackknife_plus_q(scores: np.ndarray, alpha: float) -> float:
    n   = len(scores)
    lvl = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    return float(np.quantile(scores, lvl))


def loocv_coverage(df_in: pd.DataFrame, score_col: str,
                   alpha: float = ALPHA) -> pd.DataFrame:
    scores = df_in[score_col].values
    preds  = df_in["y_pred_mean"].values
    sds    = df_in["y_pred_sd"].values
    trues  = df_in["y_true"].values
    rows   = []
    for i in range(len(df_in)):
        loo = np.delete(scores, i)
        q   = jackknife_plus_q(loo, alpha)
        if score_col == "abs_score":
            lo, hi = preds[i] - q, preds[i] + q
        else:
            lo, hi = preds[i] - q * sds[i], preds[i] + q * sds[i]
        rows.append({
            "device"       : df_in["device"].iloc[i],
            "device_label" : df_in["device_label"].iloc[i],
            "brand"        : df_in["brand"].iloc[i],
            "tier"         : df_in["tier"].iloc[i],
            "y_true"       : trues[i],
            "y_pred"       : preds[i],
            "y_sd"         : sds[i],
            "lo"           : round(lo, 2),
            "hi"           : round(hi, 2),
            "width"        : round(hi - lo, 2),
            "covered"      : bool(lo <= trues[i] <= hi),
        })
    return pd.DataFrame(rows)


abs_df  = loocv_coverage(df, "abs_score")
norm_df = loocv_coverage(df, "norm_score")

# ── 5. Summary tables ──────────────────────────────────────────────────────────
def cov(sub): return "---" if len(sub) == 0 else f"{sub['covered'].mean():.0%}"

def summarise(res, label):
    return {
        "Score Type"     : label,
        "Overall Cov."   : cov(res),
        "Flagship Cov."  : cov(res[res["tier"].str.contains("lag", case=False)]),
        "Mid-range Cov." : cov(res[res["tier"].str.contains("id",  case=False)]),
        "Apple Cov."     : cov(res[res["brand"].str.lower() == "apple"]),
        "Samsung Cov."   : cov(res[res["brand"].str.lower() == "samsung"]),
        "Google Cov."    : cov(res[res["brand"].str.lower() == "google"]),
        "Avg Width (kg)" : f"{res['width'].mean():.1f}",
        "n_covered"      : int(res["covered"].sum()),
        "n_total"        : len(res),
    }

summary = pd.DataFrame([summarise(abs_df, "Absolute"),
                         summarise(norm_df, "Normalized")])
summary.to_csv(RES_DIR / "conformal_score_comparison.csv", index=False)
print(f"\n── Summary ──────────────────────────────────────────────────────────")
print(summary.to_string(index=False))


def tertile_cov(res):
    tmp           = res.copy()
    tmp["tertile"] = pd.qcut(res["y_pred"], q=3, labels=["Low", "Mid", "High"])
    return tmp.groupby("tertile")["covered"].agg(["mean", "count"])

abs_tert  = tertile_cov(abs_df).rename(columns={"mean": "Absolute",   "count": "n_abs"})
norm_tert = tertile_cov(norm_df).rename(columns={"mean": "Normalized", "count": "n_norm"})
tert_df   = abs_tert.join(norm_tert[["Normalized"]])
tert_df.to_csv(RES_DIR / "conformal_score_tertile.csv")
print(f"\n── Tertile Coverage ──────────────────────────────────────────────────")
print(tert_df)

q33 = abs_df["y_pred"].quantile(0.33)
q67 = abs_df["y_pred"].quantile(0.67)
tert_x = [
    f"Low\n($\\leq${q33:.0f} kg)",
    f"Mid\n({q33:.0f}\u2013{q67:.0f} kg)",
    f"High\n($>${q67:.0f} kg)",
]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1  Conditional coverage by tertile
# Matches reference counterfactual bar style: clean sans-serif, horizontal
# dashed grid only, bold value annotations, legend below axes.
# ══════════════════════════════════════════════════════════════════════════════
abs_cov_tert  = tert_df["Absolute"].tolist()
norm_cov_tert = tert_df["Normalized"].tolist()

fig1, ax = plt.subplots(figsize=(5.4, 3.7))
fig1.subplots_adjust(bottom=0.30)

x, w, sep = np.arange(3), 0.28, 0.05
b1 = ax.bar(x - w/2 - sep/2, abs_cov_tert,  width=w, color=C_ABS,  linewidth=0, zorder=2)
b2 = ax.bar(x + w/2 + sep/2, norm_cov_tert, width=w, color=C_NORM, linewidth=0, zorder=2)

ax.axhline(0.90, color=C_REF, lw=1.0, ls="--", zorder=3)
ax.text(2.68, 0.905, "90%", fontsize=9.5, color=C_REF, va="bottom", ha="left")

# When both scores give equal coverage, show one centred label
for i, (va, vn) in enumerate(zip(abs_cov_tert, norm_cov_tert)):
    top = max(va, vn)
    if abs(va - vn) < 1e-9:
        ax.text(x[i], top + 0.023, f"{va:.0%}",
                ha="center", va="bottom",
                fontsize=10.5, color=C_REF, fontweight="bold")
    else:
        ax.text(x[i] - sep/2, va + 0.023, f"{va:.0%}",
                ha="center", va="bottom",
                fontsize=10, color=C_REF, fontweight="bold")
        ax.text(x[i] + sep/2 + w, vn + 0.023, f"{vn:.0%}",
                ha="center", va="bottom",
                fontsize=10, color=C_REF, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(tert_x, fontsize=10)
ax.set_ylabel("Empirical Coverage", fontsize=11)
ax.set_xlabel("Predicted PCF Tertile", fontsize=11)
ax.set_ylim(0, 1.17)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
ax.grid(axis="y", zorder=0)
ax.grid(axis="x", visible=False)
ax.set_xlim(-0.6, 2.6)

h1 = mpatches.Patch(color=C_ABS,  label=r"Absolute $|y-\hat{y}|$")
h2 = mpatches.Patch(color=C_NORM, label=r"Normalized $|y-\hat{y}|/\hat{\sigma}$")
fig1.legend(handles=[h1, h2], ncol=2,
            loc="lower center", bbox_to_anchor=(0.5, 0.01),
            frameon=True, framealpha=0.97, edgecolor="#cccccc",
            fontsize=10, handlelength=1.0, handletextpad=0.5,
            borderpad=0.5, labelspacing=0.3, columnspacing=1.0)

p1 = FIG_DIR / "conformal_coverage_by_tertile.pdf"
fig1.savefig(p1)
fig1.savefig(p1.with_suffix(".png"))
plt.close(fig1)
print(f"\nSaved: {p1}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2  Per-device interval widths — Cleveland dot plot
# Matches reference Pareto style: full dashed grid, clean dots,
# brand colour-coded y-axis labels, legend below axes.
# ══════════════════════════════════════════════════════════════════════════════
# Sort by predicted PCF ascending (lowest at bottom)
abs_sorted  = abs_df.sort_values("y_pred").reset_index(drop=True)
norm_sorted = norm_df.sort_values("y_pred").reset_index(drop=True)

n  = len(abs_sorted)
ys = np.arange(n)

fig2, ax2 = plt.subplots(figsize=(5.4, 5.2))
fig2.subplots_adjust(left=0.33, right=0.97, top=0.93, bottom=0.22)

for i in range(n):
    aw = abs_sorted["width"].iloc[i];  ac = abs_sorted["covered"].iloc[i]
    nw = norm_sorted["width"].iloc[i]; nc = norm_sorted["covered"].iloc[i]

    # Grey connector between the two score markers
    ax2.plot([min(aw, nw), max(aw, nw)], [i, i],
             color="#d0d0d0", lw=1.4, zorder=1, solid_capstyle="round")

    # Absolute score — filled square (covered) / open square (not covered)
    ax2.plot(aw, i, marker="s", ms=7.5, linestyle="none",
             markerfacecolor=C_ABS  if ac else "white",
             markeredgecolor=C_ABS  if ac else C_MISS,
             markeredgewidth=1.4, zorder=4)

    # Normalized score — filled circle (covered) / open circle (not covered)
    ax2.plot(nw, i, marker="o", ms=7.5, linestyle="none",
             markerfacecolor=C_NORM if nc else "white",
             markeredgecolor=C_NORM if nc else C_MISS,
             markeredgewidth=1.4, zorder=4)

mean_a = abs_sorted["width"].mean()
mean_n = norm_sorted["width"].mean()
ax2.axvline(mean_a, color=C_ABS,  lw=0.85, ls=":", alpha=0.6, zorder=0)
ax2.axvline(mean_n, color=C_NORM, lw=0.85, ls=":", alpha=0.6, zorder=0)
ax2.text(mean_n - 0.2, n - 0.1, f"{mean_n:.1f}",
         fontsize=8.5, color=C_NORM, ha="right", va="bottom", style="italic")
ax2.text(mean_a + 0.2, n - 0.1, f"{mean_a:.1f}",
         fontsize=8.5, color=C_ABS,  ha="left",  va="bottom", style="italic")

# Y-axis: device names coloured by brand
ax2.set_yticks(ys)
ax2.set_yticklabels(abs_sorted["device_label"].tolist(), fontsize=9.5)
for tick, brand in zip(ax2.get_yticklabels(),
                       abs_sorted["brand"].str.lower()):
    tick.set_color(BRAND_COLOR.get(brand, C_REF))

ax2.set_xlabel(r"90% Conformal Interval Width (kg CO$_2$e)", fontsize=11)
ax2.set_xlim(14, 31)
ax2.xaxis.set_major_locator(mticker.MultipleLocator(4))
ax2.set_ylim(-0.7, n + 0.2)
ax2.grid(axis="x", zorder=0)
ax2.grid(axis="y", zorder=0)
ax2.set_title(r"90% Conformal Interval Widths ($n=13$, $\alpha=0.10$)",
              fontsize=12, pad=6)

h_a = mlines.Line2D([], [], marker="s", ms=7, ls="none",
                    markerfacecolor=C_ABS, markeredgecolor=C_ABS,
                    markeredgewidth=1.2, label=r"Absolute $|y-\hat{y}|$")
h_n = mlines.Line2D([], [], marker="o", ms=7, ls="none",
                    markerfacecolor=C_NORM, markeredgecolor=C_NORM,
                    markeredgewidth=1.2, label=r"Normalized $|y-\hat{y}|/\hat{\sigma}$")
h_x = mlines.Line2D([], [], marker="s", ms=7, ls="none",
                    markerfacecolor="white", markeredgecolor=C_MISS,
                    markeredgewidth=1.2, label="Not covered (Galaxy S23)")
h_brands = [mpatches.Patch(color=BRAND_COLOR[b], label=b.capitalize())
            for b in ["apple", "samsung", "google"]]

fig2.legend(handles=[h_a, h_n, h_x] + h_brands,
            ncol=3, loc="lower center", bbox_to_anchor=(0.65, 0.01),
            fontsize=9, frameon=True, framealpha=0.97, edgecolor="#cccccc",
            handlelength=1.0, handletextpad=0.5,
            borderpad=0.5, labelspacing=0.3, columnspacing=0.8)

p2 = FIG_DIR / "conformal_width_comparison.pdf"
fig2.savefig(p2)
fig2.savefig(p2.with_suffix(".png"))
plt.close(fig2)
print(f"Saved: {p2}")

# ── LaTeX rows ─────────────────────────────────────────────────────────────────
print("\n── LaTeX rows for Table C.1 ──────────────────────────────────────────")
latex_cols = ["Score Type", "Overall Cov.", "Flagship Cov.",
              "Mid-range Cov.", "Apple Cov.", "Samsung Cov.", "Avg Width (kg)"]
for _, row in summary[latex_cols].iterrows():
    print("  " + " & ".join(str(v) for v in row.values) + " \\\\")

print("\n── LaTeX rows for Table C.2 – tertile coverage ───────────────────────")
for idx, row in tert_df.iterrows():
    print(f"  {idx} & {int(row['n_abs'])} & "
          f"{row['Absolute']:.0%} & {row['Normalized']:.0%} \\\\")

# ── JSON ───────────────────────────────────────────────────────────────────────
out_json = {
    "absolute"  : {"overall_coverage": float(abs_df["covered"].mean()),
                   "avg_width_kg"    : float(abs_df["width"].mean()),
                   "per_device"      : abs_df.to_dict(orient="records")},
    "normalized": {"overall_coverage": float(norm_df["covered"].mean()),
                   "avg_width_kg"    : float(norm_df["width"].mean()),
                   "per_device"      : norm_df.to_dict(orient="records")},
    "tertile"   : tert_df.reset_index().to_dict(orient="records"),
}
with open(RES_DIR / "conformal_score_comparison.json", "w") as f:
    json.dump(out_json, f, indent=2)

print("\nDone. Files written:")
print(f"  {RES_DIR}/conformal_score_comparison.csv")
print(f"  {RES_DIR}/conformal_score_tertile.csv")
print(f"  {RES_DIR}/conformal_score_comparison.json")
print(f"  {p1}  (+ .png)")
print(f"  {p2}  (+ .png)")