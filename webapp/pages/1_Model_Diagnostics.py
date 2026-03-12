import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ------------------------------------------------
# Project paths
# ------------------------------------------------

WEBAPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(WEBAPP)

MODELS = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")

st.set_page_config(page_title="Model Diagnostics", layout="wide")
st.header("🔍 Bayesian Model Diagnostics")

# ------------------------------------------------
# Load diagnostics summary
# ------------------------------------------------

SUMMARY_PATH = os.path.join(MODELS, "bayesian_emulator_summary.json")

with open(SUMMARY_PATH) as f:
    diag = json.load(f)["diagnostics"]

# ─────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────

def compute_metrics(df):
    mae = df["residual"].mean()
    rmse = np.sqrt((df["residual"]**2).mean())
    r2 = 1 - np.sum((df["declared"]-df["pred_median"])**2) / \
         np.sum((df["declared"]-df["declared"].mean())**2)
    return mae, rmse, r2


def mae_bar(ax, df, group_col, color_map=None, title=""):
    groups  = df.groupby(group_col)["residual"]

    labels  = list(groups.groups.keys())
    means   = [groups.get_group(l).mean() for l in labels]
    errors  = [groups.get_group(l).std(ddof=1) if len(groups.get_group(l))>1 else 0
               for l in labels]
    counts  = [len(groups.get_group(l)) for l in labels]

    colors = [color_map.get(l,"#888888") if color_map else "#5b9bd5" for l in labels]

    bars = ax.bar(labels, means, yerr=errors, capsize=4,
                  color=colors, edgecolor="white", linewidth=0.6, zorder=3)

    ax.set_ylabel("MAE (kg CO₂e)", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

    if len(means) > 0:
        ax.set_ylim(0, max(means)*1.4)

    for bar,n,m in zip(bars,counts,means):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+max(means)*0.04 if len(means)>0 else 0.2,
                f"n={n}\n{m:.1f}",
                ha="center",fontsize=7,color="#333")


def lowess_numpy(y, x, frac=0.75):
    n, h = len(x), int(np.ceil(frac * len(x)))
    out = np.empty(n)

    for i in range(n):

        d = np.abs(x-x[i])
        idx = np.argsort(d)[:h]

        dmax = d[idx].max()

        w = (1-(d[idx]/dmax)**3)**3 if dmax>0 else np.ones(h)

        X_ = np.column_stack([np.ones(h), x[idx]])

        try:
            b = np.linalg.lstsq(np.diag(w)@X_,
                                np.diag(w)@y[idx],rcond=None)[0]
        except np.linalg.LinAlgError:
            b=[y[idx].mean(),0]

        out[i]=b[0]+b[1]*x[i]

    order=np.argsort(x)

    return np.column_stack([x[order],out[order]])


# ─────────────────────────────────────────────────────────────
# 1. Convergence metrics
# ─────────────────────────────────────────────────────────────

SUMMARY_PATH = os.path.join(ROOT, "models", "bayesian_emulator_summary.json")

with open(SUMMARY_PATH) as f:
    diag = json.load(f)["diagnostics"]

c1,c2,c3=st.columns(3)

c1.metric("R-hat max",diag["rhat_max"])
c2.metric("ESS min",int(diag["ess_min"]))
c3.metric("Divergences",diag["divergences"])

if diag["divergences"]==0:
    st.success("Model converged successfully")
else:
    st.warning("Sampler reported divergences — inspect posterior traces")

st.divider()

# ─────────────────────────────────────────────────────────────
# 2. Load forensic CSV
# ─────────────────────────────────────────────────────────────

FORENSICS_PATH = os.path.join(RESULTS, "error_forensics_full.csv")

if not os.path.exists(FORENSICS_PATH):
    st.error(f"Missing file: {FORENSICS_PATH}")
    st.stop()

df = pd.read_csv(FORENSICS_PATH)

# ------------------------------------------------
# Auto-derive missing columns if needed
# ------------------------------------------------

if "residual" not in df.columns and {"declared","pred_median"} <= set(df.columns):
    df["residual"] = (df["declared"] - df["pred_median"]).abs()

if "signed_residual" not in df.columns and {"declared","pred_median"} <= set(df.columns):
    df["signed_residual"] = df["declared"] - df["pred_median"]

if "ci_width" not in df.columns and {"pred_hi90","pred_lo90"} <= set(df.columns):
    df["ci_width"] = df["pred_hi90"] - df["pred_lo90"]

required={"model_key","brand","declared","pred_median",
          "residual","signed_residual","year","tier"}

missing_cols = required - set(df.columns)

if missing_cols:
    st.error(f"CSV still missing required columns: {missing_cols}")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Interactive filters
# ─────────────────────────────────────────────────────────────

st.sidebar.header("Filters")

brands=st.sidebar.multiselect(
    "Brand",df.brand.unique(),default=list(df.brand.unique())
)

tiers=st.sidebar.multiselect(
    "Price Tier",df.tier.unique(),default=list(df.tier.unique())
)

years=st.sidebar.multiselect(
    "Release Year",sorted(df.year.unique()),default=list(df.year.unique())
)

df=df[
    (df.brand.isin(brands)) &
    (df.tier.isin(tiers)) &
    (df.year.isin(years))
]

# ─────────────────────────────────────────────────────────────
# 3. KPI row
# ─────────────────────────────────────────────────────────────

mae,rmse,r2=compute_metrics(df)

k1,k2,k3,k4=st.columns(4)

k1.metric("MAE",f"{mae:.2f} kg")
k2.metric("RMSE",f"{rmse:.2f} kg")
k3.metric("R²",f"{r2:.3f}")
k4.metric("Devices",len(df))

st.divider()

# ─────────────────────────────────────────────────────────────
# 4. MAE bar charts
# ─────────────────────────────────────────────────────────────

BRAND_COLORS={
"Apple":"#1f77b4",
"Samsung":"#d62728",
"Google":"#2ca02c"
}

col_brand,col_tier,col_year=st.columns(3)

with col_brand:

    fig,ax=plt.subplots(figsize=(3.2,3))
    mae_bar(ax,df,"brand",BRAND_COLORS,"MAE by Brand")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_tier:

    fig,ax=plt.subplots(figsize=(3.2,3))
    mae_bar(ax,df,"tier",title="MAE by Price Tier")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_year:

    fig,ax=plt.subplots(figsize=(3.2,3))
    mae_bar(ax,df,"year",title="MAE by Release Year")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ─────────────────────────────────────────────────────────────
# 5. Residual scatter
# ─────────────────────────────────────────────────────────────

st.subheader("🔬 LOOCV Residual Pattern")

show_lowess=st.checkbox("Show LOWESS trend",value=True)

fig,ax=plt.subplots(figsize=(7,4))

ax.axhline(0,color="black",lw=0.9,ls="--",alpha=0.7)

sd=df["signed_residual"].std(ddof=1)

ax.axhspan(-sd,sd,color="steelblue",alpha=0.07)

for _,row in df.iterrows():

    color=BRAND_COLORS.get(row["brand"],"#888")

    marker={"Apple":"o","Samsung":"^","Google":"s"}.get(row["brand"],"D")

    ax.scatter(
        row["pred_median"],
        row["signed_residual"],
        color=color,
        marker=marker,
        s=70,
        alpha=0.9
    )

if show_lowess:

    sm=lowess_numpy(
        df["signed_residual"].values,
        df["pred_median"].values
    )

    ax.plot(sm[:,0],sm[:,1],
            color="dimgrey",
            lw=1.8,
            label="LOWESS")

ax.set_xlabel("Posterior Predictive Mean (kg CO₂e)")
ax.set_ylabel("Residual (Observed − Predicted)")
ax.set_title("LOOCV Residuals vs Predicted PCF")

ax.grid(axis="y",ls=":",lw=0.5)

handles=[mpatches.Patch(color=v,label=k) for k,v in BRAND_COLORS.items()]

ax.legend(handles=handles,fontsize=8)

plt.tight_layout()

st.pyplot(fig)

plt.close()

st.divider()

# ─────────────────────────────────────────────────────────────
# 6. Top-3 outliers
# ─────────────────────────────────────────────────────────────

st.subheader("⚠️ Largest Prediction Errors")

top3=df.nlargest(3,"residual")[
["model_key","brand","declared","pred_median",
 "signed_residual","residual"]
].copy()

top3.columns=[
"Device","Brand","Declared",
"Predicted","Signed Residual","Abs Residual"
]

top3=top3.round(2).reset_index(drop=True)

st.dataframe(top3,use_container_width=True,hide_index=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# 7. Component uncertainty
# ─────────────────────────────────────────────────────────────

st.subheader("🧩 Emission Factor Posterior Uncertainty")

comp={
"Grid EF θ_g":{"Post SD":0.068,"Feature Scale":"30 kWh","ΔPCF SD":2.040},
"Assembly EF θ_a":{"Post SD":0.98,"Feature Scale":"1 unit","ΔPCF SD":0.980},
"Material EF θ_m":{"Post SD":4.60,"Feature Scale":"0.21 kg","ΔPCF SD":0.966},
"Display EF θ_d":{"Post SD":0.008,"Feature Scale":"110 cm²","ΔPCF SD":0.880},
"Transport EF θ_t":{"Post SD":0.58,"Feature Scale":"1 unit","ΔPCF SD":0.580},
"Battery EF θ_b":{"Post SD":28.5,"Feature Scale":"0.017 kWh","ΔPCF SD":0.485},
"Semiconductor θ_s":{"Post SD":0.244,"Feature Scale":"1.2 idx","ΔPCF SD":0.293},
}

comp_df=pd.DataFrame(comp).T.reset_index().rename(
columns={"index":"Parameter"}
)

comp_df=comp_df.sort_values("ΔPCF SD",ascending=False)

st.dataframe(comp_df,use_container_width=True,hide_index=True)

st.caption(
"ΔPCF SD = posterior SD × feature scale for a representative device. "
"Grid EF dominates because it multiplies lifetime electricity (~30 kWh)."
)