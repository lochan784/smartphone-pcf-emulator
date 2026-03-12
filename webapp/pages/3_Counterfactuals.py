import os
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

WEBAPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(WEBAPP)

RESULTS = os.path.join(ROOT, "results")

st.set_page_config(page_title="Counterfactuals", layout="wide")
st.header("🔄 Counterfactual Policy Scenarios")

RESULTS_PATH = os.path.join(RESULTS, "counterfactual_results.json")
if not os.path.exists(RESULTS_PATH):
    st.error(f"{RESULTS_PATH} not found. Run the counterfactual pipeline first.")
    st.stop()

with open(RESULTS_PATH, "r") as f:
    data = json.load(f)

# -------------------------
# Baseline summary
# -------------------------
st.subheader("Baseline")
baseline = data.get("baseline", {})
if not baseline:
    st.info("No explicit baseline in the results file — using scenario 0 as baseline if present.")
else:
    cols = st.columns(4)
    i = 0
    for k, v in baseline.items():
        if isinstance(v, (int, float)):
            cols[i % 4].metric(k.replace("_", " ").title(), f"{v:.2f}")
            i += 1

st.divider()

# -------------------------
# Scenarios selector
# -------------------------
st.subheader("Counterfactual Scenarios")

scenarios = data.get("scenarios", {})
if not scenarios:
    st.warning("No scenarios found in the JSON.")
    st.stop()

scenario_keys = list(scenarios.keys())
scenario_name = st.selectbox("Select scenario", scenario_keys, index=0)
res = scenarios.get(scenario_name, {})

# header + description if present
label = res.get("label", scenario_name)
st.markdown(f"### {label}")
if "desc" in res:
    st.caption(res["desc"])

# -------------------------
# Fleet-level metrics panel
# -------------------------
fleet = res.get("fleet", {})

def fmt_interval(lo, hi):
    try:
        return f"[{float(lo):.2f}, {float(hi):.2f}] kg CO₂e"
    except Exception:
        return "—"

col1, col2, col3 = st.columns(3)
delta_med = fleet.get("delta_median_kg", fleet.get("delta_kg", None))
delta_lo90 = fleet.get("delta_lo90_kg", fleet.get("delta_lo90", None))
delta_hi90 = fleet.get("delta_hi90_kg", fleet.get("delta_hi90", None))
delta_pct = fleet.get("delta_pct", fleet.get("relative_change_pct", None))

if delta_med is None:
    col1.metric("Fleet ΔPCF (median)", "—")
else:
    col1.metric("Fleet ΔPCF (median)", f"{delta_med:.2f} kg CO₂e")

col2.metric("Fleet 90% CI", fmt_interval(delta_lo90, delta_hi90))

if delta_pct is None:
    col3.metric("Relative change", "—")
else:
    try:
        col3.metric("Relative change", f"{float(delta_pct):.1f} %")
    except Exception:
        col3.metric("Relative change", str(delta_pct))

st.divider()

# -------------------------
# Bar chart (prefer plots.py)
# -------------------------
# Try to use plots.py's chart if available, otherwise draw a fallback chart
plot_fig = None
plot_used = None
try:
    import plots  # user-provided plotting helpers
    # attempt to find a sensible function name
    for fn_name in ("plot_counterfactual_bar", "plot_delta_bar", "plot_scenario_bar", "counterfactual_bar"):
        plot_fn = getattr(plots, fn_name, None)
        if callable(plot_fn):
            # try a few candidate argument patterns
            for arg in (res, fleet, res.get("per_device", None)):
                try:
                    plot_fig = plot_fn(arg)
                    plot_used = f"plots.{fn_name}"
                    break
                except Exception:
                    plot_fig = None
            if plot_fig is not None:
                break
except Exception:
    plot_fig = None

if plot_fig is not None:
    # Accept both matplotlib and plotly objects
    if hasattr(plot_fig, "to_html") or "plotly" in str(type(plot_fig)).lower():
        st.plotly_chart(plot_fig, use_container_width=True)
    else:
        st.pyplot(plot_fig, bbox_inches="tight")
else:
    # Fallback: draw bar chart from per-device deltas if present, else show a summary bar
    per_device = res.get("per_device", []) or []
    if per_device:
        df_pd = pd.DataFrame(per_device)
        # detect delta column
        delta_col_candidates = [
            "delta_median_kg",
            "delta_kg",
            "delta",
            "delta_median",
            "device_delta_kg",
        ]
        delta_col = next((c for c in delta_col_candidates if c in df_pd.columns), None)
        name_col_candidates = ["model", "model_key", "device", "name"]
        name_col = next((c for c in name_col_candidates if c in df_pd.columns), None)

        if delta_col is None:
            # try to infer numeric delta-like column
            numeric_cols = df_pd.select_dtypes("number").columns.tolist()
            inferred = [c for c in numeric_cols if "delta" in c or "change" in c or "diff" in c]
            delta_col = inferred[0] if inferred else (numeric_cols[0] if numeric_cols else None)

        if name_col is None:
            df_pd = df_pd.reset_index().rename(columns={"index": "model"})
            name_col = "model"

        if delta_col is not None:
            df_plot = df_pd[[name_col, delta_col]].copy()
            df_plot = df_plot.dropna(subset=[delta_col])
            # sort and limit to top 25 for readability
            df_plot[delta_col] = pd.to_numeric(df_plot[delta_col], errors="coerce")
            df_plot = df_plot.sort_values(by=delta_col, key=lambda s: np.abs(s), ascending=False).head(25)

            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(df_plot[name_col].astype(str), df_plot[delta_col], zorder=3)
            ax.axvline(0, color="k", linewidth=0.8)
            ax.set_xlabel("ΔPCF (kg CO₂e)")
            ax.set_title(f"Per-device ΔPCF — {label}")
            ax.invert_yaxis()
            ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Per-device delta numeric column not detected; cannot draw bar chart.")
    else:
        # No per-device data -> show simple single bar for fleet delta
        if delta_med is not None:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar([0], [delta_med], color="#5b9bd5", zorder=3)
            ax.set_xticks([0])
            ax.set_xticklabels([label], rotation=45, ha="right")
            ax.set_ylabel("Fleet ΔPCF (kg CO₂e)")
            ax.axhline(0, color="k", linewidth=0.7)
            ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No fleet or per-device delta available to plot.")

st.divider()

# -------------------------
# Per-device impact viewer
# -------------------------
st.subheader("Per-device Impact Viewer")
per_device = res.get("per_device", None)

if not per_device:
    st.info("No device-level deltas included in this scenario.")
else:
    df_dev = pd.DataFrame(per_device)

    # Normalize expected column names
    # Common columns: model/model_key/name, delta_median_kg, delta_lo90_kg, delta_hi90_kg, delta_pct
    col_map = {}
    for candidate in ("model", "model_key", "device", "name"):
        if candidate in df_dev.columns:
            col_map["model_col"] = candidate
            break
    if "model_col" not in col_map:
        # fallback: use index as model id
        df_dev = df_dev.reset_index().rename(columns={"index": "model"})
        col_map["model_col"] = "model"

    # ensure numeric types
    for c in ["delta_median_kg", "delta_kg", "delta", "delta_lo90_kg", "delta_hi90_kg", "delta_lo90", "delta_hi90", "delta_pct"]:
        if c in df_dev.columns:
            df_dev[c] = pd.to_numeric(df_dev[c], errors="coerce")

    # compute in-90% CI indicator if lo/hi present
    # detect columns safely
    lo_candidates = ["delta_lo90_kg", "delta_lo90", "delta_lo"]
    hi_candidates = ["delta_hi90_kg", "delta_hi90", "delta_hi"]
    med_candidates = ["delta_median_kg", "delta_kg", "delta_median", "delta"]

    lo_col = next((c for c in lo_candidates if c in df_dev.columns), None)
    hi_col = next((c for c in hi_candidates if c in df_dev.columns), None)
    med_col = next((c for c in med_candidates if c in df_dev.columns), None)

    # compute CI indicator only if columns actually exist
    if med_col and lo_col and hi_col:
        df_dev["in_90pct"] = (
            (df_dev[med_col] >= df_dev[lo_col]) &
            (df_dev[med_col] <= df_dev[hi_col])
        )
    else:
        df_dev["in_90pct"] = False

    # display interactive table with sorting, search, and download
    st.markdown("**Device-level deltas (table)**")
    display_cols = []
    display_cols.append(col_map["model_col"])
    for c in [med_col, lo_col, hi_col, "delta_pct"]:
        if c and c in df_dev.columns:
            display_cols.append(c)
    # add any other numeric columns that might be useful
    numeric_extra = [c for c in df_dev.select_dtypes("number").columns if c not in display_cols]
    display_cols += numeric_extra[:3]  # keep table compact

    st.dataframe(df_dev[display_cols + ["in_90pct"]].rename(columns={col_map["model_col"]: "Model"}), use_container_width=True)

    csv_bytes = df_dev.to_csv(index=False).encode("utf-8")
    st.download_button("Download per-device CSV", data=csv_bytes, file_name=f"{scenario_name}_per_device_deltas.csv", mime="text/csv")

    # device detail viewer
    st.markdown("**Inspect a device**")
    model_list = df_dev[col_map["model_col"]].astype(str).tolist()
    sel = st.selectbox("Choose device", ["(none)"] + model_list)
    if sel and sel != "(none)":
        row = df_dev[df_dev[col_map["model_col"]].astype(str) == sel].iloc[0]
        st.write("Model:", sel)
        # show key numbers
        if med_col in row.index:
            st.metric("ΔPCF (median)", f"{row.get(med_col, np.nan):.2f} kg" if pd.notna(row.get(med_col)) else "—")
        if lo_col and hi_col and pd.notna(row.get(lo_col)) and pd.notna(row.get(hi_col)):
            st.write("90% CI:", fmt_interval(row[lo_col], row[hi_col]))
        if "delta_pct" in row.index and pd.notna(row.get("delta_pct")):
            st.metric("Relative change", f"{row.get('delta_pct'):.1f} %")

        # simple per-device bar visual: median with CI
        if med_col:
            fig, ax = plt.subplots(figsize=(6, 1.6))
            m = row.get(med_col, 0.0)
            lo = row.get(lo_col, m - 0.01) if lo_col else m
            hi = row.get(hi_col, m + 0.01) if hi_col else m
            ax.barh([0], [m], height=0.6, zorder=3)
            ax.hlines(0, lo, hi, color="k", linewidth=3, zorder=4)
            ax.set_yticks([])
            ax.set_xlabel("ΔPCF (kg CO₂e)")
            ax.set_title(f"{sel} — ΔPCF median ± 90% CI")
            ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)

st.divider()
st.caption("Counterfactual results loaded from `results/counterfactual_results.json`. If plots.py exists with a compatible function, it will be used for the main chart; otherwise a sensible fallback plot is drawn.")