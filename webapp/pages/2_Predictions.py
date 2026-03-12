import streamlit as st
import numpy as np
import pandas as pd
import os

WEBAPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(WEBAPP)

DATA = os.path.join(ROOT, "data")

st.header("📊 Posterior Predictions vs Declared PCF")

path = os.path.join(DATA, "processed", "posterior_predictions_gold.csv")

df = pd.read_csv(path)

# 2) Basic column checks
required = ["model", "declared_pcf", "pred_median", "pred_lo90", "pred_hi90"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Posterior predictions file is missing required columns: {missing}")
    st.stop()

# 3) Narrow to the 13 verified / gold devices if possible
verified_filters = ["verified", "is_verified", "gold", "is_gold", "in_benchmark", "matched"]
filtered = None
for col in verified_filters:
    if col in df.columns:
        vals = df[col]
        mask = vals.isin([1, True, "1", "True", "true", "yes", "Yes"]) | vals.astype(str).str.lower().isin(["1", "true", "yes"])
        filtered = df[mask]
        if len(filtered) > 0:
            df = filtered
            break

# If still more than 13 rows
if len(df) > 13:
    if "is_gold" in df.columns:
        tmp = df[df["is_gold"].astype(bool)]
        if 0 < len(tmp) <= 13:
            df = tmp
    if len(df) > 13:
        st.warning(f"Found {len(df)} rows; showing the first 13 devices (if you expect a different subset, add a 'verified' or 'is_gold' column).")
        df = df.head(13)

if df.empty:
    st.warning("No verified devices found in the posterior predictions file.")
    st.stop()

# 4) Ensure numeric types
for c in ["declared_pcf", "pred_median", "pred_lo90", "pred_hi90"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

if "in_90pct_ci" not in df.columns:
    df["in_90pct_ci"] = (df["declared_pcf"] >= df["pred_lo90"]) & (df["declared_pcf"] <= df["pred_hi90"])
else:
    df["in_90pct_ci"] = df["in_90pct_ci"].astype(str).str.lower().isin(["1", "true", "yes"]) | df["in_90pct_ci"].astype(bool)

# compute error columns
df["error_abs"] = (df["declared_pcf"] - df["pred_median"]).abs()
df["error_signed"] = df["declared_pcf"] - df["pred_median"]

# 5) Create display dataframe
display_cols = [
    ("model", "Model"),
    ("declared_pcf", "Declared (kg)"),
    ("pred_median", "Pred Median (kg)"),
    ("pred_lo90", "Pred Lo90 (kg)"),
    ("pred_hi90", "Pred Hi90 (kg)"),
    ("error_signed", "Error (Declared − Pred)"),
    ("in_90pct_ci", "In 90% CI"),
]

df_display = df[[c for c,_ in display_cols]].copy()
df_display.columns = [label for _, label in display_cols]

# FIX: convert checkbox boolean → ✔ / ✖
df_display["In 90% CI"] = df_display["In 90% CI"].map(lambda x: "✔" if x else "✖")

# Round numeric columns
num_cols = ["Declared (kg)", "Pred Median (kg)", "Pred Lo90 (kg)", "Pred Hi90 (kg)", "Error (Declared − Pred)"]
for c in num_cols:
    if c in df_display.columns:
        df_display[c] = df_display[c].map(lambda v: np.nan if pd.isna(v) else round(float(v), 2))

# 6) Build pandas Styler
def _ci_cell_style(val):
    if pd.isna(val):
        return "background-color: #000000; color: #856404"
    return "background-color: #000000; color: #0b6623" if val == "✔" else "background-color: #ffecec; color: #9b111e"

def _row_highlight_outside_ci(row):
    in_ci = row["In 90% CI"]
    if in_ci == "✖":
        return ["font-weight: bold; color: #9b111e" if col in num_cols else "" for col in row.index]
    return ["" for _ in row.index]

styler = (
    df_display.style
    .set_properties(subset=["Model"], **{"text-align": "left"})
    .format({c: "{:.2f}" for c in num_cols if c in df_display.columns}, na_rep="—")
    .applymap(_ci_cell_style, subset=["In 90% CI"])
    .apply(_row_highlight_outside_ci, axis=1)
    .set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("padding", "6px 8px")]},
        ]
    )
)

# 7) Display dataframe
st.dataframe(styler, use_container_width=True)

st.caption("90% posterior predictive intervals (Jackknife+ conformal). Rows where the declared value falls outside the 90% interval are highlighted.")

# 8) Download button
csv_bytes = df_display.to_csv(index=False).encode("utf-8")
st.download_button("Download displayed table (CSV)", data=csv_bytes, file_name="posterior_predictions_verified.csv", mime="text/csv")