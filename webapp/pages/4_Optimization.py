import os
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

WEBAPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(WEBAPP)

RESULTS = os.path.join(ROOT, "results")

st.set_page_config(page_title="Optimization", layout="wide")

st.header("⚖️ Cost–Carbon Pareto Optimization")

FRONTIER_PATH = os.path.join(RESULTS, "pareto_frontier.csv")
OPT_PATH = os.path.join(RESULTS, "optimization_results.json")

if not os.path.exists(FRONTIER_PATH):
    st.error(f"{FRONTIER_PATH} not found. Run the optimization pipeline first.")
    st.stop()

df = pd.read_csv(FRONTIER_PATH)

# ------------------------------------------------
# Pareto Frontier Plot (Fig.7)
# ------------------------------------------------

st.subheader("Pareto Frontier")

fig, ax = plt.subplots(figsize=(7,4))

x = df["budget"]
y = df["pcf_reduction_kg"]

ax.plot(x, y, marker="o", linewidth=2)

ax.set_xlabel("Budget ($/device)")
ax.set_ylabel("PCF Reduction (kg CO₂e)")
ax.set_title("Cost–Carbon Pareto Frontier")

ax.grid(ls=":", lw=0.5)

st.pyplot(fig)

st.divider()

# ------------------------------------------------
# Budget selector
# ------------------------------------------------

st.subheader("Budget-Constrained Recommendation")

budget = st.slider(
    "Available Budget ($/device)",
    int(df["budget"].min()),
    int(df["budget"].max()),
    int(df["budget"].median()),
)

# choose best feasible solution
best = df[df["budget"] <= budget].iloc[-1]

col1, col2, col3 = st.columns(3)

col1.metric(
    "ΔPCF (median)",
    f"{best['pcf_reduction_kg']:.2f} kg CO₂e"
)

col2.metric(
    "Optimised Fleet PCF",
    f"{best['optimised_pcf']:.2f} kg CO₂e"
)

col3.metric(
    "Total Cost",
    f"${best['total_cost']:.2f} / device"
)

st.divider()

# ------------------------------------------------
# Selected levers
# ------------------------------------------------

st.subheader("Recommended Decarbonization Levers")

levers = best.get("selected_levers")

if isinstance(levers, str):

    try:
        levers = json.loads(levers)
    except:
        pass

if isinstance(levers, list):

    lever_df = pd.DataFrame({"Lever": levers})

    st.dataframe(
        lever_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.code(levers)

# ------------------------------------------------
# Full frontier solutions (Table XII style)
# ------------------------------------------------

st.subheader("All Optimal Lever Combinations")

display_cols = [
    "budget",
    "pcf_reduction_kg",
    "optimised_pcf",
    "total_cost",
    "selected_levers",
]

available_cols = [c for c in display_cols if c in df.columns]

table_df = df[available_cols].copy()

table_df = table_df.rename(
    columns={
        "budget":"Budget ($)",
        "pcf_reduction_kg":"ΔPCF (kg)",
        "optimised_pcf":"Optimised PCF (kg)",
        "total_cost":"Cost ($)",
        "selected_levers":"Lever Set"
    }
)

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ------------------------------------------------
# Inspect a specific solution
# ------------------------------------------------

st.subheader("Inspect Solution")

solution_index = st.selectbox(
    "Select frontier point",
    df.index.tolist()
)

row = df.loc[solution_index]

c1,c2,c3 = st.columns(3)

c1.metric("Budget", f"${row['budget']:.2f}")
c2.metric("ΔPCF", f"{row['pcf_reduction_kg']:.2f} kg")
c3.metric("Cost", f"${row['total_cost']:.2f}")

st.write("Lever combination:")

st.code(row["selected_levers"])

st.caption(
    "Pareto frontier shows the optimal trade-off between decarbonization impact and implementation cost. "
    "Each point represents the lowest-cost lever combination achieving that level of PCF reduction."
)