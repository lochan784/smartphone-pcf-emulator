import os
import json
import streamlit as st
import pandas as pd

WEBAPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(WEBAPP)

RESULTS = os.path.join(ROOT, "results")

st.set_page_config(page_title="Final Report", layout="wide")
st.header("✅ Final Validation Report")

RESULT_PATH = os.path.join(RESULTS, "final_report.json")
st.caption(f"Loaded report from: {RESULT_PATH}")

with open(RESULT_PATH, "r") as f:
    report = json.load(f)

gates = report.get("gates", {})
if not gates:
    st.warning("No `gates` key found in final_report.json.")
    st.write(report)
    st.stop()

# Normalize gates into rows
rows = []

for gate_name, gate_val in gates.items():

    row = {"gate": gate_name, "raw": gate_val}

    # ------------------------------------
    # Case 1: gate stored as dictionary
    # ------------------------------------
    if isinstance(gate_val, dict):

        status = (
            gate_val.get("status")
            or gate_val.get("result")
            or gate_val.get("pass")
            or gate_val.get("passed")   # FIX added for your JSON
        )

        value = gate_val.get("value", gate_val.get("metric"))
        threshold = gate_val.get("threshold")

        # AUTO-EVALUATE if status missing but value + threshold exist
        if status is None and value is not None and threshold is not None:
            try:
                status = "passed" if float(value) >= float(threshold) else "failed"
            except Exception:
                status = None

        if isinstance(status, bool):
            row["status"] = "passed" if status else "failed"

        elif isinstance(status, str):
            s = status.lower()

            if s in ("passed", "pass", "ok", "true", "success"):
                row["status"] = "passed"

            elif s in ("failed", "fail", "false", "error"):
                row["status"] = "failed"

            else:
                row["status"] = None

        else:
            row["status"] = None

        row["value"] = value
        row["threshold"] = threshold
        row["details"] = gate_val.get("details") or gate_val.get("note") or ""

    # ------------------------------------
    # Case 2: primitive gate values
    # ------------------------------------
    else:

        if isinstance(gate_val, bool):

            row["status"] = "passed" if gate_val else "failed"

        else:

            s = str(gate_val).lower()

            if s in ("passed", "pass", "ok", "true", "success"):
                row["status"] = "passed"

            elif s in ("failed", "fail", "false", "error"):
                row["status"] = "failed"

            else:
                row["status"] = None
                row["value"] = gate_val
                row["details"] = ""

    rows.append(row)

# Display summary header
total = len(rows)
passed = sum(1 for r in rows if r.get("status") == "passed")
pct = int(round(100 * passed / total)) if total > 0 else 0

col_summary, col_progress = st.columns([3, 1])
with col_summary:
    if passed == total:
        st.success(f"PIPELINE COMPLETE — ALL GATES PASSED ({passed}/{total})")
    else:
        st.warning(f"{passed}/{total} gates passed — review failures below.")
with col_progress:
    st.metric("Pass rate", f"{pct}%")

st.divider()

# Table-like display with per-gate rows (icons + values + expandable details)
for r in rows:
    gate = r["gate"]
    status = r.get("status")
    value = r.get("value", None)
    threshold = r.get("threshold", None)
    details = r.get("details", "")

    if status == "passed":
        icon = "✅"
        status_text = "PASSED"
        box_style = "background-color:#000000; padding:6px; border-radius:6px;"
    elif status == "failed":
        icon = "❌"
        status_text = "FAILED"
        box_style = "background-color:#000000; padding:6px; border-radius:6px;"
    else:
        icon = "ℹ️"
        status_text = "INFO"
        box_style = "background-color:#000000; padding:6px; border-radius:6px;"

    c1, c2, c3, c4 = st.columns([3, 1, 1, 4])

    with c1:
        st.markdown(f"**{gate}**")

    with c2:
        st.markdown(
            f"<div style='{box_style} text-align:center'>{icon} <strong>{status_text}</strong></div>",
            unsafe_allow_html=True,
        )

    with c3:
        if value is not None or threshold is not None:
            vtxt = ""
            if value is not None:
                try:
                    vtxt += f"Value: {float(value):.3f}"
                except Exception:
                    vtxt += f"Value: {value}"
            if threshold is not None:
                try:
                    vtxt += f"  |  Threshold: {float(threshold):.3f}"
                except Exception:
                    vtxt += f"  |  Threshold: {threshold}"
            st.markdown(vtxt)
        else:
            st.write("")

    with c4:
        with st.expander("Details", expanded=False):
            if details:
                if isinstance(details, (list, dict)):
                    st.json(details)
                else:
                    st.write(details)
            else:
                st.write("Raw:")
                st.json(r["raw"])

st.divider()

st.subheader("Report JSON")

col_view, col_download = st.columns([3, 1])

with col_view:
    if st.checkbox("Show raw final_report.json", value=False):
        st.json(report)

with col_download:
    json_bytes = json.dumps(report, indent=2).encode("utf-8")
    st.download_button(
        "Download final_report.json",
        data=json_bytes,
        file_name="final_report.json",
        mime="application/json",
    )

if passed == total:
    st.success(f"Overall result: PASSED ({passed}/{total})")
else:
    st.error(f"Overall result: FAILED ({passed}/{total}) — please inspect failing gates above.")