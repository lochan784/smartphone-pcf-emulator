import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_forensics():
    path = os.path.join(ROOT, "results", "error_forensics_full.csv")
    if not os.path.exists(path):
        st.warning("Forensics file not found. Run forensic_error_analysis.py first.")
        return None
    return pd.read_csv(path)

@st.cache_data
def load_posterior_predictions():
    path = os.path.join(ROOT, "data", "processed", "posterior_predictions_gold.csv")
    if not os.path.exists(path):
        st.warning("Posterior predictions file not found.")
        return None
    return pd.read_csv(path)

@st.cache_data
def load_counterfactuals():
    path = os.path.join(ROOT, "results", "counterfactual_results.json")
    if not os.path.exists(path):
        st.warning("Counterfactual results not found.")
        return None
    with open(path) as f:
        return json.load(f)

# ... add more loaders as needed