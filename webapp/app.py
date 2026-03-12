import streamlit as st
import os

st.set_page_config(
    page_title="Smartphone PCF Emulator",
    page_icon="📱",
    layout="wide"
)

st.title("🌍 Smartphone Product Carbon Footprint Emulator")

st.markdown("""
This interactive dashboard presents a **hybrid lifecycle emulator** that combines:

- A **physics-based baseline** (additive component model)  
- **Hierarchical Bayesian calibration** with literature-anchored priors  
- **Conformal prediction** (Jackknife+/LOOCV) for finite-sample guarantees  
- **Policy simulation** and **cost-constrained Pareto optimization**

### Key Results
- **MAE (LOOCV):** 5.66 kg CO₂e (Bayesian) / 5.01 kg (Ridge)
- **90% conformal coverage:** 92.3% on 13 verified devices
- **Most impactful lever:** Grid decarbonisation (–6.1% fleet average)
- **Lifetime extension:** +12.2% under current grid mix
""")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(BASE_DIR, "assets", "conformal_coverage_by_tertile.png")

st.image(
    IMG_PATH,
    caption="Conformal coverage by emission tertile",
    width=650
)

st.info("Use the sidebar to explore diagnostics, predictions, counterfactual scenarios, and optimization.")