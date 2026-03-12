import streamlit as st
import utils
import os

st.header("🌍 Smartphone Product Carbon Footprint Emulator")

st.markdown("""
This interactive dashboard presents a **hybrid lifecycle emulator** that combines:
- A **physics-based baseline** (additive component model)
- **Hierarchical Bayesian calibration** with literature-anchored priors
- **Conformal prediction** (Jackknife+/LOOCV) for finite-sample coverage guarantees
- **Policy simulation** and **cost-constrained Pareto optimization**

### Key Results
- **MAE (LOOCV):** 5.66 kg CO₂e (Bayesian) / 5.01 kg (Ridge)
- **90% conformal coverage:** 92.3% on 13 verified devices
- **Most impactful lever:** Grid decarbonisation (–6.1% fleet average)
- **Lifetime extension:** +12.2% under current grid (conditional on grid mix)

### Pipeline
""")

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_PATH = os.path.join(BASE_DIR, "webapp", "assets", "conformal_coverage_by_tertile.png")

st.image(
    IMG_PATH,
    caption="Conformal coverage by emission tertile",
    width=600
)