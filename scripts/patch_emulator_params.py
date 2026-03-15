#!/usr/bin/env python3
"""
scripts/patch_emulator_params.py
==================================
Minimal patch for your existing emulator / feature script.

Instead of replacing your emulator, this provides a drop-in
parameter loader that you can import at the top of any existing
script to replace hard-coded constants.

BEFORE (your current code):
    BATTERY_EF = 150
    GRID_EF = 0.45
    MATERIAL_EF = 20

AFTER (add 3 lines at the top of your script):
    from scripts.patch_emulator_params import load_ef_params
    EF = load_ef_params()
    # Then use: EF['battery_ef'], EF['grid_ef'], EF['material_ef'] etc.

Usage in your compute_pcf_base or equivalent function:
    C_bat  = df["battery_kwh"]              * EF['battery_ef']
    C_disp = df["display_area_cm2"]         * EF['display_ef']
    C_mat  = (df["estimated_mass_g"]/1000)  * EF['material_ef']
    C_use  = df["annual_kwh"] * EF['grid_ef'] * EF['lifetime_years']
    C_elec = df["performance_index"]        * EF['semiconductor_intensity']
    C_trans = EF['assembly_ef'] + EF['transport_ef']
    PCF = C_bat + C_disp + C_mat + C_use + C_elec + C_trans
"""

import os
import pandas as pd

_DEFAULT_PATH = "data/processed/prior_params.csv"

# Fallbacks — only used if CSV not found (avoids breaking existing code)
_FALLBACKS = {
    "battery_ef":             120.0,
    "display_ef":             0.03,
    "semiconductor_intensity": 0.8,
    "material_ef":            18.0,
    "assembly_ef":            3.5,
    "transport_ef":           2.0,
    "lifetime_years":         3.0,
    "grid_ef":                0.45,
}


def load_ef_params(path: str = _DEFAULT_PATH, region: str = "global") -> dict:
    """
    Load emission factor means from prior_params.csv.
    Returns dict keyed by param name → float (prior mean).

    Parameters
    ----------
    path   : path to prior_params.csv
    region : 'global' | 'us' | 'uk' — selects grid_ef variant

    Returns
    -------
    dict with keys matching the column names in your feature pipeline:
        battery_ef, display_ef, semiconductor_intensity,
        material_ef, assembly_ef, transport_ef, lifetime_years, grid_ef
    """
    if not os.path.exists(path):
        print(f"⚠️  prior_params.csv not found at '{path}'. Using hard-coded fallbacks.")
        print("   Run scripts/03_prior_assembly.py first.")
        return _FALLBACKS.copy()

    df = pd.read_csv(path).set_index("param")

    def get(param, fallback):
        if param in df.index and pd.notnull(df.loc[param, "prior_mean"]):
            return float(df.loc[param, "prior_mean"])
        return fallback

    # Select grid EF by region
    grid_param = {"us": "grid_ef_us", "uk": "grid_ef_uk"}.get(region, "grid_ef")

    params = {
        "battery_ef":              get("battery_ef",             _FALLBACKS["battery_ef"]),
        "display_ef":              get("display_ef",             _FALLBACKS["display_ef"]),
        "semiconductor_intensity": get("semiconductor_intensity",_FALLBACKS["semiconductor_intensity"]),
        "material_ef":             get("material_ef",            _FALLBACKS["material_ef"]),
        "assembly_ef":             get("assembly_ef",            _FALLBACKS["assembly_ef"]),
        "transport_ef":            get("transport_ef",           _FALLBACKS["transport_ef"]),
        "lifetime_years":          get("lifetime_years",         _FALLBACKS["lifetime_years"]),
        "grid_ef":                 get(grid_param,               _FALLBACKS["grid_ef"]),
    }

    return params


def compute_pcf_base(df, ef: dict = None) -> "pd.Series":
    """
    Compute baseline PCF using YOUR exact column names.
    Drop-in replacement for your existing compute_pcf_base function.

    Columns required in df:
        battery_kwh, display_area_cm2, estimated_mass_g,
        performance_index, annual_kwh

    Optional: lifetime_years column (overrides ef['lifetime_years'])
    """
    import pandas as pd
    if ef is None:
        ef = load_ef_params()

    lifetime = (
        df["lifetime_years"]
        if "lifetime_years" in df.columns
        else ef["lifetime_years"]
    )

    C_bat   = df["battery_kwh"]             * ef["battery_ef"]
    C_disp  = df["display_area_cm2"]        * ef["display_ef"]
    C_mat   = (df["estimated_mass_g"] / 1000.0) * ef["material_ef"]
    C_use   = df["annual_kwh"] * ef["grid_ef"] * lifetime
    C_elec  = df["performance_index"]       * ef["semiconductor_intensity"]
    C_trans = ef["assembly_ef"] + ef["transport_ef"]

    return C_bat + C_disp + C_mat + C_use + C_elec + C_trans


def get_prior_sd(param: str, path: str = _DEFAULT_PATH) -> float:
    """Get prior SD for a parameter (used in PyMC model definition)."""
    if not os.path.exists(path):
        return 1.0
    df = pd.read_csv(path).set_index("param")
    if param in df.index and pd.notnull(df.loc[param, "prior_sd"]):
        return float(df.loc[param, "prior_sd"])
    return 1.0


if __name__ == "__main__":
    ef = load_ef_params()
    print("Loaded emission factor parameters:")
    print("-" * 45)
    for k, v in ef.items():
        print(f"  {k:<28}: {v}")
    print("\nThese replace the hard-coded constants:")
    print("  BATTERY_EF  = 150  →", ef["battery_ef"])
    print("  GRID_EF     = 0.45 →", ef["grid_ef"])
    print("  MATERIAL_EF = 20   →", ef["material_ef"])
