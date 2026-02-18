import pandas as pd
import numpy as np
import os
from datetime import datetime

RAW_PATH = "data/raw/cleaned_data.csv"
PROCESSED_PATH = "data/processed/smartphones_structured.csv"

df = pd.read_csv(RAW_PATH)

# --------------------------
# Basic Cleaning
# --------------------------

# Ensure numeric types
numeric_cols = [
    "battery_mah",
    "screen_size_in",
    "ram_gb",
    "storage_gb",
    "clock_ghz",
    "rear_camera_count",
    "rear_camera_max_mp",
    "front_camera_mp",
    "refresh_rate_hz"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# --------------------------
# Brand extraction
# --------------------------
df["brand"] = df["model"].str.split().str[0].str.lower()

# --------------------------
# Clean Display Area
# --------------------------

median_screen = df["screen_size_in"].median()
df["screen_size_in"] = df["screen_size_in"].fillna(median_screen)

aspect_ratio = 19.5 / 9
diag_cm = df["screen_size_in"] * 2.54

height_cm = diag_cm / np.sqrt(aspect_ratio**2 + 1)
width_cm = height_cm * aspect_ratio

df["display_area_cm2"] = height_cm * width_cm

# --------------------------
# Clean Battery Values
# --------------------------

# Cap extreme values
df["battery_mah"] = df["battery_mah"].clip(upper=7000)

# Impute missing values with median
median_battery = df["battery_mah"].median()
df["battery_mah"] = df["battery_mah"].fillna(median_battery)

# --------------------------
# Battery Energy (kWh)
# --------------------------

df["battery_kwh"] = (
    df["battery_mah"] * 3.7 / 1_000_000
)

# --------------------------
# Battery Carbon (kg CO2e)
# --------------------------

BATTERY_EF = 150  # kg CO2e per kWh

df["carbon_battery"] = df["battery_kwh"] * BATTERY_EF

# --------------------------
# Use Phase Carbon (3-Year Lifetime)
# --------------------------

LIFETIME_YEARS = 3
GRID_EF = 0.45  # kg CO2e per kWh (global mid-range)

DAILY_CHARGE_FACTOR = 1.15

df["annual_kwh"] = df["battery_kwh"] * 365 * DAILY_CHARGE_FACTOR

df["lifetime_kwh"] = df["annual_kwh"] * LIFETIME_YEARS

df["carbon_use"] = df["lifetime_kwh"] * GRID_EF

# --------------------------
# Display Carbon (kg CO2e)
# --------------------------

DISPLAY_EF_PER_CM2 = 0.03  # kg CO2e per cm2 (literature mid-range)

df["carbon_display"] = df["display_area_cm2"] * DISPLAY_EF_PER_CM2

# --------------------------
# Estimated Mass (Physically Tuned Proxy)
# --------------------------

df["estimated_mass_g"] = (
    110
    + 0.012 * df["battery_mah"].fillna(0)
    + 0.3 * df["display_area_cm2"].fillna(0)
)

df["estimated_mass_g"] = df["estimated_mass_g"].clip(130, 240)
df["mass_estimated"] = True

# --------------------------
# Performance Index
# --------------------------

# Cap unrealistic values
df["clock_ghz"] = df["clock_ghz"].clip(upper=3.5)
df["ram_gb"] = df["ram_gb"].clip(upper=24)

# Impute missing
median_clock = df["clock_ghz"].median()
median_ram = df["ram_gb"].median()

df["clock_ghz"] = df["clock_ghz"].fillna(median_clock)
df["ram_gb"] = df["ram_gb"].fillna(median_ram)

# --------------------------
# Enhanced Performance Proxy (Semiconductor Intensity Proxy)
# --------------------------

CORE_WEIGHT = 0.8
REFRESH_WEIGHT = 0.02

df["core_type_numeric"] = df["core_type"].str.extract(r'(\d+)').astype(float)
df["core_type_numeric"] = df["core_type_numeric"].fillna(8)

df["performance_index"] = (
    df["clock_ghz"] * df["ram_gb"]
    + CORE_WEIGHT * df["core_type_numeric"]
    + REFRESH_WEIGHT * df["refresh_rate_hz"].fillna(60)
)

# --------------------------
# Reconstruct Storage (Percentile Based)
# --------------------------

p33 = df["performance_index"].quantile(0.33)
p66 = df["performance_index"].quantile(0.66)

df["storage_gb"] = 128  # default

df.loc[df["performance_index"] <= p33, "storage_gb"] = 64
df.loc[df["performance_index"] > p66, "storage_gb"] = 256

df["storage_reconstructed"] = True

# --------------------------
# Total Camera MP
# --------------------------

df["rear_camera_max_mp"] = df["rear_camera_max_mp"].clip(upper=150)
df["front_camera_mp"] = df["front_camera_mp"].clip(upper=60)

df["total_camera_mp"] = (
    df["rear_camera_max_mp"].fillna(0)
    + df["front_camera_mp"].fillna(0)
)

# --------------------------
# Electronics / Semiconductor Carbon (kg CO2e)
# --------------------------

ELECTRONICS_BASE = 28
PERF_COEF = 0.18
STORAGE_COEF = 0.035
CAMERA_COEF = 0.02

df["carbon_electronics"] = (
    ELECTRONICS_BASE
    + PERF_COEF * df["performance_index"]
    + STORAGE_COEF * df["storage_gb"].fillna(0)
    + CAMERA_COEF * df["total_camera_mp"]
)

# --------------------------
# Structural Materials Carbon
# --------------------------

MATERIAL_EF = 20  # kg CO2e per kg material

df["carbon_materials"] = (
    (df["estimated_mass_g"] / 1000) * MATERIAL_EF
)

# --------------------------
# Assembly & Transport Carbon
# --------------------------

df["carbon_assembly"] = 3
df["carbon_transport"] = 2

# --------------------------
# Base Total PCF (No Noise)
# --------------------------

df["pcf_base"] = (
    df["carbon_battery"]
    + df["carbon_display"]
    + df["carbon_electronics"]
    + df["carbon_materials"]
    + df["carbon_assembly"]
    + df["carbon_transport"]
    + df["carbon_use"]
)

df["synthetic_label"] = True

# --------------------------
# Controlled Measurement Noise
# --------------------------

np.random.seed(42)
NOISE_STD = 1.5

df["pcf_kg_co2e"] = (
    df["pcf_base"]
    + np.random.normal(0, NOISE_STD, len(df))
)

# --------------------------
# --------------------------
# Component Contribution Percentages
# --------------------------

df["pct_use"] = df["carbon_use"] / df["pcf_base"]
df["pct_electronics"] = df["carbon_electronics"] / df["pcf_base"]
df["pct_battery"] = df["carbon_battery"] / df["pcf_base"]
df["pct_display"] = df["carbon_display"] / df["pcf_base"]
df["pct_materials"] = df["carbon_materials"] / df["pcf_base"]
df["pct_other"] = (
    (df["carbon_assembly"] + df["carbon_transport"])
    / df["pcf_base"]
)

# Integrity check
df["pct_sum"] = (
    df["pct_use"]
    + df["pct_electronics"]
    + df["pct_battery"]
    + df["pct_display"]
    + df["pct_materials"]
    + df["pct_other"]
)

# --------------------------
# Data Tier
# --------------------------

df["data_tier"] = "synthetic"

# --------------------------
# OOD Flag (High-Performance Devices)
# --------------------------

df["is_flagship"] = (
    (df["price"] > df["price"].quantile(0.75))
    & (df["performance_index"] > df["performance_index"].quantile(0.75))
)

# --------------------------
# Metadata
# --------------------------

df["data_processed_year"] = datetime.now().year

os.makedirs("data/processed", exist_ok=True)
df.to_csv(PROCESSED_PATH, index=False)
