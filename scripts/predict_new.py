"""
scripts/predict_new.py — Single-device PCF prediction using the trained Bayesian emulator.

Implements Equation 1 from the paper exactly:
    PCFbase,i = θb·BkWh + θd·Adisp + θm·Mkg + θg·Li·Eann + θs·Pidx + θa + θt

Feature conversions (Section IV-B):
    BkWh    = battery_mah * 3.7 / 1_000_000   (mAh → kWh at nominal voltage)
    Adisp   = from diagonal inches using 19.5:9 aspect ratio → cm²
    Mkg     = estimated_mass_g / 1000          (g → kg)
    Pidx    = performance_index (composite: clock speed, RAM, cores, refresh rate)
    Eann    = annual_kwh (estimated use-phase energy consumption)

Residual correction h(s; φ) uses phi (shape: n_resid=2) on:
    [performance_index, total_camera_mp]   (StandardScaler-normalised)

Usage:
    python scripts/predict_new.py --battery 4000 --display 6.5 --mass 180 \\
        --perf 27.5 --annual 6.5 --camera 48

    # Minimal (camera defaults to fleet median):
    python scripts/predict_new.py --battery 4000 --display 6.5 --mass 180 \\
        --perf 27.5 --annual 6.5
"""

import argparse
import sys
import os
import json
import math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join(ROOT, "models")
POSTERIOR = os.path.join(MODEL_DIR, "bayesian_emulator_posterior.nc")
SCALER    = os.path.join(MODEL_DIR, "resid_feature_scaler.joblib")
BRAND_IDX = os.path.join(MODEL_DIR, "brand_to_idx.joblib")

# Fleet statistics derived from data/processed/gold_holdout.csv and
# models/resid_feature_scaler.joblib (camera_mp from scaler feature-1 mean).
FLEET = {
    "performance_index": 27.535,  # mean from gold_holdout.csv; range 20.0–37.12
    "annual_kwh":         6.517,  # mean from gold_holdout.csv; range 4.97–7.77 kWh/yr
    "camera_mp":         65.252,  # scaler mean (resid_feature_scaler.joblib, feature 1)
    "lifetime_years":     3.0,    # prior mean from Table I
}


# ── Input conversions (Section IV-B) ─────────────────────────────────────────

def battery_to_kwh(mah: float, voltage: float = 3.7) -> float:
    """mAh → kWh using nominal 3.7 V (Section IV-B-b)."""
    return mah * voltage / 1_000_000


def display_to_cm2(diagonal_in: float, aspect: tuple = (19.5, 9)) -> float:
    """Diagonal inches → area cm² using fixed 19.5:9 AR (Section IV-B-a)."""
    aw, ah   = aspect
    ar       = math.sqrt(aw**2 + ah**2)
    w_in     = diagonal_in * aw / ar
    h_in     = diagonal_in * ah / ar
    area_in2 = w_in * h_in
    return area_in2 * (2.54 ** 2)


def estimate_mass(battery_mah: float, display_area_cm2: float) -> float:
    """
    Physics-tuned mass proxy (Section IV-B-c).
    Mirrors the heuristic in scripts/01_engineer_structured_features.py.
    Bounds: [130, 240] g.
    """
    mass = 0.022 * battery_mah + 0.35 * display_area_cm2 + 50.0
    return float(np.clip(mass, 130.0, 240.0))


# ── Posterior helpers ─────────────────────────────────────────────────────────

def _flat(post, name: str) -> np.ndarray:
    """Flatten (chains, draws[, ...]) → 1-D array."""
    return post[name].values.reshape(-1)


def _posterior_predict(idata, features: dict, scaler, debug: bool = False) -> np.ndarray:
    """
    Reconstruct posterior predictive mean (µ) from Equation 4:
        µ = PCFbase(s, θ) + β0 + b_brand[i] + h(s; φ)

    b_brand is set to 0 (population mean) for a new, unknown device.

    Parameters match the PyMC model in scripts/04_fit_bayesian_emulator.py exactly.
    """
    post = idata.posterior

    if debug:
        print("\n  Physics features passed to model:")
        for k, v in features.items():
            print(f"    {k:25s} = {v:.6f}")

    # ── Physics emission factors (Table I posteriors) ─────────────────────────
    battery_ef              = _flat(post, "battery_ef")        # kg CO2e / kWh
    display_ef              = _flat(post, "display_ef")        # kg CO2e / cm²
    material_ef             = _flat(post, "material_ef")       # kg CO2e / kg
    semiconductor_intensity = _flat(post, "semiconductor_intensity")  # per Pidx unit
    assembly_ef             = _flat(post, "assembly_ef")       # kg CO2e / device
    transport_ef            = _flat(post, "transport_ef")      # kg CO2e / device
    lifetime_years          = _flat(post, "lifetime_years")    # years
    grid_ef                 = _flat(post, "grid_ef")           # kg CO2e / kWh

    # ── Equation 1: Physics baseline ──────────────────────────────────────────
    pcf_base = (
        features["battery_kwh"]       * battery_ef
        + features["display_area_cm2"]  * display_ef
        + features["mass_kg"]           * material_ef
        + features["annual_kwh"]        * grid_ef * lifetime_years
        + features["performance_index"] * semiconductor_intensity
        + assembly_ef
        + transport_ef
    )

    # ── Intercept β0 (global bias correction) ────────────────────────────────
    beta0 = _flat(post, "beta0")

    # ── Residual h(s; φ) = φ · X_scaled, where X = [perf_idx, camera_mp] ────
    phi      = post["phi"].values.reshape(-1, post["phi"].shape[-1])  # (N, 2)
    X_resid  = np.array([[features["performance_index"],
                          features["camera_mp"]]])
    try:
        X_scaled = scaler.transform(X_resid).flatten()
    except Exception:
        # Fallback: manual z-score using scaler parameters
        X_scaled = ((X_resid - scaler.mean_) / scaler.scale_).flatten()

    h = (phi * X_scaled).sum(axis=1)

    # ── µ = PCFbase + β0 + b_brand(=0) + h ───────────────────────────────────
    mu = pcf_base + beta0 + h

    return mu


# ── Brand random effect (posterior mean) ─────────────────────────────────────

def _brand_b(idata, brand: str) -> float:
    """
    Return posterior mean of b_brand[brand_idx] from the hierarchical model.
    This uses the actual MCMC brand random effect, not the ridge calibrator.
    Returns 0.0 if brand is unknown (population mean).
    """
    try:
        import joblib
        brand_to_idx = joblib.load(BRAND_IDX)
        key = brand.lower().strip()
        if key not in brand_to_idx:
            print(f"  [brand] '{brand}' not in training set. "
                  f"Known: {sorted(brand_to_idx.keys())}")
            return 0.0
        idx    = brand_to_idx[key]
        b_mean = float(idata.posterior["b_brand"].values[:, :, idx].mean())
        return b_mean
    except Exception as e:
        print(f"  [brand] Skipped: {e}")
        return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def predict(
    battery_mah:  float,
    display_in:   float,
    mass_g:       float | None,
    perf_idx:     float,
    annual_kwh:   float,
    camera_mp:    float,
    brand:        str   = "unknown",
    ci_level:     float = 0.90,
    output_json:  bool  = False,
    debug:        bool  = False,
):
    try:
        import arviz as az
        import joblib
    except ImportError as e:
        print(f"ERROR: {e}\nRun: pip install -r requirements.txt")
        sys.exit(1)

    for path, label in [(POSTERIOR, "Posterior (04_fit_bayesian_emulator.py)"),
                        (SCALER,    "Scaler (03_train_brand_calibrator.py)")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found:\n  {path}")
            sys.exit(1)

    print("Loading posterior…")
    idata  = az.from_netcdf(POSTERIOR)
    scaler = joblib.load(SCALER)

    # ── Convert inputs → physics features (Section IV-B) ─────────────────────
    battery_kwh      = battery_to_kwh(battery_mah)
    display_area_cm2 = display_to_cm2(display_in)
    if mass_g is None:
        mass_g = estimate_mass(battery_mah, display_area_cm2)
        print(f"  [mass] Estimated from battery+display heuristic: {mass_g:.1f} g")
    mass_kg = mass_g / 1000.0

    features = {
        "battery_kwh":       battery_kwh,
        "display_area_cm2":  display_area_cm2,
        "mass_kg":           mass_kg,
        "performance_index": perf_idx,
        "annual_kwh":        annual_kwh,
        "camera_mp":         camera_mp,
    }

    # ── Brand random effect (using actual posterior b_brand) ──────────────────
    brand_offset = 0.0
    if brand.lower() != "unknown":
        brand_offset = _brand_b(idata, brand)
        print(f"  [brand] '{brand}' posterior b_brand mean: {brand_offset:+.2f} kg CO₂e")

    # ── Posterior predictive samples ──────────────────────────────────────────
    mu_samples = _posterior_predict(idata, features, scaler, debug=debug)
    samples    = mu_samples + brand_offset

    # ── Summary statistics ────────────────────────────────────────────────────
    a          = (1 - ci_level) / 2 * 100
    mean_pcf   = float(np.mean(samples))
    median_pcf = float(np.median(samples))
    ci_lo      = float(np.percentile(samples, a))
    ci_hi      = float(np.percentile(samples, 100 - a))
    std_pcf    = float(np.std(samples))

    result = {
        "inputs": {
            "battery_mah":  battery_mah,
            "display_in":   display_in,
            "mass_g":       mass_g,
            "perf_idx":     perf_idx,
            "annual_kwh":   annual_kwh,
            "camera_mp":    camera_mp,
            "brand":        brand,
        },
        "physics_features": {
            "battery_kwh":      round(battery_kwh, 6),
            "display_area_cm2": round(display_area_cm2, 2),
            "mass_kg":          round(mass_kg, 4),
        },
        "pcf_mean_kgco2e":            round(mean_pcf,   1),
        "pcf_median_kgco2e":          round(median_pcf, 1),
        "pcf_std_kgco2e":             round(std_pcf,    1),
        f"ci_{int(ci_level*100)}_lo": round(ci_lo, 1),
        f"ci_{int(ci_level*100)}_hi": round(ci_hi, 1),
        "n_posterior_samples":        len(samples),
    }

    if output_json:
        print(json.dumps(result, indent=2))
    else:
        w = 58
        print(f"\n{'═'*w}")
        print(f"  Smartphone PCF Prediction")
        print(f"{'─'*w}")
        print(f"  Battery         : {battery_mah:>8.0f} mAh  → {battery_kwh*1000:.2f} Wh")
        print(f"  Display         : {display_in:>8.2f} in   → {display_area_cm2:.1f} cm²")
        print(f"  Mass            : {mass_g:>8.1f} g    → {mass_kg:.3f} kg")
        print(f"  Perf. index     : {perf_idx:>8.2f}       (fleet mean: {FLEET['performance_index']:.3f})")
        print(f"  Annual energy   : {annual_kwh:>8.2f} kWh/yr (fleet mean: {FLEET['annual_kwh']:.3f})")
        print(f"  Camera          : {camera_mp:>8.0f} MP       (fleet mean: {FLEET['camera_mp']:.1f})")
        if brand.lower() != "unknown":
            print(f"  Brand           : {brand:>8s}       (b_brand: {brand_offset:+.2f})")
        print(f"{'─'*w}")
        print(f"  PCF Mean        : {mean_pcf:>8.1f} kg CO₂e")
        print(f"  PCF Median      : {median_pcf:>8.1f} kg CO₂e")
        print(f"  Std deviation   : {std_pcf:>8.1f} kg CO₂e")
        print(f"  {int(ci_level*100)}% Credible Interval : [{ci_lo:.1f},  {ci_hi:.1f}] kg CO₂e")
        print(f"{'═'*w}")
        print(f"\n  Note: perf_idx range in calibration set: 20.0–37.12")
        print(f"        annual_kwh range in calibration set: 4.97–7.77 kWh/yr\n")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Predict smartphone PCF using the trained Bayesian emulator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Feature guidance (Section IV-B of paper):
  --perf     Composite performance index combining clock speed, RAM,
             core count, and refresh rate. Range in calibration set: 20–37.12.
             Example values: budget ~20, mid-range ~25, flagship ~30-37.
  --annual   Annual use-phase energy (kWh/yr). Range: 4.97–7.77.
             Typical flagship: 6.5–7.77 kWh/yr.
  --camera   Total camera megapixels (sum of all rear sensors).
             Fleet mean: 65.3 MP. Typical flagship: 48–200 MP.

Examples:
  # Typical flagship (5nm class):
  python scripts/predict_new.py --battery 4000 --display 6.5 --mass 180 --perf 27.5 --annual 6.5

  # iPhone 15 spec (with known values, brand posterior effect):
  python scripts/predict_new.py --battery 3279 --display 6.1 --mass 171 --perf 21.76 --annual 5.1 --camera 48 --brand apple

  # Pixel 8 Pro:
  python scripts/predict_new.py --battery 5050 --display 6.7 --mass 213 --perf 37.12 --annual 7.77 --camera 108 --brand google
        """,
    )
    p.add_argument("--battery", type=float, required=True,
                   help="Battery capacity in mAh (e.g. 4000)")
    p.add_argument("--display", type=float, required=True,
                   help="Display diagonal in inches (e.g. 6.5)")
    p.add_argument("--mass",    type=float, default=None,
                   help="Device mass in grams (optional; estimated if omitted)")
    p.add_argument("--perf",    type=float, required=True,
                   help="Performance index (20–37.12; see epilog for guidance)")
    p.add_argument("--annual",  type=float, required=True,
                   help="Annual use-phase energy in kWh/yr (4.97–7.77)")
    p.add_argument("--camera",  type=float, default=FLEET["camera_mp"],
                   help=f"Total camera MP (default: {FLEET['camera_mp']:.1f}, fleet mean)")
    p.add_argument("--brand",   type=str,   default="unknown",
                   help="Brand for posterior b_brand offset (e.g. apple, samsung)")
    p.add_argument("--ci",      type=float, default=0.90,
                   help="Credible interval level (default: 0.90)")
    p.add_argument("--json",    action="store_true", help="Output as JSON")
    p.add_argument("--debug",   action="store_true", help="Show feature values")

    args = p.parse_args()
    predict(
        battery_mah=args.battery,
        display_in=args.display,
        mass_g=args.mass,
        perf_idx=args.perf,
        annual_kwh=args.annual,
        camera_mp=args.camera,
        brand=args.brand,
        ci_level=args.ci,
        output_json=args.json,
        debug=args.debug,
    )