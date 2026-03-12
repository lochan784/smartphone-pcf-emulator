import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# CONFIG
# -----------------------------
GOLD_PATH = "data/processed/gold_holdout.csv"
PRED_PATH = "data/processed/posterior_predictions_gold.csv"

TARGET_COL = "declared_pcf_kgco2e"
PRED_COL = "pred_mean"
LOWER_COL = "pred_lower"
UPPER_COL = "pred_upper"

# -----------------------------
# LOAD DATA
# -----------------------------
gold = pd.read_csv(GOLD_PATH)
preds = pd.read_csv(PRED_PATH)

df = gold.merge(preds, on="device_name", how="inner")

y_true = df[TARGET_COL].values
y_pred = df[PRED_COL].values

# -----------------------------
# POINT METRICS
# -----------------------------
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

# -----------------------------
# UNCERTAINTY METRICS
# -----------------------------
coverage = np.mean(
    (y_true >= df[LOWER_COL]) &
    (y_true <= df[UPPER_COL])
)

avg_interval_width = np.mean(df[UPPER_COL] - df[LOWER_COL])

# -----------------------------
# PRINT RESULTS
# -----------------------------
print("\n========== MODEL EVALUATION ==========\n")
print(f"MAE  (kg CO2e): {mae:.3f}")
print(f"RMSE (kg CO2e): {rmse:.3f}")
print(f"R^2           : {r2:.3f}\n")

print("---- UNCERTAINTY QUALITY ----")
print(f"Coverage (interval): {coverage*100:.1f}%")
print(f"Avg interval width : {avg_interval_width:.2f} kg CO2e\n")

print("======================================")
