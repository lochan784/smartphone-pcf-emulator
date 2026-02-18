import pandas as pd
import numpy as np
import os
import joblib

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MERGED_PATH = "data/processed/merged_for_calibration.csv"
MODEL_PATH = "models/brand_ridge_calibrator.joblib"

merged = pd.read_csv(MERGED_PATH)

X_cols = [
    "carbon_battery",
    "carbon_display",
    "carbon_electronics",
    "carbon_materials",
    "carbon_assembly",
    "carbon_transport",
    "carbon_use"
]

# Brand dummies
X_brand = pd.get_dummies(merged["manufacturer"], drop_first=True)
X_full = pd.concat([merged[X_cols], X_brand], axis=1)

y = merged["pcf_kg_co2e_y"]

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=[0.01,0.1,1,10,100], cv=5))
])

pipeline.fit(X_full, y)

y_pred = pipeline.predict(X_full)

mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
bias = (y_pred - y).mean()
r2 = r2_score(y, y_pred)

print("\n--- Brand-Aware Calibration (In-Sample) ---")
print("MAE:", round(mae,2))
print("RMSE:", round(rmse,2))
print("Bias:", round(bias,2))
print("R2:", round(r2,3))

os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)
