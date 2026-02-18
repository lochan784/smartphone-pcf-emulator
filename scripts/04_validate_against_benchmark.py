import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_PATH = "data/processed/smartphones_structured.csv"
BENCH_PATH = "data/processed/pcf_benchmark_strict.csv"

df = pd.read_csv(PROCESSED_PATH)
bench = pd.read_csv(BENCH_PATH)

# --- Standardize ---
df["model"] = df["model"].str.lower().str.strip()
df["brand"] = df["brand"].str.lower().str.strip()
bench["model"] = bench["model"].str.lower().str.strip()

# Remove brand + 5g for merge
import re
def clean_model(row):
    model = str(row["model"])
    brand = str(row.get("brand",""))
    model = model.replace(brand, "")
    model = model.replace("5g", "")
    model = re.sub(r'[^a-z0-9 ]+', ' ', model)
    model = re.sub(r'\s+', ' ', model).strip()
    return model

df["merge_model"] = df.apply(clean_model, axis=1)
bench["merge_model"] = bench["model"]

# Keep one synthetic row per model
df_unique = df.sort_values("storage_gb").drop_duplicates(subset="merge_model", keep="first")

# Merge (model-level)
merged = df_unique.merge(
    bench,
    how="inner",
    left_on="merge_model",
    right_on="merge_model"
)

print("\nMatched Models:", len(merged))

if len(merged) > 0:
    y_true = merged["pcf_kg_co2e_y"]
    y_pred = merged["pcf_kg_co2e_x"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = (y_pred - y_true).mean()
    r2 = r2_score(y_true, y_pred)

    print("\n--- Validation Metrics ---")
    print("MAE:", round(mae,2))
    print("RMSE:", round(rmse,2))
    print("Bias:", round(bias,2))
    print("R2:", round(r2,3))

    print("\n--- Per-Model Errors ---")
    merged["error"] = y_pred - y_true
    print(merged[["merge_model","pcf_kg_co2e_x","pcf_kg_co2e_y","error"]])
else:
    print("No matches found.")
