import pandas as pd
import re
import os

PROCESSED_PATH = "data/processed/smartphones_structured.csv"
BENCH_PATH = "data/processed/pcf_benchmark_strict.csv"
OUT_PATH = "data/processed/merged_for_calibration.csv"

df = pd.read_csv(PROCESSED_PATH)
bench = pd.read_csv(BENCH_PATH)

# Standardize
df["model"] = df["model"].str.lower().str.strip()
df["brand"] = df["brand"].str.lower().str.strip()
bench["model"] = bench["model"].str.lower().str.strip()

def clean_model(row):
    model = str(row["model"])
    brand = str(row.get("brand", ""))
    model = model.replace(brand, "")
    model = model.replace("5g", "")
    model = re.sub(r'[^a-z0-9 ]+', ' ', model)
    model = re.sub(r'\s+', ' ', model).strip()
    return model

df["merge_model"] = df.apply(clean_model, axis=1)
bench["merge_model"] = bench["model"]

# Keep one synthetic row per model
df_unique = df.sort_values("storage_gb").drop_duplicates(
    subset="merge_model", keep="first"
)

merged = df_unique.merge(
    bench,
    how="inner",
    left_on="merge_model",
    right_on="merge_model"
)

os.makedirs("data/processed", exist_ok=True)
merged.to_csv(OUT_PATH, index=False)

print("Merged rows:", len(merged))
print("Saved to:", OUT_PATH)
