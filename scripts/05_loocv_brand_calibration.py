import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv("data/processed/smartphones_structured.csv")
bench = pd.read_csv("data/processed/pcf_benchmark_strict.csv")

# Standardize
df["model"] = df["model"].str.lower().str.strip()
df["brand"] = df["brand"].str.lower().str.strip()
bench["model"] = bench["model"].str.lower().str.strip()

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

df_unique = df.sort_values("storage_gb").drop_duplicates(subset="merge_model", keep="first")

merged = df_unique.merge(
    bench,
    how="inner",
    left_on="merge_model",
    right_on="merge_model"
)

# Features
X_cols = [
    "carbon_battery",
    "carbon_display",
    "carbon_electronics",
    "carbon_materials",
    "carbon_assembly",
    "carbon_transport",
    "carbon_use"
]

X_brand = pd.get_dummies(merged["manufacturer"], drop_first=True)
X_full = pd.concat([merged[X_cols], X_brand], axis=1)
y = merged["pcf_kg_co2e_y"]

# LOOCV
loo = LeaveOneOut()

y_true_cv = []
y_pred_cv = []

for train_idx, test_idx in loo.split(X_full):

    X_train = X_full.iloc[train_idx]
    X_test  = X_full.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test  = y.iloc[test_idx]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.01,0.1,1,10,100], cv=5))
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    y_true_cv.append(y_test.values[0])
    y_pred_cv.append(pred[0])

mae_cv = mean_absolute_error(y_true_cv, y_pred_cv)
r2_cv = r2_score(y_true_cv, y_pred_cv)

print("\n--- LOOCV Brand-Aware ---")
print("MAE:", round(mae_cv,2))
print("R2:", round(r2_cv,3))
