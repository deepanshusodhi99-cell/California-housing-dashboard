"""
generate_model.py

Trains a real regression model on the California Housing Prices dataset
(from Aurelien Geron's "Hands-On Machine Learning" repository, itself
sourced from the 1990 California census -- a standard, widely-used, and
ethically uncontroversial public regression dataset). This is used here
as a direct, real-data replacement for the deprecated/ethically-flagged
Boston Housing dataset.

This uses REAL public data (housing.csv, included in this repo) and a
REAL trained model -- the R^2 score, MAE, and predictions in
model_results.json are genuine model output, not illustrative numbers.

Run: python generate_model.py
Requires: housing.csv (in the same folder)
Output: model_results.json
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---- Load real data ----
df = pd.read_csv("housing.csv").dropna()  # a few total_bedrooms nulls in the original dataset

y = df["median_house_value"]
X = df.drop(columns=["median_house_value"])

numeric_features = [c for c in X.columns if c != "ocean_proximity"]
categorical_features = ["ocean_proximity"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
], remainder="passthrough")

pipeline = Pipeline([
    ("prep", preprocess),
    ("model", RandomForestRegressor(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1)),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Train real model ----
pipeline.fit(X_train, y_train)
pred_test = pipeline.predict(X_test)

r2 = r2_score(y_test, pred_test)
mae = mean_absolute_error(y_test, pred_test)

# ---- Feature importances (map back to readable names) ----
ohe = pipeline.named_steps["prep"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(categorical_features))
all_feature_names = cat_names + numeric_features
importances = pipeline.named_steps["model"].feature_importances_

feat_importance = sorted(
    [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(all_feature_names, importances)],
    key=lambda x: x["importance"],
    reverse=True,
)[:10]

# ---- Sample of actual vs predicted (for scatter plot) ----
rng = np.random.default_rng(seed=1)
sample_idx = rng.choice(len(y_test), size=min(500, len(y_test)), replace=False)
y_test_arr = y_test.values
scatter_points = [
    {"actual": round(float(y_test_arr[i]), 0), "predicted": round(float(pred_test[i]), 0)}
    for i in sample_idx
]

# ---- Residual distribution (binned, in dollars) ----
residuals = y_test_arr - pred_test
hist, bin_edges = np.histogram(residuals, bins=20)
residual_bins = [
    {"range": f"{int(bin_edges[i]/1000)}k to {int(bin_edges[i+1]/1000)}k", "count": int(hist[i])}
    for i in range(len(hist))
]

# ---- Median value by region ----
by_region = (
    df.groupby("ocean_proximity")["median_house_value"]
    .agg(["mean", "count"])
    .reset_index()
    .sort_values("mean", ascending=False)
)
region_breakdown = [
    {"region": r["ocean_proximity"], "avg_value": round(float(r["mean"]), 0), "count": int(r["count"])}
    for _, r in by_region.iterrows()
]

output = {
    "meta": {
        "dataset": "California Housing Prices (1990 census) via Aurelien Geron's Hands-On Machine Learning repo",
        "note": "Real public dataset and a real trained model. Metrics below are genuine model output.",
        "n_rows": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model": "RandomForestRegressor (n_estimators=200, max_depth=14) in a scikit-learn Pipeline with one-hot encoding",
    },
    "metrics": {
        "r2_score": round(float(r2), 4),
        "mae_dollars": round(float(mae), 0),
    },
    "feature_importance": feat_importance,
    "scatter_sample": scatter_points,
    "residual_bins": residual_bins,
    "region_breakdown": region_breakdown,
}

with open("model_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"R2: {r2:.4f}  |  MAE: ${mae:,.0f}  |  rows: {len(df)}")
print("Wrote model_results.json")
