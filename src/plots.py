"""
make_plots.py
-------------
Generates the charts you'll embed in your README:
1. SMAPE comparison bar chart (baseline vs models)
2. Actual vs predicted sales for a sample store-item pair
3. LightGBM feature importance
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb

# ---- 1. SMAPE comparison ----
scores = {
    "Naive Baseline": 22.048,
    "LightGBM": 12.388,
    "XGBoost": 12.383,
    "Ensemble": 12.358,
}
plt.figure(figsize=(7, 4.5))
bars = plt.bar(scores.keys(), scores.values(), color=["#999999", "#4C72B0", "#DD8452", "#55A868"])
plt.ylabel("SMAPE (lower is better)")
plt.title("Forecast Accuracy: Baseline vs ML Models")
for bar, val in zip(bars, scores.values()):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.3, f"{val:.2f}", ha="center")
plt.tight_layout()
plt.savefig("reports/figures/smape_comparison.png", dpi=150)
plt.close()

# ---- 2. Actual vs predicted for one store-item pair ----
val = pd.read_csv("data/processed/val_predictions.csv", parse_dates=["date"])
sample = val[(val["store"] == 1) & (val["item"] == 1)].sort_values("date")
plt.figure(figsize=(10, 4.5))
plt.plot(sample["date"], sample["y_true"], label="Actual", linewidth=2)
plt.plot(sample["date"], sample["y_pred_ensemble"], label="Predicted (Ensemble)", linewidth=2, linestyle="--")
plt.title("Actual vs Predicted Sales — Store 1, Item 1")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/actual_vs_predicted.png", dpi=150)
plt.close()

# ---- 3. Feature importance ----
model = lgb.Booster(model_file="models/lgbm_model.txt")
importance = pd.DataFrame({
    "feature": model.feature_name(),
    "importance": model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=True).tail(15)

plt.figure(figsize=(8, 6))
plt.barh(importance["feature"], importance["importance"], color="#4C72B0")
plt.title("Top 15 Feature Importances (LightGBM, gain)")
plt.tight_layout()
plt.savefig("reports/figures/feature_importance.png", dpi=150)
plt.close()

print("Saved 3 charts to reports/figures/")
