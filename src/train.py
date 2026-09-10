"""
train.py
--------
Trains LightGBM + XGBoost, ensembles them, and benchmarks against
a naive seasonal baseline using SMAPE.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from features import build_features, FEATURE_COLUMNS


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error.
    This is the standard metric for demand forecasting because it's
    scale-independent (works whether an item sells 5/day or 500/day)
    and bounded between 0% and 200%.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1, denom)  # avoid divide-by-zero on days with 0 sales
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)


def load_and_split(path="data/raw/train.csv", val_days=90):
    """Time-based split: last `val_days` days become validation.
    We NEVER shuffle time series data randomly - that would let the
    model "see the future" during training.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=val_days)
    train_df = df[df["date"] <= cutoff].copy()
    val_df = df[df["date"] > cutoff].copy()
    return df, train_df, val_df


def naive_seasonal_baseline(full_df, val_df):
    """Baseline: predict this year's sales using 'sales from 365 days ago'.
    This is what a supply chain team might do with zero ML - our
    job is to beat this by a meaningful margin (target: ~20%+ SMAPE improvement).
    """
    lookup = full_df.set_index(["store", "item", "date"])["sales"]
    preds = []
    for _, row in val_df.iterrows():
        key = (row["store"], row["item"], row["date"] - pd.Timedelta(days=365))
        preds.append(lookup.get(key, np.nan))
    preds = pd.Series(preds, index=val_df.index)
    preds = preds.fillna(val_df["sales"].mean())  # fallback for missing history
    return preds.values


if __name__ == "__main__":
    print("Loading data and splitting...")
    full_df, train_df, val_df = load_and_split()
    print(f"Train: {train_df.shape}, Val: {val_df.shape}")

    print("\nBuilding features...")
    # Use the FULL train_df as history so lag/rolling features for the
    # validation period can look back correctly.
    combined = pd.concat([train_df, val_df], axis=0).sort_values(["store", "item", "date"])
    combined_feat = build_features(combined, train_ref=train_df)

    train_feat = combined_feat[combined_feat["date"] <= train_df["date"].max()].copy()
    val_feat = combined_feat[combined_feat["date"] > train_df["date"].max()].copy()

    # Drop early rows where lag_365 can't be computed (first year of data)
    train_feat = train_feat.dropna(subset=["lag_365"])

    X_train, y_train = train_feat[FEATURE_COLUMNS], train_feat["sales"]
    X_val, y_val = val_feat[FEATURE_COLUMNS], val_feat["sales"]

    print(f"Final training rows after dropna: {len(X_train)}")

    # ---------------- Baseline ----------------
    print("\nComputing naive seasonal baseline...")
    baseline_preds = naive_seasonal_baseline(full_df, val_df)
    baseline_smape = smape(val_df["sales"].values, baseline_preds)
    print(f"Naive baseline SMAPE: {baseline_smape:.3f}")

    # ---------------- LightGBM ----------------
    print("\nTraining LightGBM...")
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    lgb_params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }
    lgb_model = lgb.train(
        lgb_params, lgb_train, num_boost_round=1000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    lgb_preds = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    lgb_smape = smape(y_val.values, lgb_preds)
    print(f"LightGBM SMAPE: {lgb_smape:.3f}")

    # ---------------- XGBoost ----------------
    print("\nTraining XGBoost...")
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_val = xgb.DMatrix(X_val, label=y_val)
    xgb_params = {
        "objective": "reg:squarederror",
        "eta": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mae",
    }
    xgb_model = xgb.train(
        xgb_params, xgb_train, num_boost_round=1000,
        evals=[(xgb_val, "val")],
        early_stopping_rounds=50, verbose_eval=100,
    )
    xgb_preds = xgb_model.predict(xgb_val, iteration_range=(0, xgb_model.best_iteration))
    xgb_smape = smape(y_val.values, xgb_preds)
    print(f"XGBoost SMAPE: {xgb_smape:.3f}")

    # ---------------- Ensemble ----------------
    print("\nSearching best ensemble weight...")
    best_w, best_smape = 0.5, 999
    for w in np.arange(0, 1.05, 0.05):
        blend = w * lgb_preds + (1 - w) * xgb_preds
        s = smape(y_val.values, blend)
        if s < best_smape:
            best_smape, best_w = s, w
    ensemble_preds = best_w * lgb_preds + (1 - best_w) * xgb_preds
    print(f"Best ensemble weight (LGB): {best_w:.2f} | Ensemble SMAPE: {best_smape:.3f}")

    improvement = 100 * (baseline_smape - best_smape) / baseline_smape
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"Naive baseline SMAPE : {baseline_smape:.3f}")
    print(f"LightGBM SMAPE       : {lgb_smape:.3f}")
    print(f"XGBoost SMAPE        : {xgb_smape:.3f}")
    print(f"Ensemble SMAPE       : {best_smape:.3f}")
    print(f"Improvement vs base  : {improvement:.1f}%")

    # Save models + predictions for reuse in inventory.py
    lgb_model.save_model("models/lgbm_model.txt")
    xgb_model.save_model("models/xgb_model.json")
    val_feat["y_true"] = y_val.values
    val_feat["y_pred_ensemble"] = ensemble_preds
    val_feat.to_csv("data/processed/val_predictions.csv", index=False)
    print("\nSaved models to models/ and predictions to data/processed/val_predictions.csv")
