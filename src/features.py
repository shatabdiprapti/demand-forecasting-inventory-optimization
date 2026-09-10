"""
features.py
------------
Builds engineered features for the demand forecasting model.

Feature groups:
1. Calendar features       -> capture weekly/monthly/yearly patterns
2. Lag features             -> "what were sales N days ago"
3. Rolling window stats     -> mean/std/min/max over trailing windows
4. Exponentially weighted   -> EWM gives more weight to recent days
5. Seasonality (Fourier)    -> smooth cyclical encoding of day-of-year
6. Group aggregates         -> store-level / item-level average behavior

IMPORTANT: every lag/rolling/EWM feature is built using .shift(1) first,
so the model never "sees" the current day's own sales value when
predicting it. This prevents data leakage.
"""

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Basic calendar features: 7 features."""
    df["dayofweek"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


def add_fourier_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Smooth yearly/weekly seasonality using sine/cosine encoding: 4 features.
    This is better than raw month/day numbers because it tells the model
    that December (12) and January (1) are actually close together.
    """
    day_of_year = df["date"].dt.dayofyear
    df["sin_year"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["cos_year"] = np.cos(2 * np.pi * day_of_year / 365.25)
    df["sin_week"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["cos_week"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_lag_features(df: pd.DataFrame, lags=(1, 7, 14, 28, 90, 365)) -> pd.DataFrame:
    """Sales value N days ago, per store-item pair: 6 features.
    Grouping by (store, item) ensures we never leak sales from a
    different store or item into the lag.
    """
    grp = df.groupby(["store", "item"], sort=False)["sales"]
    for lag in lags:
        df[f"lag_{lag}"] = grp.shift(lag).astype("float32")
    return df


def add_rolling_features(df: pd.DataFrame, windows=(7, 14, 30, 90)) -> pd.DataFrame:
    """Rolling mean/std/min/max over trailing windows: 4 windows x 4 stats = 16 features.
    .shift(1) first so the window only looks at PAST days, never today.

    Memory-efficient version: sort once, then for each window compute all
    4 stats in a SINGLE groupby-rolling pass (via .agg) instead of 4
    separate full-dataframe transforms. This avoids the OOM issues you'd
    otherwise hit with 900K+ rows x 500 store-item groups.
    """
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    df["_shifted_sales"] = df.groupby(["store", "item"], sort=False)["sales"].shift(1)

    for w in windows:
        roll = (
            df.groupby(["store", "item"], sort=False)["_shifted_sales"]
            .rolling(window=w, min_periods=1)
            .agg(["mean", "std", "min", "max"])
            .reset_index(drop=True)
        )
        df[f"roll_mean_{w}"] = roll["mean"].astype("float32")
        df[f"roll_std_{w}"] = roll["std"].astype("float32")
        df[f"roll_min_{w}"] = roll["min"].astype("float32")
        df[f"roll_max_{w}"] = roll["max"].astype("float32")
        del roll

    df.drop(columns=["_shifted_sales"], inplace=True)
    return df


def add_ewm_features(df: pd.DataFrame, spans=(7, 30, 90)) -> pd.DataFrame:
    """Exponentially weighted mean: gives more weight to recent sales: 3 features."""
    df["_shifted_sales"] = df.groupby(["store", "item"], sort=False)["sales"].shift(1)
    for span in spans:
        df[f"ewm_{span}"] = (
            df.groupby(["store", "item"], sort=False)["_shifted_sales"]
            .transform(lambda s: s.ewm(span=span, min_periods=1).mean())
            .astype("float32")
        )
    df.drop(columns=["_shifted_sales"], inplace=True)
    return df


def add_group_aggregates(df: pd.DataFrame, train_ref: pd.DataFrame) -> pd.DataFrame:
    """Historical average behavior of each store / item / store-item pair.
    Computed ONLY from train_ref (training data) to avoid leaking
    validation/test information back into the features. 3 features.
    """
    store_mean = train_ref.groupby("store")["sales"].mean().rename("store_avg_sales")
    item_mean = train_ref.groupby("item")["sales"].mean().rename("item_avg_sales")
    pair_mean = train_ref.groupby(["store", "item"])["sales"].mean().rename("pair_avg_sales")

    df = df.merge(store_mean, on="store", how="left")
    df = df.merge(item_mean, on="item", how="left")
    df = df.merge(pair_mean, on=["store", "item"], how="left")
    return df


def build_features(df: pd.DataFrame, train_ref: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline. `train_ref` = the training portion,
    used for lag/rolling history and for computing group averages safely.
    """
    df = df.copy()
    df["store"] = df["store"].astype("int16")
    df["item"] = df["item"].astype("int16")
    df["sales"] = df["sales"].astype("float32")
    df = add_calendar_features(df)
    df = add_fourier_seasonality(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_ewm_features(df)
    df = add_group_aggregates(df, train_ref)
    return df


FEATURE_COLUMNS = [
    "store", "item", "dayofweek", "day", "month", "year", "weekofyear",
    "is_weekend", "is_month_start", "is_month_end",
    "sin_year", "cos_year", "sin_week", "cos_week",
    "lag_1", "lag_7", "lag_14", "lag_28", "lag_90", "lag_365",
    "roll_mean_7", "roll_std_7", "roll_min_7", "roll_max_7",
    "roll_mean_14", "roll_std_14", "roll_min_14", "roll_max_14",
    "roll_mean_30", "roll_std_30", "roll_min_30", "roll_max_30",
    "roll_mean_90", "roll_std_90", "roll_min_90", "roll_max_90",
    "ewm_7", "ewm_30", "ewm_90",
    "store_avg_sales", "item_avg_sales", "pair_avg_sales",
]
