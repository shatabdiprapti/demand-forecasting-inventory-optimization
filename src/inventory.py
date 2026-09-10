"""
inventory.py
------------
Turns ML demand forecasts into replenishment decisions per store-item pair:
  - Safety Stock   : buffer stock to protect against forecast error
  - Reorder Point  : inventory level that triggers a new order
  - EOQ            : Economic Order Quantity, the cost-optimal order size

Assumptions (typical for a portfolio project - the raw Kaggle dataset has
no cost/lead-time data, so these are declared, reasonable placeholders):
  - Lead time        : 7 days (time between placing and receiving an order)
  - Ordering cost (S) : $50 per order (admin + shipping overhead)
  - Holding cost (H)  : 20% of item value per year; we assume $1 unit value
                        as a stand-in since the dataset has no price column
  - Service level     : 95% -> Z-score = 1.645 (from the standard normal table)
"""

import numpy as np
import pandas as pd

Z_95 = 1.645          # Z-score for 95% service level
LEAD_TIME_DAYS = 7
ORDERING_COST = 50.0   # $ per order
HOLDING_COST_PER_UNIT_YEAR = 0.20  # 20% of $1 assumed unit value


def compute_forecast_error(val_predictions: pd.DataFrame) -> pd.DataFrame:
    """Per store-item pair, compute the standard deviation of forecast
    residuals (actual - predicted) from the backtested validation set.
    This becomes our estimate of demand uncertainty (sigma) used in the
    safety stock formula - a forecast that's usually spot-on needs less
    buffer stock than one that swings wildly.
    """
    val_predictions = val_predictions.copy()
    val_predictions["error"] = val_predictions["y_true"] - val_predictions["y_pred_ensemble"]
    error_std = (
        val_predictions.groupby(["store", "item"])["error"]
        .std()
        .rename("forecast_error_std")
        .reset_index()
    )
    avg_daily_demand = (
        val_predictions.groupby(["store", "item"])["y_pred_ensemble"]
        .mean()
        .rename("avg_daily_demand")
        .reset_index()
    )
    return error_std.merge(avg_daily_demand, on=["store", "item"])


def compute_inventory_policy(demand_stats: pd.DataFrame) -> pd.DataFrame:
    """Given avg_daily_demand and forecast_error_std per store-item,
    compute Safety Stock, Reorder Point, and EOQ.
    """
    df = demand_stats.copy()
    df["forecast_error_std"] = df["forecast_error_std"].fillna(df["forecast_error_std"].mean())

    # Safety Stock: buffer against demand variability during lead time
    # SS = Z * sigma_demand * sqrt(lead_time)
    df["safety_stock"] = Z_95 * df["forecast_error_std"] * np.sqrt(LEAD_TIME_DAYS)

    # Reorder Point: stock level that triggers a new purchase order
    # ROP = (avg daily demand * lead time) + safety stock
    df["reorder_point"] = (df["avg_daily_demand"] * LEAD_TIME_DAYS) + df["safety_stock"]

    # EOQ: cost-optimal order quantity
    # EOQ = sqrt( (2 * annual_demand * ordering_cost) / holding_cost_per_unit )
    df["annual_demand"] = df["avg_daily_demand"] * 365
    df["eoq"] = np.sqrt(
        (2 * df["annual_demand"] * ORDERING_COST) / HOLDING_COST_PER_UNIT_YEAR
    )

    # Round to whole units (you can't order half an item)
    for col in ["safety_stock", "reorder_point", "eoq"]:
        df[col] = df[col].round().astype(int)

    return df[["store", "item", "avg_daily_demand", "forecast_error_std",
               "safety_stock", "reorder_point", "eoq"]]


if __name__ == "__main__":
    val_predictions = pd.read_csv("data/processed/val_predictions.csv")
    demand_stats = compute_forecast_error(val_predictions)
    policy = compute_inventory_policy(demand_stats)

    policy = policy.sort_values(["store", "item"])
    policy.to_csv("data/processed/inventory_policy.csv", index=False)

    print(f"Computed inventory policy for {len(policy)} store-item pairs.")
    print("\nSample output:")
    print(policy.head(10).to_string(index=False))
    print(f"\nSaved full table to data/processed/inventory_policy.csv")
