"""rule-based baselines (RQ4).

Each baseline returns a score per (user_id, product_id); higher = suggest first.

1. most_frequent   the user's most often bought items                  score = up_count (ties: most recent)
2. last_order      repeat the user's last order                        score = in last order, then up_count
3. fixed_interval  items whose usual repurchase gap has elapsed         score = closeness of due ratio to 1
                   (due ratio = days since last purchase / mean gap; user's mean order gap if bought once)
4. popularity      the globally most popular products (same list for everyone, may include new items)

Baselines 1-3 rank the user's candidates; baseline 4 is a non-personalised recommender.
"""
import numpy as np
import pandas as pd


def most_frequent(f: pd.DataFrame) -> pd.Series:
    return f["up_count"] - 1e-3 * f["up_orders_since_last"]


def last_order(f: pd.DataFrame) -> pd.Series:
    return f["up_in_last_order"] * 1e6 + f["up_count"]


def fixed_interval(f: pd.DataFrame) -> pd.Series:
    due = f["up_due_ratio"].fillna(np.inf)
    # 1 when exactly due, falls towards 0 the further from due (early or overdue); ties: more frequent first
    return 1 / (1 + (due - 1).abs()) + 1e-6 * f["up_count"]


def popularity(users: pd.Index, product_popularity: pd.Series, k: int) -> pd.DataFrame:
    top = product_popularity.sort_values(ascending=False).head(k)
    return pd.DataFrame({
        "user_id": np.repeat(users.values, len(top)),
        "product_id": np.tile(top.index.values, len(users)),
        "score": np.tile(top.values.astype(float), len(users)),
    })


CANDIDATE_BASELINES = {
    "most_frequent": most_frequent,
    "last_order": last_order,
    "fixed_interval": fixed_interval,
}
