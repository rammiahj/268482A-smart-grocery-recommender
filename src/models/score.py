"""Batch scoring service: score every test candidate with the trained LightGBM model.

Writes data/processed/predictions_test.parquet (sorted by user for fast per-user lookups),
which the Streamlit dashboard reads.

Usage:
    python -m src.models.score
"""
import time

import duckdb
import lightgbm as lgb
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR
from src.models.baselines import most_frequent
from src.models.train import FEATURES

KEEP = ["user_id", "product_id", "label", "u_orders", "u_avg_basket", "u_mean_order_gap", "u_reorder_ratio",
        "up_count", "up_orders_since_last", "up_days_since_last", "up_mean_gap", "up_due_ratio",
        "up_in_last_order", "t_days_since_prior"]


def main() -> None:
    t0 = time.time()
    model = lgb.Booster(model_file=str(MODELS_DIR / "lgbm.txt"))
    f = pd.read_parquet(PROCESSED_DIR / "features_test.parquet",
                        columns=list(dict.fromkeys(["user_id", "product_id", "label", *FEATURES, *KEEP])))
    out = f[KEEP].copy()
    out["p_lgbm"] = model.predict(f[FEATURES].astype("float32"))
    out["s_most_frequent"] = most_frequent(f)
    del f

    path = (PROCESSED_DIR / "predictions_test.parquet").as_posix()
    duckdb.register("out", out)
    duckdb.execute(f"COPY (SELECT * FROM out ORDER BY user_id, p_lgbm DESC) TO '{path}' "
                   f"(FORMAT PARQUET, ROW_GROUP_SIZE 100000)")
    print(f"Scored {len(out):,} candidates for {out.user_id.nunique():,} users in {time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
