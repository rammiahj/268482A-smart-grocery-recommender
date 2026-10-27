"""Evaluate the rule-based baselines on the TEST target (each user's last order).

Writes reports/baselines_test.csv and prints a comparison table.

Usage:
    python -m src.models.run_baselines
"""
import duckdb
import pandas as pd

from src.config import PROCESSED_DIR, ROOT
from src.models.baselines import CANDIDATE_BASELINES, popularity
from src.models.evaluate import KS, candidate_ceiling, classification_metrics, topk_metrics

COLS = ["user_id", "product_id", "label", "up_count", "up_orders_since_last",
        "up_in_last_order", "up_due_ratio", "u_avg_basket", "p_purchases"]


def main() -> None:
    f = pd.read_parquet(PROCESSED_DIR / "features_test.parquet", columns=COLS)
    truth = pd.read_parquet(PROCESSED_DIR / "truth_test.parquet")
    basket_k = f.groupby("user_id")["u_avg_basket"].first()
    print(f"{len(f):,} candidates | {truth.user_id.nunique():,} users | positive rate {f.label.mean():.3f}")

    rows = []
    for name, fn in CANDIDATE_BASELINES.items():
        score = fn(f)
        scored = pd.DataFrame({"user_id": f.user_id, "product_id": f.product_id, "score": score})
        row = {"method": name}
        row.update(topk_metrics(scored, truth, KS, basket_k))
        row.update(classification_metrics(f.label.values, score.values))
        rows.append(row)
        print(f"done: {name}")

    # Global popularity (non-personalised; computed from history + val purchases)
    pop = duckdb.sql(f"""
        SELECT product_id, COUNT(*) AS n
        FROM '{(PROCESSED_DIR / "order_products_clean.parquet").as_posix()}'
        WHERE split IN ('history', 'val') GROUP BY product_id
    """).df().set_index("product_id")["n"]
    users = pd.Index(truth.user_id.unique())
    max_k = int(max(max(KS), basket_k.round().max()))
    scored = popularity(users, pop, max_k)
    row = {"method": "popularity"}
    row.update(topk_metrics(scored, truth, KS, basket_k))
    rows.append(row)
    print("done: popularity")

    res = pd.DataFrame(rows).set_index("method")
    out = ROOT / "reports" / "baselines_test.csv"
    res.to_csv(out)

    ceiling = candidate_ceiling(f, truth)
    show = [f"{m}@{k}" for k in (*KS, "basket") for m in ("precision", "recall", "f1", "hit_rate")]
    show += ["roc_auc", "pr_auc", "best_f1"]
    pd.set_option("display.width", 200)
    print("\n=== Baselines on test (each user's last order) ===")
    print(res[[c for c in show if c in res.columns]].T.round(4))
    print(f"\nRecall ceiling for any candidate-only method: per-user {ceiling['recall_ceiling_per_user']:.3f}, "
          f"overall {ceiling['recall_ceiling_overall']:.3f}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
