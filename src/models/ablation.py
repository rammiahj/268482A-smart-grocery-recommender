"""Feature-group ablation (RQ2): does each group of features add value?

LightGBM is retrained with one feature group removed at a time, on the same user sample,
and evaluated on test. A drop in PR-AUC / F1@basket shows how much that group contributes.

Output: reports/ablation_test.csv

Usage:
  python -m src.models.ablation                # 25% of training users (default, faster)
  python -m src.models.ablation --sample 0.5
"""
import argparse
import time

import pandas as pd

from src.config import PROCESSED_DIR, ROOT
from src.models.evaluate import topk_metrics
from src.models.train import FEATURE_GROUPS, FEATURES, LABEL, load, predict, split_users, train_lgbm
from sklearn.metrics import average_precision_score, roc_auc_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=float, default=0.25)
    args = ap.parse_args()

    train = load("val", args.sample)
    tr, tune = split_users(train)
    test = load("test")
    truth = pd.read_parquet(PROCESSED_DIR / "truth_test.parquet")
    basket_k = pd.read_parquet(PROCESSED_DIR / "features_test.parquet", columns=["user_id", "u_avg_basket"]) \
        .groupby("user_id")["u_avg_basket"].first()

    runs = {"all_features": FEATURES}
    runs.update({f"without_{g}": [c for c in FEATURES if c not in cols] for g, cols in FEATURE_GROUPS.items()})

    rows = []
    for name, feats in runs.items():
        t0 = time.time()
        gbm = train_lgbm(tr, tune, feats)
        p = predict(gbm, test, feats)
        scored = pd.DataFrame({"user_id": test.user_id, "product_id": test.product_id, "score": p})
        m = topk_metrics(scored, truth, (10,), basket_k)
        rows.append({"run": name, "n_features": len(feats),
                     "pr_auc": average_precision_score(test[LABEL], p), "roc_auc": roc_auc_score(test[LABEL], p),
                     "f1@basket": m["f1@basket"], "recall@10": m["recall@10"]})
        print(f"{name:<22} PR-AUC {rows[-1]['pr_auc']:.4f}  F1@basket {rows[-1]['f1@basket']:.4f}  "
              f"({time.time() - t0:.0f}s)")

    res = pd.DataFrame(rows).set_index("run")
    full = res.loc["all_features"]
    res["pr_auc_change"] = res["pr_auc"] - full["pr_auc"]
    res["f1@basket_change"] = res["f1@basket"] - full["f1@basket"]
    res.to_csv(ROOT / "reports" / "ablation_test.csv")
    print("\n=== Ablation (negative change = the removed group was useful) ===")
    print(res.round(4))


if __name__ == "__main__":
    main()
