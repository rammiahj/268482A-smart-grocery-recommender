"""Train and evaluate the replenishment models (RQ1, RQ2, RQ4).

Setup (no leakage):
  - TRAIN on features_val   (features from history, label = item in each user's val order)
      * users are split 90/10: the 10% "tuning users" are used for early stopping and
        to choose the decision threshold. The test set is never used for tuning.
  - EVALUATE on features_test (features from history + val, label = item in the test order),
    with exactly the same metrics as the rule-based baselines.

Models:
  - Logistic Regression  (median imputation + missing-value flags + scaling): simple, interpretable ML baseline
  - LightGBM             (gradient-boosted trees): improved model

Outputs:
  models/logreg.joblib, models/lgbm.txt
  reports/models_test.csv            metrics for both models
  reports/comparison_test.csv        baselines + models side by side
  reports/feature_importance.csv     LightGBM gain importance
  reports/figures/11_feature_importance.png

Usage:
  python -m src.models.train                 # trains on 50% of users (default, faster)
  python -m src.models.train --sample 1.0    # all users
"""
from __future__ import annotations

import argparse
import time

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_SEED, ROOT
from src.models.evaluate import KS, classification_metrics, topk_metrics

ID_COLS = ["user_id", "product_id", "aisle_id", "department_id"]
LABEL = "label"

FEATURE_GROUPS = {
    "user": ["u_orders", "u_avg_basket", "u_mean_order_gap", "u_reorder_ratio", "u_distinct_products"],
    "product": ["p_purchases", "p_users", "p_reorder_rate", "p_mean_cart_pos", "a_reorder_rate", "d_reorder_rate"],
    "frequency": ["up_count", "up_share", "up_rate", "up_mean_cart_pos"],
    "recency": ["up_orders_since_last", "up_in_last_order"],
    "timing": ["up_days_since_last", "up_mean_gap", "up_gap_cv", "up_due_ratio",
               "t_days_since_prior", "t_capped", "t_dow", "t_hour"],
}
FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    metric="average_precision", seed=RANDOM_SEED, verbose=-1, num_threads=0,
)


# ---------------------------------------------------------------- data
def load(target: str, sample: float = 1.0) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / f"features_{target}.parquet", columns=["user_id", "product_id", *FEATURES, LABEL])
    if sample < 1.0:
        users = df["user_id"].drop_duplicates().sample(frac=sample, random_state=RANDOM_SEED)
        df = df[df["user_id"].isin(users)]
    df[FEATURES] = df[FEATURES].astype("float32")
    return df.reset_index(drop=True)


def split_users(df: pd.DataFrame, frac: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    users = df["user_id"].unique()
    tune_users = set(rng.choice(users, size=max(1, int(len(users) * frac)), replace=False))
    is_tune = df["user_id"].isin(tune_users)
    return df[~is_tune], df[is_tune]


# ---------------------------------------------------------------- models
def train_logreg(tr: pd.DataFrame, features=FEATURES):
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=300, random_state=RANDOM_SEED),
    )
    model.fit(tr[features], tr[LABEL])
    return model


def train_lgbm(tr: pd.DataFrame, tune: pd.DataFrame, features=FEATURES, rounds: int = 2000):
    dtr = lgb.Dataset(tr[features], tr[LABEL])
    dva = lgb.Dataset(tune[features], tune[LABEL], reference=dtr)
    return lgb.train(LGB_PARAMS, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])


def predict(model, df: pd.DataFrame, features=FEATURES) -> np.ndarray:
    if isinstance(model, lgb.Booster):
        return model.predict(df[features], num_iteration=model.best_iteration)
    return model.predict_proba(df[features])[:, 1]


def tuned_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold that maximises F1 on the tuning users (never on test)."""
    return float(classification_metrics(y, p)["best_threshold"])


# ---------------------------------------------------------------- evaluation
def evaluate(name: str, p_test: np.ndarray, test: pd.DataFrame, truth: pd.DataFrame,
             basket_k: pd.Series, threshold: float) -> dict:
    scored = pd.DataFrame({"user_id": test.user_id, "product_id": test.product_id, "score": p_test})
    row = {"method": name}
    row.update(topk_metrics(scored, truth, KS, basket_k))
    row.update(classification_metrics(test[LABEL].values, p_test))
    # Candidate-level F1 at the threshold chosen on tuning users (honest, not tuned on test)
    y = test[LABEL].values
    pred = p_test >= threshold
    tp = (pred & (y == 1)).sum()
    prec = tp / max(pred.sum(), 1)
    rec = tp / max((y == 1).sum(), 1)
    row.update({"threshold_tuned": threshold, "precision_at_tuned": prec, "recall_at_tuned": rec,
                "f1_at_tuned": 2 * prec * rec / max(prec + rec, 1e-12)})
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=float, default=0.5, help="fraction of training users (default 0.5)")
    args = ap.parse_args()
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train = load("val", args.sample)
    tr, tune = split_users(train)
    test = load("test")
    truth = pd.read_parquet(PROCESSED_DIR / "truth_test.parquet")
    basket_k = pd.read_parquet(PROCESSED_DIR / "features_test.parquet", columns=["user_id", "u_avg_basket"]) \
        .groupby("user_id")["u_avg_basket"].first()
    print(f"train {len(tr):,} rows | tuning {len(tune):,} | test {len(test):,} "
          f"| features {len(FEATURES)} | loaded in {time.time() - t0:.0f}s")

    rows = []

    # Logistic Regression
    t0 = time.time()
    lr = train_logreg(tr)
    thr = tuned_threshold(tune[LABEL].values, predict(lr, tune))
    rows.append(evaluate("logistic_regression", predict(lr, test), test, truth, basket_k, thr))
    joblib.dump(lr, MODELS_DIR / "logreg.joblib")
    print(f"logistic regression done in {time.time() - t0:.0f}s")

    # LightGBM
    t0 = time.time()
    gbm = train_lgbm(tr, tune)
    thr = tuned_threshold(tune[LABEL].values, predict(gbm, tune))
    rows.append(evaluate("lightgbm", predict(gbm, test), test, truth, basket_k, thr))
    gbm.save_model(str(MODELS_DIR / "lgbm.txt"), num_iteration=gbm.best_iteration)
    print(f"lightgbm done in {time.time() - t0:.0f}s (best iteration {gbm.best_iteration}, "
          f"tuning PR-AUC {average_precision_score(tune[LABEL], predict(gbm, tune)):.4f})")

    res = pd.DataFrame(rows).set_index("method")
    res.to_csv(ROOT / "reports" / "models_test.csv")

    # Feature importance
    imp = pd.DataFrame({"feature": FEATURES,
                        "gain": gbm.feature_importance("gain", iteration=gbm.best_iteration)})
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    imp["group"] = imp["feature"].map({c: g for g, cols in FEATURE_GROUPS.items() for c in cols})
    imp = imp.sort_values("gain", ascending=False)
    imp.to_csv(ROOT / "reports" / "feature_importance.csv", index=False)
    plt.figure(figsize=(8, 7))
    plt.barh(imp["feature"][::-1], imp["gain_share"][::-1], color="steelblue")
    plt.title("LightGBM feature importance (share of total gain)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "11_feature_importance.png", dpi=150)
    plt.close()

    # Comparison with baselines
    base_path = ROOT / "reports" / "baselines_test.csv"
    comp = pd.concat([pd.read_csv(base_path, index_col="method"), res]) if base_path.exists() else res
    comp.to_csv(ROOT / "reports" / "comparison_test.csv")

    show = ["precision@basket", "recall@basket", "f1@basket", "hit_rate@basket",
            "precision@10", "recall@10", "f1@10", "recall@20", "roc_auc", "pr_auc", "f1_at_tuned"]
    pd.set_option("display.width", 200)
    print("\n=== Test results: baselines vs models ===")
    print(comp[[c for c in show if c in comp.columns]].T.round(4))
    print("\nTop 10 features (LightGBM gain share):")
    print(imp.head(10)[["feature", "group", "gain_share"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
