"""Evaluation framework shared by baselines and ML models.

Two views of quality:

1. Ranking / shopping-list metrics (per user, then averaged over all users in the target order):
     Precision@K  share of the K suggested items that were actually bought
     Recall@K     share of the actual basket that was suggested (denominator = the WHOLE basket,
                  including new items, so replenishment-only methods are capped at ~59%)
     HitRate@K    share of users with at least one correct suggestion
     F1@K         harmonic mean of the per-user Precision@K and Recall@K
   K = 5, 10, 20 and "basket" (each user's own average basket size, rounded).

2. Candidate-level classification metrics (only for methods that score every candidate):
     ROC-AUC, PR-AUC (average precision), and Precision / Recall / F1 at the best-F1 threshold.
     PR-AUC is the main metric because only ~10% of candidates are positive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

KS = (5, 10, 20)


def topk_metrics(scored: pd.DataFrame, truth: pd.DataFrame, ks=KS,
                 basket_k: pd.Series | None = None) -> dict:
    """scored: user_id, product_id, score (higher = more likely)
    truth: user_id, product_id of items actually bought in the target order
    basket_k: optional Series indexed by user_id with a personalised K."""
    ranked = scored.sort_values(["user_id", "score", "product_id"], ascending=[True, False, True])
    ranked = ranked.assign(rank=ranked.groupby("user_id").cumcount() + 1)
    ranked = ranked.merge(truth.assign(hit=1), on=["user_id", "product_id"], how="left")
    ranked["hit"] = ranked["hit"].fillna(0)

    basket = truth.groupby("user_id").size().rename("basket_size")
    users = basket.index
    out = {}

    def _summarise(hits: pd.Series, k_per_user: pd.Series, label: str) -> None:
        hits = hits.reindex(users, fill_value=0)
        p = hits / k_per_user.reindex(users)
        r = hits / basket
        f1 = (2 * p * r / (p + r)).fillna(0)
        out[f"precision@{label}"] = p.mean()
        out[f"recall@{label}"] = r.mean()
        out[f"f1@{label}"] = f1.mean()
        out[f"hit_rate@{label}"] = (hits > 0).mean()

    for k in ks:
        hits = ranked.loc[ranked["rank"] <= k].groupby("user_id")["hit"].sum()
        _summarise(hits, pd.Series(k, index=users), str(k))

    if basket_k is not None:
        kk = basket_k.round().clip(lower=1).astype(int)
        r2 = ranked.merge(kk.rename("k").reset_index(), on="user_id", how="left")
        hits = r2.loc[r2["rank"] <= r2["k"]].groupby("user_id")["hit"].sum()
        _summarise(hits, kk, "basket")
    return out


def classification_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    """Candidate-level metrics; threshold chosen to maximise F1."""
    prec, rec, thr = precision_recall_curve(y, score)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return {
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "best_f1": f1[i],
        "precision_at_best_f1": prec[i],
        "recall_at_best_f1": rec[i],
        "best_threshold": thr[i] if len(thr) else np.nan,
    }


def candidate_ceiling(features: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Best possible recall if every candidate (previously bought item) were suggested."""
    hits = features.loc[features["label"] == 1].groupby("user_id").size()
    basket = truth.groupby("user_id").size()
    return {"recall_ceiling_per_user": (hits.reindex(basket.index, fill_value=0) / basket).mean(),
            "recall_ceiling_overall": hits.sum() / basket.sum()}
