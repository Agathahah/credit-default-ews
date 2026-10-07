"""Evaluation helpers for imbalanced binary classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def summary(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y)
    p = np.asarray(p)
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "ks": round(float(ks_2samp(p[y == 1], p[y == 0]).statistic), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "mean_predicted": round(float(p.mean()), 4),
        "base_rate": round(float(y.mean()), 4),
    }


def decile_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Default rate by score decile (decile 10 = highest scores)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    out = df.groupby("decile").agg(n=("y", "size"), defaults=("y", "sum"),
                                   default_rate=("y", "mean"), mean_score=("p", "mean"))
    out["share_of_all_defaults"] = out["defaults"] / out["defaults"].sum()
    return out.round(4)


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index with quantile buckets taken from ``expected``."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
