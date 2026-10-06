"""Reproducible training script for the credit default early-warning model.

Run:
    python -m src.train --data data/application_train.csv.zip

Protocol
--------
1. 60/20/20 stratified split into train / validation / test (seed 42).
2. Everything learned from data (which columns are too sparse, category
   levels, imputation and scaling for the baseline) is learned on **train**.
3. XGBoost early-stops on validation. Isotonic calibration is fitted on
   validation. The test split is scored once.
4. CODE_GENDER is excluded by default; ``--include-protected`` trains the
   comparison model so the cost of excluding it is measured, not assumed.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from src.features import (  # noqa: E402
    TARGET,
    add_ratios,
    feature_columns,
    high_missing_columns,
    learn_categories,
    load_application,
    to_model_frame,
)
from src.metrics import decile_table, summary  # noqa: E402

SEED = 42
XGB_PARAMS = dict(
    n_estimators=2000, learning_rate=0.03, max_depth=5, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0, tree_method="hist",
    enable_categorical=True, max_cat_to_onehot=1, eval_metric="aucpr",
    early_stopping_rounds=100, random_state=SEED, n_jobs=-1,
)


def split(df: pd.DataFrame):
    tmp, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[TARGET])
    train, val = train_test_split(tmp, test_size=0.25, random_state=SEED, stratify=tmp[TARGET])
    return train, val, test


def fit_baseline(train: pd.DataFrame, cols: list[str], cats: dict[str, list[str]]):
    num = [c for c in cols if c not in cats]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=50)), list(cats)),
    ])
    model = make_pipeline(pre, LogisticRegression(max_iter=3000, C=0.5))
    X = train[cols].copy()
    for c in cats:
        X[c] = X[c].astype(object)
    model.fit(X, train[TARGET])
    return model


def baseline_proba(model, df: pd.DataFrame, cols: list[str], cats: dict[str, list[str]]) -> np.ndarray:
    X = df[cols].copy()
    for c in cats:
        X[c] = X[c].astype(object)
    return model.predict_proba(X)[:, 1]


def fit_xgb(train, val, cols, cats) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(to_model_frame(train, cols, cats), train[TARGET],
              eval_set=[(to_model_frame(val, cols, cats), val[TARGET])], verbose=False)
    return model


def run(data: str, out: str = "reports", models_dir: str = "models", sample: int | None = None,
        plots: bool = True) -> dict:
    out_dir, model_dir = Path(out), Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = add_ratios(load_application(data))
    if sample:
        df = df.sample(sample, random_state=SEED)
    train, val, test = split(df)

    drop = high_missing_columns(train)
    cols = feature_columns(train, drop)
    cats = learn_categories(train, cols)
    cols_g = feature_columns(train, drop, include_protected=True)
    cats_g = learn_categories(train, cols_g)

    baseline = fit_baseline(train, cols, cats)
    p_base = baseline_proba(baseline, test, cols, cats)

    model = fit_xgb(train, val, cols, cats)
    p_val = model.predict_proba(to_model_frame(val, cols, cats))[:, 1]
    p_test = model.predict_proba(to_model_frame(test, cols, cats))[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_val, val[TARGET])
    p_test_cal = iso.predict(p_test)

    model_g = fit_xgb(train, val, cols_g, cats_g)
    p_test_g = model_g.predict_proba(to_model_frame(test, cols_g, cats_g))[:, 1]

    deciles = decile_table(test[TARGET].values, p_test_cal)
    deciles.to_csv(out_dir / "decile_table.csv")

    importance = (pd.Series(model.get_booster().get_score(importance_type="gain"))
                  .sort_values(ascending=False).head(15).round(2))

    metrics = {
        "dataset": "Home Credit Default Risk, application_train",
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "default_rate": round(float(df[TARGET].mean()), 4),
        "columns_dropped_over_40pct_missing_in_train": len(drop),
        "n_features": len(cols),
        "protected_excluded": ["CODE_GENDER"],
        "test": {
            "logistic_regression_baseline": summary(test[TARGET], p_base),
            "xgboost_raw": summary(test[TARGET], p_test),
            "xgboost_calibrated": summary(test[TARGET], p_test_cal),
            "xgboost_with_gender": summary(test[TARGET], p_test_g),
        },
        "best_iteration": int(model.best_iteration),
        "top10pct_share_of_defaults": float(deciles.loc[10, "share_of_all_defaults"]),
        "top_features_by_gain": importance.to_dict(),
        "versions": {"python": platform.python_version(), "xgboost": xgb.__version__,
                     "scikit-learn": sklearn.__version__, "pandas": pd.__version__},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    model.save_model(model_dir / "xgb_credit_default.json")
    (model_dir / "preprocessing.json").write_text(json.dumps(
        {"columns": cols, "categories": cats, "dropped_columns": drop}, indent=2))

    if plots:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        RocCurveDisplay.from_predictions(test[TARGET], p_test, ax=axes[0], name="XGBoost")
        RocCurveDisplay.from_predictions(test[TARGET], p_base, ax=axes[0], name="Logistic baseline")
        axes[0].set_title("ROC (test)")
        PrecisionRecallDisplay.from_predictions(test[TARGET], p_test, ax=axes[1], name="XGBoost")
        PrecisionRecallDisplay.from_predictions(test[TARGET], p_base, ax=axes[1], name="Logistic baseline")
        axes[1].axhline(test[TARGET].mean(), color="k", ls="--", lw=1)
        axes[1].set_title("Precision-recall (test)")
        axes[2].plot(deciles["mean_score"], deciles["default_rate"], "o-", label="calibrated XGBoost")
        lim = max(deciles["default_rate"].max(), deciles["mean_score"].max()) * 1.05
        axes[2].plot([0, lim], [0, lim], "k--", lw=1, label="perfect calibration")
        axes[2].set(xlabel="Mean predicted probability (decile)", ylabel="Observed default rate",
                    title="Calibration by decile (test)")
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "16_test_evaluation.png", dpi=130)
        plt.close(fig)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/application_train.csv.zip")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--models", default="models")
    ap.add_argument("--sample", type=int, default=None, help="subsample rows for a quick run")
    args = ap.parse_args()
    metrics = run(args.data, args.out, args.models, args.sample)
    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()
