"""Feature preparation for the Home Credit application table.

All functions are row-wise or take the training split as input, so they can be
reused at prediction time without touching validation or test data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "TARGET"
ID_COL = "SK_ID_CURR"
# DAYS_EMPLOYED uses 365243 for applicants with no current employment
# (mostly pensioners). It is a code, not a duration of 1,000 years.
DAYS_EMPLOYED_SENTINEL = 365243
# Protected attribute: kept out of the model by default (see README, Fairness).
PROTECTED = ["CODE_GENDER"]


def load_application(path: str | Path) -> pd.DataFrame:
    """Read application_train from .csv or .csv.zip."""
    return pd.read_csv(path)


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Domain ratios used in the original notebook, plus an explicit sentinel flag."""
    df = df.copy()
    df["DAYS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_SENTINEL, np.nan)
    income = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / income
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / income
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYMENT_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["INCOME_PER_PERSON"] = income / df["CNT_FAM_MEMBERS"]
    return df


def high_missing_columns(train: pd.DataFrame, threshold: float = 0.4) -> list[str]:
    """Columns with more than ``threshold`` missing **in the training split**."""
    rate = train.isna().mean()
    return sorted(rate[rate > threshold].index.tolist())


def feature_columns(df: pd.DataFrame, drop: list[str], include_protected: bool = False) -> list[str]:
    excluded = {TARGET, ID_COL, *drop}
    if not include_protected:
        excluded.update(PROTECTED)
    return [c for c in df.columns if c not in excluded]


def to_model_frame(df: pd.DataFrame, columns: list[str], categories: dict[str, list[str]]) -> pd.DataFrame:
    """Select columns and cast object columns to categories learned on train.

    Unknown categories at prediction time become missing instead of raising.
    """
    X = df[columns].copy()
    for col, cats in categories.items():
        if col in X:
            X[col] = pd.Categorical(X[col], categories=cats)
    return X


def learn_categories(train: pd.DataFrame, columns: list[str]) -> dict[str, list[str]]:
    obj = [c for c in columns if train[c].dtype == object or str(train[c].dtype) in ("str", "string")]
    return {c: sorted(train[c].dropna().unique().tolist()) for c in obj}
