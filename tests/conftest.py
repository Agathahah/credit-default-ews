"""Synthetic data with the Home Credit column names used by src/."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def application(tmp_path):
    rng = np.random.default_rng(1)
    n = 3000
    ext2 = rng.uniform(0, 1, n)
    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(n),
        "AMT_INCOME_TOTAL": rng.lognormal(12, 0.5, n),
        "AMT_CREDIT": rng.lognormal(13, 0.6, n),
        "AMT_ANNUITY": rng.lognormal(10, 0.4, n),
        "AMT_GOODS_PRICE": rng.lognormal(12.9, 0.6, n),
        "DAYS_BIRTH": -rng.integers(21 * 365, 69 * 365, n),
        "DAYS_EMPLOYED": -rng.integers(0, 40 * 365, n),
        "CNT_FAM_MEMBERS": rng.integers(1, 6, n).astype(float),
        "CODE_GENDER": rng.choice(["F", "M"], n),
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
        "EXT_SOURCE_1": np.where(rng.uniform(size=n) < 0.6, np.nan, rng.uniform(0, 1, n)),
        "EXT_SOURCE_2": ext2,
    })
    df.loc[:299, "DAYS_EMPLOYED"] = 365243
    df["TARGET"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-1.5 - 3 * ext2)))).astype(int)
    path = tmp_path / "application_train.csv"
    df.to_csv(path, index=False)
    return df, path
