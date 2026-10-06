import numpy as np

from src.features import (
    DAYS_EMPLOYED_SENTINEL,
    add_ratios,
    feature_columns,
    high_missing_columns,
    learn_categories,
    to_model_frame,
)


def test_sentinel_days_employed_is_flagged_not_treated_as_duration(application):
    df, _ = application
    out = add_ratios(df)
    assert out["DAYS_EMPLOYED_ANOMALY"].sum() == 300
    assert not (out["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).any()
    assert out.loc[:299, "EMPLOYMENT_YEARS"].isna().all()


def test_high_missing_is_learned_from_train_only(application):
    df, _ = application
    assert high_missing_columns(df.iloc[:100].assign(EXT_SOURCE_2=np.nan)) == ["EXT_SOURCE_1", "EXT_SOURCE_2"]


def test_gender_excluded_by_default(application):
    df, _ = application
    assert "CODE_GENDER" not in feature_columns(df, drop=[])
    assert "CODE_GENDER" in feature_columns(df, drop=[], include_protected=True)


def test_unknown_category_becomes_missing(application):
    df, _ = application
    cols = ["NAME_CONTRACT_TYPE"]
    cats = learn_categories(df, cols)
    new = df.head(2).copy()
    new["NAME_CONTRACT_TYPE"] = ["Cash loans", "Brand new product"]
    X = to_model_frame(new, cols, cats)
    assert X["NAME_CONTRACT_TYPE"].isna().tolist() == [False, True]
