import numpy as np

from src.metrics import decile_table, psi, summary


def test_psi_is_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, a) < 1e-9
    assert psi(a, a + 1.0) > 0.2


def test_decile_table_covers_all_rows_and_defaults():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=1000)
    y = (rng.uniform(size=1000) < p).astype(int)
    t = decile_table(y, p)
    assert t["n"].sum() == 1000
    assert np.isclose(t["share_of_all_defaults"].sum(), 1.0)
    assert t.loc[10, "default_rate"] > t.loc[1, "default_rate"]


def test_summary_random_scores_have_auc_near_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    s = summary(y, rng.uniform(size=4000))
    assert 0.45 < s["roc_auc"] < 0.55
