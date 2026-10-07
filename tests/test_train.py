import json

from src.train import run


def test_training_end_to_end_on_synthetic_data(application, tmp_path):
    _, path = application
    metrics = run(str(path), out=str(tmp_path / "r"), models_dir=str(tmp_path / "m"), plots=False)
    saved = json.loads((tmp_path / "r" / "metrics.json").read_text())
    assert saved["test"]["xgboost_raw"]["roc_auc"] > 0.6
    assert "CODE_GENDER" not in json.loads((tmp_path / "m" / "preprocessing.json").read_text())["columns"]
    assert sum(metrics["rows"].values()) == 3000
