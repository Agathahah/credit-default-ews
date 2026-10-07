# ============================================================
# drift_detector.py
# Deteksi data drift menggunakan Population Stability Index (PSI)
#
# Dataset: Home Credit Default Risk (Kaggle)
# 307,511 loan applications · 122 features · 8.07% default rate
#
# PSI < 0.1        → Stabil
# 0.1 <= PSI < 0.2 → Warning
# PSI >= 0.2       → Critical, retrain segera
# ============================================================

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    breakpoints = np.unique(np.percentile(expected, np.linspace(0, 100, buckets + 1)))
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct   = np.histogram(actual,   bins=breakpoints)[0] / len(actual)
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct   = np.where(actual_pct   == 0, 1e-6, actual_pct)
    return round(float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))), 6)


def interpret_psi(psi: float) -> dict:
    if psi < 0.1:
        return {"status": "STABLE",   "level": "green",  "action": "No action required."}
    elif psi < 0.2:
        return {"status": "WARNING", "level": "yellow",
                "action": "Increase monitoring frequency. Consider model re-evaluation."}
    else:
        return {"status": "CRITICAL", "level": "red",
                "action": "Investigate the shifted features; retrain if performance has dropped."}


class DriftDetector:
    """
    Deteksi data drift untuk model credit default (Home Credit Default Risk).

    Memonitor top features berdasarkan SHAP importance dari model XGBoost
    yang ditraining di notebooks/02_modeling.ipynb dan dijelaskan
    di notebooks/03_explainability.ipynb.
    """

    # Numeric inputs that the model actually uses (EXT_SOURCE_1 is not
    # monitored: it is >40% missing and dropped before training).
    FEATURES_TO_MONITOR = [
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
    ]

    def __init__(self, reference_data: pd.DataFrame, output_dir: str = "monitoring/reports"):
        self.reference_data = reference_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DriftDetector initialized. Reference: {len(reference_data):,} rows")

    def detect(self, production_data: pd.DataFrame, features: Optional[list] = None,
               simulated: bool = False) -> dict:
        features  = features or self.FEATURES_TO_MONITOR
        available = [f for f in features
                     if f in self.reference_data.columns and f in production_data.columns]

        if not available:
            raise ValueError("No matching features found. Check dataset schema.")

        logger.info(f"Checking drift for {len(available)} features...")
        results, critical_list, warning_list = {}, [], []

        for feature in available:
            ref_vals  = self.reference_data[feature].dropna().values
            prod_vals = production_data[feature].dropna().values
            psi       = calculate_psi(ref_vals, prod_vals)
            interp    = interpret_psi(psi)

            results[feature] = {
                "psi": psi, **interp,
                "reference_mean":  round(float(np.mean(ref_vals)), 4),
                "production_mean": round(float(np.mean(prod_vals)), 4),
                "mean_shift_pct":  round(
                    abs(np.mean(prod_vals) - np.mean(ref_vals))
                    / (abs(np.mean(ref_vals)) + 1e-9) * 100, 2
                )
            }

            if interp["status"] == "CRITICAL":
                critical_list.append(feature)
                logger.warning(f"  CRITICAL: {feature} (PSI={psi:.4f})")
            elif interp["status"] == "WARNING":
                warning_list.append(feature)
                logger.info(f"  WARNING:  {feature} (PSI={psi:.4f})")
            else:
                logger.info(f"  STABLE:   {feature} (PSI={psi:.4f})")

        overall_psi = float(np.mean([r["psi"] for r in results.values()]))
        # The overall status is the worst feature status. Averaging PSI would
        # let one CRITICAL feature hide behind nine stable ones.
        worst = max(results.values(), key=lambda r: r["psi"])["psi"]
        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset": "Home Credit Default Risk",
            # True when the "production" sample is a hold-out slice with drift
            # injected on purpose. Such a report demonstrates the detector; it
            # says nothing about real-world drift.
            "simulated_drift": simulated,
            "summary": {
                "overall_psi":            round(overall_psi, 6),
                "overall_status":         interpret_psi(worst)["status"],
                "max_feature_psi":        round(worst, 6),
                "total_features_checked": len(available),
                "critical_features":      critical_list,
                "warning_features":       warning_list,
                "stable_features":        len(available) - len(critical_list) - len(warning_list),
                "reference_rows":         len(self.reference_data),
                "production_rows":        len(production_data),
                "recommendation":         interpret_psi(worst)["action"]
            },
            "features": results
        }

        s = report["summary"]
        logger.info("=" * 52)
        logger.info(f"Overall PSI    : {s['overall_psi']:.4f} — {s['overall_status']}")
        logger.info(
            f"Critical: {len(s['critical_features'])} | Warning: {len(s['warning_features'])}"
            f" | Stable: {s['stable_features']}"
        )
        logger.info(f"Recommendation : {s['recommendation']}")
        logger.info("=" * 52)
        return report

    def save_report(self, report: dict, filename: Optional[str] = None) -> str:
        if not filename:
            filename = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved: {path}")
        return str(path)


if __name__ == "__main__":
    import os

    DATA_PATH = os.getenv("DATA_PATH", "data/application_train.csv")

    if not Path(DATA_PATH).exists():
        print(f"\nDataset not found: {DATA_PATH}")
        print("Download dataset first:")
        print("  kaggle competitions download -c home-credit-default-risk")
        print("  unzip home-credit-default-risk.zip -d data/")
        exit(1)

    logger.info(f"Loading {DATA_PATH}...")
    cols = DriftDetector.FEATURES_TO_MONITOR + ["TARGET"]
    df   = pd.read_csv(DATA_PATH, usecols=[c for c in cols
                                           if c in pd.read_csv(DATA_PATH, nrows=0).columns])
    logger.info(f"Loaded: {len(df):,} rows (missing values are ignored per feature)")

    split        = int(len(df) * 0.7)
    df_reference = df.iloc[:split].copy()
    df_production = df.iloc[split:].copy()

    # DEMO ONLY: there is no production traffic for this project. The last 30%
    # of rows plays "production" and drift is injected into two features so the
    # detector has something to find. The report is marked simulated_drift=True.
    np.random.seed(42)
    if "AMT_INCOME_TOTAL" in df_production.columns:
        df_production["AMT_INCOME_TOTAL"] *= np.random.uniform(0.85, 1.15, len(df_production))
    if "DAYS_EMPLOYED" in df_production.columns:
        df_production["DAYS_EMPLOYED"] += np.random.normal(30, 15, len(df_production))

    detector = DriftDetector(reference_data=df_reference)
    report   = detector.detect(production_data=df_production, simulated=True)
    path     = detector.save_report(report)
    print(f"\nReport saved: {path}")
