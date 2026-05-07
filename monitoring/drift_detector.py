# ============================================================
# drift_detector.py
# Deteksi data drift menggunakan Population Stability Index (PSI)
#
# PSI mengukur seberapa besar distribusi data produksi bergeser
# dari distribusi data training.
#
# Interpretasi PSI:
# PSI < 0.1   → Tidak ada drift (aman)
# 0.1 <= PSI < 0.2 → Drift sedang (perlu dimonitor)
# PSI >= 0.2  → Drift signifikan (model perlu retrain)
# ============================================================

import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10
) -> float:
    """
    Hitung Population Stability Index (PSI) antara dua distribusi.

    PSI = sum((actual% - expected%) * ln(actual% / expected%))

    Args:
        expected: distribusi referensi (data training)
        actual:   distribusi yang diperiksa (data produksi)
        buckets:  jumlah bin untuk histogram

    Returns:
        nilai PSI (float)
    """
    # Buat breakpoints berdasarkan distribusi expected
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)  # hapus duplikat

    # Hitung frekuensi per bin
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts   = np.histogram(actual,   bins=breakpoints)[0]

    # Konversi ke proporsi, hindari division by zero
    expected_pct = expected_counts / len(expected)
    actual_pct   = actual_counts   / len(actual)

    # Hindari log(0) — ganti 0 dengan nilai kecil
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct   = np.where(actual_pct   == 0, 1e-6, actual_pct)

    # Hitung PSI
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = np.sum(psi_values)

    return round(float(psi), 6)


def interpret_psi(psi: float) -> dict:
    """
    Interpretasikan nilai PSI menjadi status dan rekomendasi.
    """
    if psi < 0.1:
        return {
            "status": "STABLE",
            "level": "green",
            "message": "Distribusi data stabil. Model masih valid.",
            "action": "Tidak ada tindakan yang diperlukan."
        }
    elif psi < 0.2:
        return {
            "status": "WARNING",
            "level": "yellow",
            "message": "Ada pergeseran distribusi sedang.",
            "action": "Monitor lebih ketat. Pertimbangkan evaluasi ulang model."
        }
    else:
        return {
            "status": "CRITICAL",
            "level": "red",
            "message": "Drift signifikan terdeteksi!",
            "action": "Segera retrain model dengan data terbaru."
        }


class DriftDetector:
    """
    Kelas utama untuk mendeteksi dan melaporkan data drift.

    Cara pakai:
        detector = DriftDetector(reference_data=df_train)
        report = detector.detect(production_data=df_prod)
        detector.save_report(report)
    """

    # Fitur yang dimonitor — sesuaikan dengan feature store kamu
    FEATURES_TO_MONITOR = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents"
    ]

    def __init__(
        self,
        reference_data: pd.DataFrame,
        output_dir: str = "monitoring/reports"
    ):
        self.reference_data = reference_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DriftDetector initialized. Reference data: {len(reference_data)} rows")

    def detect(
        self,
        production_data: pd.DataFrame,
        features: Optional[list] = None
    ) -> dict:
        """
        Jalankan deteksi drift untuk semua fitur.

        Args:
            production_data: data dari produksi yang ingin diperiksa
            features: list fitur yang ingin diperiksa (default: semua)

        Returns:
            report dict berisi PSI per fitur dan summary keseluruhan
        """
        features = features or self.FEATURES_TO_MONITOR
        # Filter hanya fitur yang ada di kedua dataset
        available = [f for f in features
                     if f in self.reference_data.columns
                     and f in production_data.columns]

        if not available:
            raise ValueError("Tidak ada fitur yang cocok antara reference dan production data.")

        logger.info(f"Memeriksa drift untuk {len(available)} fitur...")

        results = {}
        critical_features = []
        warning_features  = []

        for feature in available:
            ref_values  = self.reference_data[feature].dropna().values
            prod_values = production_data[feature].dropna().values

            psi         = calculate_psi(ref_values, prod_values)
            interpreted = interpret_psi(psi)

            results[feature] = {
                "psi": psi,
                **interpreted,
                "reference_mean": round(float(np.mean(ref_values)), 4),
                "production_mean": round(float(np.mean(prod_values)), 4),
                "mean_shift_pct": round(
                    abs(np.mean(prod_values) - np.mean(ref_values))
                    / (abs(np.mean(ref_values)) + 1e-9) * 100, 2
                )
            }

            if interpreted["status"] == "CRITICAL":
                critical_features.append(feature)
                logger.warning(f"  ⚠️  CRITICAL drift: {feature} (PSI={psi:.4f})")
            elif interpreted["status"] == "WARNING":
                warning_features.append(feature)
                logger.info(f"  🟡 WARNING drift: {feature} (PSI={psi:.4f})")
            else:
                logger.info(f"  ✅ STABLE: {feature} (PSI={psi:.4f})")

        # Summary keseluruhan
        overall_psi = np.mean([r["psi"] for r in results.values()])

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "overall_psi": round(float(overall_psi), 6),
                "overall_status": interpret_psi(overall_psi)["status"],
                "total_features_checked": len(available),
                "critical_features": critical_features,
                "warning_features": warning_features,
                "stable_features": len(available) - len(critical_features) - len(warning_features),
                "reference_rows": len(self.reference_data),
                "production_rows": len(production_data),
                "recommendation": interpret_psi(overall_psi)["action"]
            },
            "features": results
        }

        self._log_summary(report)
        return report

    def _log_summary(self, report: dict) -> None:
        summary = report["summary"]
        logger.info("=" * 50)
        logger.info("DRIFT DETECTION REPORT")
        logger.info("=" * 50)
        logger.info(f"Overall PSI    : {summary['overall_psi']:.4f}")
        logger.info(f"Overall status : {summary['overall_status']}")
        logger.info(f"Critical       : {len(summary['critical_features'])} fitur")
        logger.info(f"Warning        : {len(summary['warning_features'])} fitur")
        logger.info(f"Stable         : {summary['stable_features']} fitur")
        logger.info(f"Rekomendasi    : {summary['recommendation']}")
        logger.info("=" * 50)

    def save_report(self, report: dict, filename: Optional[str] = None) -> str:
        """Simpan report ke JSON file."""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drift_report_{ts}.json"

        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved: {path}")
        return str(path)


# ============================================================
# Main: contoh penggunaan dengan dataset kredit
# ============================================================
if __name__ == "__main__":
    import os

    DATA_PATH = os.getenv("DATA_PATH", "data/raw/cs-training.csv")

    logger.info("Loading dataset...")
    df = pd.read_csv(DATA_PATH).drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.dropna()

    # Simulasi: gunakan 70% pertama sebagai reference (training)
    # dan 30% sisanya sebagai production data
    split = int(len(df) * 0.7)
    df_reference  = df.iloc[:split].copy()
    df_production = df.iloc[split:].copy()

    logger.info(f"Reference data : {len(df_reference)} rows")
    logger.info(f"Production data: {len(df_production)} rows")

    # Simulasi drift: tambahkan noise ke production data
    # Ini mensimulasikan kondisi nyata di mana data produksi
    # mulai bergeser dari distribusi training
    np.random.seed(42)
    df_production["age"] = df_production["age"] + np.random.normal(3, 2, len(df_production))
    df_production["MonthlyIncome"] = df_production["MonthlyIncome"] * np.random.uniform(0.85, 1.15, len(df_production))

    # Jalankan deteksi
    detector = DriftDetector(reference_data=df_reference)
    report   = detector.detect(production_data=df_production)

    # Simpan report
    saved_path = detector.save_report(report)
    print(f"\n✅ Report saved to: {saved_path}")
