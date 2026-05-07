# Model Monitoring — Credit Default EWS

Data drift detection module using Population Stability Index (PSI) to monitor production model health.

## Overview

Monitors feature distribution shift between training reference data and production data. Triggers alerts when drift exceeds defined thresholds, enabling proactive model retraining before performance degrades.

## PSI Thresholds

| PSI | Status | Action |
|---|---|---|
| < 0.1 | STABLE | No action required |
| 0.1 – 0.2 | WARNING | Increase monitoring frequency |
| ≥ 0.2 | CRITICAL | Retrain model immediately |

## Usage

```bash
# Run with default data path
python monitoring/drift_detector.py

# Run with custom data path
DATA_PATH=data/raw/cs-training.csv python monitoring/drift_detector.py
```

## Output

Reports are saved to `monitoring/reports/drift_report_YYYYMMDD_HHMMSS.json`:

```json
{
  "timestamp": "2026-05-07T10:30:00",
  "summary": {
    "overall_psi": 0.087,
    "overall_status": "STABLE",
    "critical_features": [],
    "warning_features": ["MonthlyIncome"],
    "recommendation": "Increase monitoring frequency"
  },
  "features": {
    "age": { "psi": 0.043, "status": "STABLE" },
    "MonthlyIncome": { "psi": 0.134, "status": "WARNING" }
  }
}
```

## Monitored Features

- `RevolvingUtilizationOfUnsecuredLines`
- `age`, `DebtRatio`, `MonthlyIncome`
- `NumberOfTime30-59DaysPastDueNotWorse`
- `NumberOfOpenCreditLinesAndLoans`
- `NumberOfTimes90DaysLate`
- `NumberRealEstateLoansOrLines`
- `NumberOfTime60-89DaysPastDueNotWorse`
- `NumberOfDependents`

## Integration

Can be scheduled via cron or GitHub Actions to run weekly drift checks automatically.
