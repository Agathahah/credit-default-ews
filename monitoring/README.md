# Drift monitoring (demo)

`drift_detector.py` computes the Population Stability Index (PSI) between a reference sample and a newer sample for the numeric inputs of the Home Credit model.

| PSI | Status | Suggested action |
|---|---|---|
| < 0.1 | STABLE | none |
| 0.1 – 0.2 | WARNING | look at the feature, monitor more often |
| ≥ 0.2 | CRITICAL | investigate; consider retraining |

## Important: the shipped report is a simulation

This project has no production traffic. The script splits `application_train` into a 70% reference and a 30% "production" slice and **injects drift on purpose** (income × U(0.85, 1.15), employment days + N(30, 15)). The resulting report is marked `"simulated_drift": true`. A WARNING on `DAYS_EMPLOYED` in that report shows that the detector works; it is not a finding about the data.

```bash
DATA_PATH=data/application_train.csv.zip python monitoring/drift_detector.py
```

Real monitoring would compare the training sample with each new batch of applications, run on a schedule (cron, Airflow or a GitHub Action), and alert when a feature crosses the CRITICAL threshold.
