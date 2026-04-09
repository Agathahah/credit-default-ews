# Credit Default Early Warning System
### *Because a model that can't explain itself won't get used*

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## The Problem

Every year, lending companies write off billions in bad debt — not because they lack predictive models, but because those models can't explain their decisions to the humans who need to act on them.

A model that outputs `probability: 0.73` gets ignored.

A model that says **"this applicant's debt-to-income ratio is 3.2x their peer group, employment tenure is 4 months vs. median 24 months, and external credit score is in the bottom 10th percentile"** gets used.

**This project builds the second kind of model.**

---

## My Hypothesis

Most credit scoring models are optimized for AUC on paper — not for decisions that survive scrutiny in a credit committee. A model with solid AUC that **explains every prediction** is more valuable than a black-box model with marginally higher AUC.

This project proves that explainability and performance can coexist.

---

## Results

### Model Performance

| Model | ROC-AUC | PR-AUC | Notes |
|-------|---------|--------|-------|
| Logistic Regression (baseline) | ~0.69 | ~0.20 | Linear baseline |
| Random Forest | ~0.73 | ~0.25 | Ensemble, no explainability |
| **XGBoost (final)** | **0.7632** | **0.2531** | Best performance + SHAP ready |

> **Why PR-AUC matters more than ROC-AUC here:** With only 8.07% default rate, a random classifier scores ROC-AUC of 0.50 but PR-AUC of only 0.08. XGBoost's PR-AUC of 0.2531 means it is **3.1× better than random** at identifying actual defaulters — which is what credit analysts need.

---

## Key Findings

**Finding 1 — The Imbalance Problem Is a Metric Problem**
Default rate is 8.07%. Optimizing for accuracy gives a useless model that predicts "no default" always with 92% accuracy. PR-AUC is the correct metric — and at 0.2531, this model is 3× better than random on the class that actually matters.

**Finding 2 — Employment Type Predicts Default Better Than Income Level**
Raw income is weakly correlated with default. Employment type and tenure are far stronger predictors. A high-income applicant with 3 months of employment is riskier than a moderate-income applicant with 5+ years of stable employment.

**Finding 3 — External Scores Are Powerful But Have a Coverage Gap**
EXT_SOURCE features (external credit bureau scores) dominate all other predictors in SHAP importance. The problem: they only exist for applicants with prior credit history. For new-to-credit applicants — the exact segment fintech is trying to serve — these scores are unavailable. A robust model must perform without them.

**Finding 4 — Three Distinct Risk Personas Emerge from SHAP**

| Risk Tier | Avg Risk Probability | Key Profile Signals |
|-----------|---------------------|---------------------|
| 🟢 Low Risk | < 30% | High EXT_SOURCE, 5+ years employment, low credit-to-income ratio |
| 🟡 Medium Risk | 30–60% | Mixed signals — stable income but limited credit history |
| 🔴 High Risk | > 60% | Low EXT_SOURCE, < 1 year employment, high debt-to-income ratio |

**Finding 5 — The Most Expensive Mistake Is Rejecting Good Applicants**
Rejecting a creditworthy applicant is lost revenue AND a competitor acquisition. The optimal decision threshold should be calibrated based on the *business cost ratio* of false positives to false negatives — not just model performance metrics.

---

## What Makes This Different

| Most GitHub Projects | This Project |
|---------------------|--------------|
| Optimize for accuracy | Optimize for PR-AUC (correct metric) |
| Report ROC-AUC only | Report both ROC-AUC (0.7632) and PR-AUC (0.2531) |
| Feature importance only | SHAP per-applicant explanation |
| Technical findings | Business findings with decision recommendations |
| Model as endpoint | Model as decision support tool |

---

## Project Structure
credit-default-ews/
├── notebooks/
│   ├── 01_EDA.ipynb             # Who defaults? 5 business findings, 6 charts
│   ├── 02_modeling.ipynb        # XGBoost ROC-AUC 0.7632 | PR-AUC 0.2531
│   └── 03_explainability.ipynb  # SHAP: per-applicant risk explanations
├── reports/                     # 15 charts — EDA through SHAP
├── models/
│   └── xgb_credit_default.pkl  # Saved production model
├── requirements.txt
└── README.md

---

## How to Reproduce
```bash
git clone https://github.com/Agathahah/credit-default-ews.git
cd credit-default-ews
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Download: kaggle competitions download -c home-credit-default-risk
jupyter notebook notebooks/01_EDA.ipynb
```

---

## Tech Stack

`Python 3.12` · `XGBoost` · `LightGBM` · `SHAP` · `Scikit-learn` · `Plotly` · `Pandas` · `Imbalanced-learn`

---

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) — Kaggle  
307,511 loan applications · 122 features · **8.07% default rate**

---

*Author: Agatha Ulina Silalahi*  
*[LinkedIn](https://www.linkedin.com/in/agatha-silalahi-722507215/) · [Kaggle](https://www.kaggle.com/agathasilalahi) · [GitHub](https://github.com/Agathahah)*
