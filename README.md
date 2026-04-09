# Credit Default Early Warning System
### *Because a model that can't explain itself won't get used*

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## The Problem

Every year, lending companies write off billions in bad debt — not because they lack models, but because those models can't explain their decisions to the humans who need to act on them.

A model that outputs `probability: 0.73` gets ignored.

A model that says **"this applicant's debt-to-income ratio is 3.2x their peer group, employment tenure is 4 months vs. median 24 months"** gets used.

**This project builds the second kind of model.**

## My Hypothesis

Most credit scoring models are optimized for AUC on paper — not for decisions that survive scrutiny in a credit committee. A model with AUC 0.78 that explains every prediction is more valuable than a black-box AUC 0.85 model. This project tests that hypothesis.

## Approach

| Phase | Notebook | Status |
|-------|----------|--------|
| 1. Exploratory Analysis | `01_EDA.ipynb` | 🚧 In Progress |
| 2. Modeling & Evaluation | `02_modeling.ipynb` | ⏳ Pending |
| 3. Explainability (SHAP) | `03_explainability.ipynb` | ⏳ Pending |

## Key Questions Answered

1. **Who is most likely to default?** → XGBoost prediction model
2. **Why are they risky?** → SHAP values per applicant
3. **What features matter most?** → Global feature importance
4. **What's the business cost of getting this wrong?** → Precision-Recall tradeoff

## Tech Stack

`Python` · `XGBoost` · `LightGBM` · `SHAP` · `Scikit-learn` · `Plotly` · `Pandas`

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) — Kaggle
307,511 applications · 122 features · ~8% default rate

## How to Reproduce
```bash
git clone https://github.com/Agathahah/credit-default-ews.git
cd credit-default-ews
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_EDA.ipynb
```

*Author: Agatha Ulina Silalahi — [LinkedIn](https://www.linkedin.com/in/agatha-silalahi-722507215/) · [Kaggle](https://www.kaggle.com/agathasilalahi)*
