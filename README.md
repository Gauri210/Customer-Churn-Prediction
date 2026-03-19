# 📉 Customer Churn Prediction
### Kaggle — Telco Customer Churn

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat-square&logo=pandas&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-PGS6E3-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9087-brightgreen?style=flat-square)

---

## 🎯 Problem Statement

Customer churn — when a subscriber cancels their service — is one of the most expensive problems in the telecom industry. Acquiring a new customer costs **5–25× more** than retaining an existing one.

This project builds a **binary classification model** to predict whether a telecom customer will churn, using their demographics, account details, and service usage patterns.

> **Business Question:** *Given what we know about a customer's plan and behavior, how likely are they to leave?*

---

## 📊 Dataset

- **Source:** [Kaggle Playground Series Season 6, Episode 3](https://www.kaggle.com/competitions/playground-series-s6e3)
- **Train:** 594,194 rows × 21 columns (includes `Churn` target)
- **Test:** 254,655 rows × 20 columns
- **Class balance:** 77.5% No Churn / 22.5% Churn — imbalanced dataset
- **Evaluation metric:** ROC-AUC

### Feature Overview

| Category | Features |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account | `tenure`, `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Target | `Churn` (binary: 0 = Stay, 1 = Churn) |

---

## 🔍 Key EDA Findings

| Factor | Finding |
|---|---|
| **Contract type** | Month-to-month = **42% churn** vs Two-year = **1% churn** |
| **Tenure** | Churners leave at avg **17 months** vs stayers at **42 months** |
| **Monthly Charges** | Churners pay avg **₹81.6/month** vs ₹61.3 for stayers |
| **New customers** | Spike in churn at tenure = 1 month — first-month drop-off is significant |

> Contract type is the single strongest predictor of churn — month-to-month customers have zero friction to leave.

---

## ⚙️ Feature Engineering

Four domain-informed features were created on top of the raw data:

| Feature | Logic | Rationale |
|---|---|---|
| `IsNewCustomer` | tenure ≤ 6 → 1 else 0 | New customers are highest early-churn risk |
| `TotalServices` | Sum of 8 service flags | More services = more embedded in ecosystem |
| `ChargesPerTenure` | MonthlyCharges / (tenure + 1) | Normalized spend relative to loyalty |
| `ChargesPerService` | MonthlyCharges / (TotalServices + 1) | Effective cost per service — high = overpriced feel |

---

## 🤖 Model

### Logistic Regression

Logistic Regression was chosen for its interpretability and strong probabilistic output — essential for a business use case where scores feed into retention workflows.

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
```

**Why Logistic Regression?**
- Coefficients are directly interpretable as feature impact on churn probability
- Outputs calibrated probabilities — ideal for ranking customers by churn risk
- Trains efficiently on large datasets (475K rows)

### Result

| Metric | Score |
|---|---|
| **ROC-AUC** | **0.9087** |

An AUC of 0.9087 means the model correctly ranks a random churner above a random non-churner **~91% of the time** — strong performance for a linear model on an imbalanced dataset.

---

## 💡 Business Recommendations

| Finding | Recommended Action |
|---|---|
| Month-to-month churn = 42% | Offer contract-upgrade incentives at the 3-month mark |
| New customers at high risk | Trigger proactive outreach in the first 6 months |
| High-paying, low-service customers churn more | Bundle discounts targeting single-service premium users |
| Fiber optic customers churn more than DSL | Investigate Fiber service quality and pricing |

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── Customer_Churn_Prediction.ipynb   # Main notebook
├── train.csv                         # Training data (from Kaggle)
├── test.csv                          # Test data (from Kaggle)
└── README.md
```

---

## 🚀 Potential Extensions

- [ ] **OneHotEncoding** — better suited for Logistic Regression than LabelEncoder
- [ ] **Regularization tuning** — experiment with `C` and L1/L2 penalty
- [ ] **XGBoost / LightGBM** — benchmark non-linear models
- [ ] **SHAP values** — explain individual predictions for retention teams
- [ ] **Threshold tuning** — optimize precision/recall tradeoff
- [ ] **Cross-validation** — 5-fold CV for more robust AUC estimates

---

## 🛠️ Tech Stack

- **Python 3.10+**
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — preprocessing, modeling, evaluation

---

## 👩‍💻 Author

**Gauri** | MBATech Data Science, NMIMS Mumbai
GitHub: [@gauri210](https://github.com/gauri210)
