# Customer Churn & Retention Risk Analysis

Predicting customer churn and quantifying the retention revenue at stake, using 594K+ telecom customer records.

## Motivation

Acquiring a new customer typically costs far more than retaining an existing one, so identifying which customers are at risk of leaving — and why — is a direct lever on revenue. This project builds a churn risk model and translates its output into a concrete retention recommendation, rather than stopping at a classification score.

## Data

- **Source:** Telecom customer dataset, 594K+ records
- **Target:** Binary churn flag (churned / retained)
- **Features:** Customer demographics, contract type, tenure, monthly cost, and service usage, plus 4 engineered features (including number of services used and cost per month)

## Methodology

| Stage | Approach |
|---|---|
| Preprocessing | Missing-value handling, categorical encoding, feature scaling |
| Feature engineering | 4 derived features capturing usage intensity and cost-per-service |
| Modeling | Logistic Regression, chosen for interpretability of coefficients over marginal accuracy gains |
| Evaluation | ROC-AUC, precision/recall trade-off analysis at multiple decision thresholds |

## Key Results

- **ROC-AUC: 0.9087** on held-out test data
- Churn rate is sharply concentrated by contract type: **42%** of month-to-month customers churn versus **~1%** of two-year contract customers
- The engineered cost-per-service feature was among the strongest predictors, suggesting price sensitivity — not just tenure — drives churn risk

## Business Recommendation

Month-to-month customers represent the highest-risk, highest-volume segment. Migrating a portion of this base onto annual contracts (via a modest price incentive) would be expected to reduce churn substantially in the segment responsible for the majority of revenue attrition — a targeted intervention rather than a blanket retention campaign.

## Tech Stack

Python · Pandas · scikit-learn · Logistic Regression · Matplotlib/Seaborn

## Limitations

- Logistic Regression was prioritized for interpretability; a gradient-boosted model would likely improve raw predictive accuracy at the cost of explainability
- The recommendation is directional (based on the dataset's aggregate patterns), not a causal estimate of contract-migration effects — a proper A/B test would be needed before rollout

---
Built by Gauri Deshmukh
