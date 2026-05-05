# Customer Segmentation Classification — Loan Status Prediction

**Group Members:**
- 13523136 — Danendra Shafi Athallah
- 13523139 — Jonathan Kenan Budianto

**Class:** K03

---

## Overview

This project builds a machine learning system to predict loan approval decisions based on applicant financial and demographic data. The goal is to improve decision-making efficiency and reduce risk using **Ensemble Learning** methods.

The dataset contains **36,000 training samples** and **9,000 test samples** with 13 features including the target `loan_status`.

---

## Objectives

- Perform in-depth analysis of applicant financial data
- Build and compare **4 ensemble learning methods**
- Conduct hyperparameter tuning for optimal performance
- Identify the best model based on **Macro F1-Score**
- Understand misclassification patterns through Error Analysis

---

## Dataset

**Features in the dataset:**

| Feature | Description |
|---|---|
| `person_id` | Unique applicant ID |
| `person_age` | Applicant's age |
| `person_gender` | Gender |
| `person_income` | Annual income |
| `person_emp_exp` | Employment experience (years) |
| `person_home_ownership` | Home ownership status (OWN/RENT/MORTGAGE/OTHER) |
| `loan_amnt` | Loan amount requested |
| `loan_int_rate` | Loan interest rate |
| `loan_percent_income` | Loan-to-income ratio |
| `cb_person_cred_hist_length` | Credit history length (years) |
| `credit_score` | Applicant's credit score |
| `previous_loan_defaults_on_file` | Previous loan defaults on record (Yes/No) |

**Target:**
- `loan_status` — 0 (Rejected) / 1 (Approved)

**Class Distribution:**
- Rejected (0): 77.8% (28,000 samples)
- Approved (1): 22.2% (8,000 samples)

> The dataset exhibits **class imbalance** with a ~3.5:1 ratio

---

## Workflow

### 1. Exploratory Data Analysis (EDA)
- Dataset overview
- Visualization 1: Target class distribution (countplot)
- Visualization 2: Feature correlation heatmap
- Visualization 3: Boxplot of features by class (outlier detection)
- Visualization 4: Categorical feature distribution vs target

**Key Insights:**
- `loan_percent_income`, `previous_loan_defaults_on_file`, and `loan_int_rate` have the highest correlation with the target
- Significant outliers in `person_age`, `person_income`, and `loan_amnt`

### 2. Data Cleaning and Preprocessing
- **Missing Values**: None (dataset is clean)
- **Outliers**: Handled via IQR Clipping to bounds (Q1-1.5×IQR, Q3+1.5×IQR)
- **Encoding**: Label Encoding for categorical features
- **Feature Scaling**: Not performed (tree-based models do not require scaling)
- **Train-Validation Split**: 80/20 with stratification

### 3. Modeling and Validation

> **Primary Metric:** Macro F1-Score

| No | Model | Type | Macro F1-Score |
|---|---|---|---|
| 1 | **Gradient Boosting (Tuned)** | Boosting | **0.8836** ✅ |
| 2 | Voting (Soft) | Voting | 0.8800 |
| 3 | Random Forest (Bagging) | Bagging | 0.8735 |
| 4 | Stacking Ensemble | Stacking | 0.8586 |

**Model Details:**

- **Random Forest (Bagging)**: 200 estimators, max_depth=15, class_weight='balanced'
- **Gradient Boosting (Tuned)**: Hyperparameter tuning with RandomizedSearchCV (100 iterations, 5-fold CV)
  - Best params: n_estimators=500, learning_rate=0.1, max_depth=7, subsample=0.9
- **Stacking Ensemble**: Base learners (Logistic Regression, Decision Tree, KNN) + Logistic Regression meta-learner
- **Voting (Soft)**: Random Forest + Gradient Boosting + Logistic Regression

### 4. Error Analysis

**Confusion Matrix (Best Model):**
- Total Error: 7.8% (562/7200 samples)
- False Negative: 347 (Rejected when should be Approved — lost revenue opportunity)
- False Positive: 215 (Approved when should be Rejected — credit risk)

**Top Most Influential Features:**
1. `loan_percent_income` — Loan-to-income ratio
2. `previous_loan_defaults_on_file` — Previous default history
3. `person_income` — Applicant's income
4. `loan_int_rate` — Loan interest rate
5. `person_emp_exp` — Employment experience

### 5. Kaggle Submission
- Final Model: Gradient Boosting (Tuned)
- Macro F1-Score (Validation): **0.8836**
- Output file: `submission.csv` (9,000 predictions)

---

## Conclusion

**Best Model: Gradient Boosting (Tuned)**

| Metric | Value |
|---|---|
| Macro F1-Score (Validation) | 0.8836 |
| Accuracy | 92% |
| Precision (Approved) | 0.85 |
| Recall (Approved) | 0.78 |

**Recommended Improvements:**
1. Threshold tuning to optimize False Positive / False Negative trade-off
2. Additional feature engineering (ratio features, interactions)
3. More complex model ensemble (XGBoost + LightGBM stacking)
4. More aggressive class imbalance handling (SMOTE or class weight tuning)

---

## Tech Stack

- **Language:** Python
- **ML Library:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Hyperparameter Tuning:** RandomizedSearchCV
