---
name: kaggle-eda-loop
description: Perform a systematic Exploratory Data Analysis (EDA) on a tabular dataset. Use when starting a new data exploration, analyzing dataset distributions, checking outliers, evaluating class balance, or computing linear/non-linear correlations before modeling.
---

# Exploratory Data Analysis (EDA) Loop

Use this skill to systematically analyze new datasets, preventing modeling errors and gaining solid insights about data quality.

## Quick start

Run a fast quality check using Python and pandas:
```python
import pandas as pd
df = pd.read_csv("data/raw/train.csv")
print(df.info())
print("Nulos:\n", df.isnull().sum())
print("Duplicados:", df.duplicated().sum())
```

## Workflows

### 1. Data Quality & Profiling Checklist
- [ ] **Data Types:** Check if numerical columns are parsed as floats/ints and categorical columns as objects/categories.
- [ ] **Missing Values:** Calculate the percentage of missing values per column. If >50%, consider drop/special indicators; otherwise, plan imputation (median, mode, or iterative).
- [ ] **Duplicates:** Identify and remove exact duplicated rows unless they represent distinct valid events.
- [ ] **Cardinality:** Check unique values in categorical columns. High cardinality (e.g., hash IDs) may need target encoding or grouping.

### 2. Distribution & Target Balance Check
- [ ] **Target Distribution:**
  - For **Classification**: Calculate the ratio of each class (e.g., `df['target'].value_counts(normalize=True)`). If highly imbalanced (<10% minority class), plan stratification, SMOTE, or adjust loss weights (class weights).
  - For **Regression**: Plot/check skewness. Highly skewed targets may benefit from a log transform `log1p` or Box-Cox.
- [ ] **Numerical Skewness:** Identify highly skewed features. Extreme distributions can heavily impact linear models and neural networks.

### 3. Outlier Analysis
- [ ] **IQR Method:** Identify outliers in continuous numerical columns:
  $$\text{IQR} = Q_3 - Q_1$$
  $$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$
- [ ] **Impact Mitigation:** Decide if outliers should be clipped (e.g., 1st and 99th percentiles), transformed, or if robust tree-based models (which handle outliers naturally) are sufficient.

### 4. Correlation & Multicollinearity
- [ ] **Pearson Correlation:** Generate a correlation matrix for linear relationships.
- [ ] **Multicollinearity:** Identify features with correlation $> 0.85$. Plan to remove or combine redundant variables to prevent instability.
- [ ] **Target Relation:** Identify features with the highest absolute correlation to the target.
