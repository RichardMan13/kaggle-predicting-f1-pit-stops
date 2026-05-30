---
name: ml-scikit-pipelines
description: Build modular and leak-free machine learning pipelines using Scikit-Learn. Use when building preprocessing pipelines, using ColumnTransformer, applying categorical encoders, or scaling features.
---

# Scikit-Learn Machine Learning Pipelines

Use this skill to build reproducible, robust, and modular preprocessing and modeling pipelines using Scikit-Learn.

## Quick start

Build a clean preprocessing pipeline with `ColumnTransformer` and fit a model safely:
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

num_features = ["age", "duration"]
cat_features = ["tire_type", "team"]

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Training is completely self-contained and free of data leakage!
pipeline.fit(X_train, y_train)
```

## Workflows

### 1. Robust Imputation Strategies
- [ ] **Numerical:** Prefer median/mean imputation via `SimpleImputer` or advanced models like `KNNImputer`.
- [ ] **Categorical:** Use `SimpleImputer(strategy="most_frequent")` or create a distinct category for missing values `SimpleImputer(strategy="constant", fill_value="missing")`.

### 2. Feature Encoding & Transformation
- [ ] **Low Cardinality Categoricals:** Use `OneHotEncoder(handle_unknown="ignore")`.
- [ ] **High Cardinality Categoricals:** Prefer `TargetEncoder` or Ordinal/Label encoding depending on model constraints.
- [ ] **Scale Numerical Columns:** Always apply standardizing scales (`StandardScaler`, `MinMaxScaler` or `RobustScaler` if outliers are present) when using distance-based or linear algorithms.
