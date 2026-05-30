---
name: kaggle-cv-guardrails
description: Structure robust cross-validation pipelines and prevent data leakage. Use when designing data splitting strategies, preparing validation protocols, or evaluating model performance on tabular data to ensure CV results match public/private leaderboard scores.
---

# Cross-Validation (CV) Guardrails

Use this skill to design reliable validation strategies and block all forms of data leakage (leakage of information from the validation/test folds into the training fold).

## Quick start

Execute a clean, leak-free Stratified K-Fold validation using an explicit pipeline pattern:
```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # Preprocessing fit is ISOLATED to training fold only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) # transform only!
    
    model = LGBMClassifier()
    model.fit(X_train_scaled, y_train)
    scores.append(model.score(X_val_scaled, y_val))
```

## Workflows

### 1. Identify Split Strategy based on Domain
- [ ] **Temporal splits:** If the dataset has a time component (e.g., chronological Formula 1 races across years), use a time-based split (e.g., `TimeSeriesSplit` or split by Year) rather than random K-Fold. Random splitting into the past and future will cause high leakage.
- [ ] **Group-based splits:** If multiple rows belong to the same entity (e.g., multiple laps/stops by the same Driver or Car in a single race), use `GroupKFold` or `StratifiedGroupKFold` grouped by Driver/Race to prevent the model from memorizing specific driver/team constants.
- [ ] **Standard Stratified splits:** If the task is classification with imbalanced target labels, always use `StratifiedKFold`.

### 2. Preprocessing & Leakage Checklists
- [ ] **Scaling & Imputation:** Never run `.fit_transform()` on the whole dataset. Perform `.fit_transform()` *only* on the training fold, and use `.transform()` on the validation fold.
- [ ] **Feature Engineering Leakage:** Avoid calculating global statistics (e.g., calculating the average pit-stop duration across the entire dataset) and joining it back. Calculate these aggregations *only* within the training fold.
- [ ] **Target Encoding:** If target encoding is used, it **must** be calculated out-of-fold (e.g., using `CategoryEncoder`'s `TargetEncoder` wrapped in cross-validation) to prevent target leakage.
