---
name: ml-tree-models
description: Train, evaluate, and tune gradient boosted tree models (LightGBM, XGBoost, CatBoost). Use when working with tabular tree models, configuring early stopping, handling categorical features natively, or defining custom tree metrics.
---

# Gradient Boosted Tree Models

Use this skill to build highly performing and regularized tree-based models (LightGBM, XGBoost, CatBoost) for tabular datasets.

## Quick start

Train a LightGBM model safely with native categorical handling and early stopping:
```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert string columns to pandas category type for native LightGBM support
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
for c in cat_cols:
    X_train[c] = X_train[c].astype('category')
    X_val[c] = X_val[c].astype('category')

model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    random_state=42,
    verbose=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)
```

## Workflows

### 1. Configure Native Categorical Handling
- [ ] **LightGBM:** Convert object/string columns to `category` dtype in pandas. Do not use One-Hot encoding; LightGBM's native partition split is much faster and highly performing.
- [ ] **CatBoost:** Pass `cat_features` indexes or names directly to `CatBoostClassifier` or `CatBoostRegressor`.
- [ ] **XGBoost:** Enable categorical support via `enable_categorical=True` and use `category` pandas dtypes.

### 2. Regularization Checklists (Anti-Overfitting)
- [ ] **Early Stopping:** Always specify `eval_set` and an early stopping callback (e.g., 50 rounds) to prevent models from memorizing train fold noise.
- [ ] **Tree Depth Controls:** Restrict complexity by setting `max_depth` (e.g., between 4 and 8) and `num_leaves` (LightGBM standard is usually 31 or 63).
- [ ] **Regularization Params:** Tune `reg_alpha` (L1), `reg_lambda` (L2), and feature fraction (`colsample_bytree`) to add robust noise to splits.
