---
name: kaggle-model-tuner
description: Optimize algorithm hyperparameters using search tools like Optuna. Use when setting up Optuna trials, defining hyperparameter search spaces, or tuning model params to ensure stable optimization.
---

# Kaggle Model Tuner

Use this skill to optimize machine learning models (LightGBM, XGBoost, CatBoost) cleanly, avoiding validation overfitting and resource exhaustion.

## Quick start

Define and execute a robust Optuna tuning trial integrated with Stratified K-Fold:
```python
import optuna
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "random_state": 42,
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))
        
    return sum(scores) / len(scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print("Best Params:", study.best_params)
```

## Workflows

### 1. Define Realistic Search Spaces
- [ ] **Tree models (LightGBM/XGBoost/CatBoost):** 
  - `learning_rate`: Logarithmic uniform `[0.01, 0.15]`.
  - `max_depth` or `num_leaves`: Do not allow excessively deep trees (`max_depth > 10` or `num_leaves > 128` usually overfits rapidly).
  - `subsample` and `colsample_bytree`: Real uniform `[0.6, 1.0]` to encourage regularization.

### 2. Tuning Guardrails (Anti-Overfitting)
- [ ] **Out-of-Fold (OOF) Target:** Never optimize parameters on a single validation set. Tuning must always maximize the average cross-validation score across all folds.
- [ ] **Pruning:** Integrate Optuna's `PyTorchLightningPruningCallback` or built-in early stopping tracking for long training cycles to terminate bad trials early.
- [ ] **Track parameters:** Always save the final tuned dictionary to a config file (`src/config.py` or a dedicated JSON) rather than letting parameters wander between experiments.
