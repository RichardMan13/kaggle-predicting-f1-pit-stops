import os
from pathlib import Path

# Caminhos do Projeto
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

# Configurações Globais de Reprodutibilidade
SEED = 25844181

# Configuração de Cross-Validation (Mudar conforme a competição)
N_SPLITS = 5
SHUFFLE = True
STRATIFIED = True  # True para classificação, False para regressão

# Configurações de Dados
TARGET_COL = "target"
ID_COL = "id"

# Parâmetros dos Modelos
MODEL_PARAMS = {
    "lightgbm": {
        "objective": "binary",  # ou "regression", "multiclass"
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "random_state": SEED,
        "verbose": -1,
        "n_jobs": -1
    },
    "xgboost": {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "random_state": SEED,
        "n_jobs": -1
    },
    "catboost": {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 1000,
        "learning_rate": 0.05,
        "random_seed": SEED,
        "verbose": 0,
        "thread_count": -1
    }
}
