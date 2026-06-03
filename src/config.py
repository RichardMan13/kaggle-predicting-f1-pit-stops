import os
from pathlib import Path

from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Carrega o token do arquivo access_token se KAGGLE_API_TOKEN não estiver definido
if "KAGGLE_API_TOKEN" not in os.environ:
    ACCESS_TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
    if ACCESS_TOKEN_PATH.exists():
        try:
            token = ACCESS_TOKEN_PATH.read_text(encoding="utf-8").strip()
            if token:
                os.environ["KAGGLE_API_TOKEN"] = token
        except Exception:
            pass

# Valida se o token foi configurado com sucesso
if "KAGGLE_API_TOKEN" not in os.environ:
    token_path = Path.home() / ".kaggle" / "access_token"
    raise ValueError(
        "\n[ERRO] A chave de API do Kaggle ('KAGGLE_API_TOKEN') não está configurada!\n"
        "Por favor, configure o seu token de acesso salvando-o no arquivo:\n"
        f"  {token_path}\n"
        "ou definindo KAGGLE_API_TOKEN no arquivo '.env' do seu projeto."
    )

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
TARGET_COL = "PitNextLap"
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
        "n_jobs": -1,
    },
    "xgboost": {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 3495,
        "learning_rate": 0.013221741833916828,
        "max_depth": 8,
        "subsample": 0.9049310134679247,
        "colsample_bytree": 0.4253497805796257,
        "reg_alpha": 0.5551388408352103,
        "reg_lambda": 0.047217876626177865,
        "random_state": SEED,
        "n_jobs": -1,
        "enable_categorical": True,
    },
    "catboost": {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 1000,
        "learning_rate": 0.05,
        "random_seed": SEED,
        "verbose": 0,
        "thread_count": -1,
    },
    "logistic": {
        "C": 0.030865858700458795,
        "solver": "liblinear",
        "max_iter": 1350,
        "penalty": "l1",
        "random_state": SEED,
    },
    "mlp": {
        "hidden_layer_sizes": (102, 98),
        "alpha": 0.0003982026072876022,
        "learning_rate_init": 0.001156208547201977,
        "activation": "relu",
        "solver": "adam",
        "max_iter": 482,
        "early_stopping": True,
        "random_state": SEED,
    },
    "knn": {
        "n_neighbors": 49,
        "weights": "uniform",
        "metric": "euclidean",
        "algorithm": "auto",
        "leaf_size": 50,
        "n_jobs": -1,
    },
}
