import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src import config


def load_train_data():
    """Carrega o conjunto de dados de treino."""
    train_path = config.RAW_DATA_DIR / "train.csv"
    if not train_path.exists():
        msg = (
            f"Arquivo de treino não encontrado em {train_path}. "
            "Por favor, baixe os dados usando a API do Kaggle."
        )
        raise FileNotFoundError(msg)
    return pd.read_csv(train_path)


def load_test_data():
    """Carrega o conjunto de dados de teste."""
    test_path = config.RAW_DATA_DIR / "test.csv"
    if not test_path.exists():
        msg = (
            f"Arquivo de teste não encontrado em {test_path}. "
            "Por favor, baixe os dados usando a API do Kaggle."
        )
        raise FileNotFoundError(msg)
    return pd.read_csv(test_path)


def get_cv_splits(df):
    """Gera as dobras de Cross-Validation com base na configuração.

    Retorna:
        list: Lista de tuplas (train_idx, val_idx)
    """
    kf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.SEED,
    )
    return list(kf.split(df, df[config.TARGET_COL]))
