from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from src import config

def get_model(model_name: str, mode: str = "classifier"):
    """
    Retorna a instância do modelo desejado baseado no nome e configuração.
    
    Args:
        model_name: Nome do algoritmo ('lightgbm', 'xgboost', 'catboost')
        mode: Tipo de tarefa ('classifier' ou 'regressor')
    """
    params = config.MODEL_PARAMS.get(model_name, {})
    
    if mode == "classifier":
        if model_name == "lightgbm":
            return LGBMClassifier(**params)
        elif model_name == "xgboost":
            return XGBClassifier(**params)
        elif model_name == "catboost":
            return CatBoostClassifier(**params)
        else:
            raise ValueError(f"Modelo {model_name} desconhecido para classificação.")
            
    elif mode == "regressor":
        if model_name == "lightgbm":
            return LGBMRegressor(**params)
        elif model_name == "xgboost":
            return XGBRegressor(**params)
        elif model_name == "catboost":
            return CatBoostRegressor(**params)
        else:
            raise ValueError(f"Modelo {model_name} desconhecido para regressão.")
    else:
        raise ValueError(f"Modo {mode} desconhecido. Escolha 'classifier' ou 'regressor'.")
