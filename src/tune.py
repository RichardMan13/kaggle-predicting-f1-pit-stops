import argparse
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

from src import config, data_loader, features


def objective(trial, model_name):
    if model_name == "xgboost":
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_estimators": trial.suggest_int("n_estimators", 1000, 5000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": config.SEED,
            "n_jobs": -1,
            "enable_categorical": True,
        }
        model = XGBClassifier(**params)

    elif model_name == "logistic":
        params = {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "solver": trial.suggest_categorical(
                "solver", ["lbfgs", "liblinear", "saga"]
            ),
            "max_iter": trial.suggest_int("max_iter", 500, 2000),
            "random_state": config.SEED,
            "n_jobs": -1,
        }
        if params["solver"] in ["liblinear", "saga"]:
            params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
            if params["solver"] == "liblinear":
                del params["n_jobs"]  # liblinear não suporta processamento paralelo
        else:
            params["penalty"] = "l2"

        model = LogisticRegression(**params)

    elif model_name == "mlp":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f"n_units_l{i}", 32, 256, log=True))

        params = {
            "hidden_layer_sizes": tuple(layers),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-1, log=True
            ),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "solver": "adam",
            "max_iter": trial.suggest_int("max_iter", 200, 500),
            "early_stopping": True,
            "random_state": config.SEED,
        }
        model = MLPClassifier(**params)

    elif model_name == "knn":
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical(
                "metric", ["euclidean", "manhattan", "minkowski"]
            ),
            "n_jobs": -1,
        }
        model = KNeighborsClassifier(**params)

    else:
        raise ValueError(f"Tuning para o modelo '{model_name}' não implementado.")

    # Carregar dados
    df_train = data_loader.load_train_data()
    # Aplicar engenharia de features com mitigacao de colinearidade
    df_train = features.engineer_features(df_train, is_train=True, prune_collinear=True)

    features_cols = [
        c for c in df_train.columns if c not in [config.TARGET_COL, config.ID_COL]
    ]
    X = df_train[features_cols]
    y = df_train[config.TARGET_COL]

    # Preencher NaNs, já que regressão logística, MLP e KNN não suportam valores ausentes
    if model_name in ["logistic", "mlp", "knn"]:
        # Seleciona apenas colunas numéricas para o preenchimento por mediana (evita erro com strings)
        num_cols = X.select_dtypes(include=[np.number]).columns
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())

        # Para modelos baseados em matemática linear/redes neurais, o ideal seria escalar e codificar categóricas,
        # assumindo que engineer_features faz ou fará isso. Garantimos remover strings brutas aqui.
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            X = X.drop(columns=cat_cols)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    scores = []

    # Validacao Cruzada robusta para evitar overfitting no tuning
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(
            f"    [Trial {trial.number}] Treinando {model_name} - Fold {fold + 1}/5...",
            end="\r",
        )
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        if model_name == "xgboost":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))

    return np.mean(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Otimização de Hiperparâmetros")
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "logistic", "mlp", "knn"],
        help="Nome do modelo para otimizar",
    )
    args = parser.parse_args()

    print(f"Iniciando busca de hiperparametros para '{args.model}' com Optuna...")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        study_name=f"tune_{args.model}",
        storage="sqlite:///tuning_history.db",
        load_if_exists=True,
        direction="maximize",
    )

    # Função auxiliar para limpar a linha antes do Optuna logar
    def trial_callback(study, trial):
        print(" " * 60, end="\r")  # Limpa o texto do Fold progress

    # Lambda wrapper para passar o model_name para a objective function
    study.optimize(
        lambda trial: objective(trial, args.model),
        n_trials=10,
        callbacks=[trial_callback],
    )

    print("\n========================================")
    print("Busca Completa!")
    print(f"Melhor Score AUC OOF: {study.best_value:.5f}")
    print("Melhores Hiperparametros encontrados:")
    for k, v in study.best_params.items():
        print(f"  '{k}': {v}")
    print("========================================")
