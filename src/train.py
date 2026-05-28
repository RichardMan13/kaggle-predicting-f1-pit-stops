import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, mean_squared_error
from src import config, data_loader, features, models

def train_pipeline(model_name="lightgbm", mode="classifier"):
    print(f"Iniciando Pipeline de Treinamento: {model_name} ({mode})")
    
    # 1. Carregar dados de treino
    try:
        df_train = data_loader.load_train_data()
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        print("Crie um arquivo train.csv dummy em data/raw/ para testar o pipeline executando o script.")
        return
        
    # 2. Aplicar Engenharia de Features
    df_train = features.engineer_features(df_train, is_train=True)
    
    # 3. Separar Features e Target
    features_cols = [c for c in df_train.columns if c not in [config.TARGET_COL, config.ID_COL]]
    X = df_train[features_cols]
    y = df_train[config.TARGET_COL]
    
    # 4. Configurar Arrays OOF e Splits
    oof_predictions = np.zeros(len(df_train))
    cv_splits = data_loader.get_cv_splits(df_train)
    
    scores = []
    
    # 5. Loop de Treino por Fold
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n--- Treinando Fold {fold + 1}/{config.N_SPLITS} ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Obter e treinar modelo
        model = models.get_model(model_name, mode=mode)
        
        # Lógica especial de Early Stopping ou fit dependendo do framework
        if model_name in ["lightgbm", "xgboost"]:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[]  # Adicione callbacks de early stopping se desejado
            )
        else:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            
        # Predições de validação
        if mode == "classifier":
            val_preds = model.predict_proba(X_val)[:, 1]
            fold_score = roc_auc_score(y_val, val_preds)
            print(f"Fold {fold + 1} AUC: {fold_score:.5f}")
        else:
            val_preds = model.predict(X_val)
            fold_score = np.sqrt(mean_squared_error(y_val, val_preds))
            print(f"Fold {fold + 1} RMSE: {fold_score:.5f}")
            
        # Armazenar OOF e score
        oof_predictions[val_idx] = val_preds
        scores.append(fold_score)
        
        # Salvar o modelo treinado
        model_path = config.ARTIFACTS_DIR / f"{model_name}_fold_{fold}.pkl"
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")
        
    # 6. Avaliar Consistência OOF consolidada
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    print("\n========================================")
    if mode == "classifier":
        overall_score = roc_auc_score(y, oof_predictions)
        print(f"Consolidado OOF AUC: {overall_score:.5f}")
        print(f"CV AUC Medio: {mean_score:.5f} +/- {std_score:.5f}")
    else:
        overall_score = np.sqrt(mean_squared_error(y, oof_predictions))
        print(f"Consolidado OOF RMSE: {overall_score:.5f}")
        print(f"CV RMSE Medio: {mean_score:.5f} +/- {std_score:.5f}")
    print("========================================")
    
    # Salvar predições Out-of-Fold para blendings futuros
    oof_df = pd.DataFrame({
        config.ID_COL: df_train[config.ID_COL],
        "oof_pred": oof_predictions,
        config.TARGET_COL: y
    })
    oof_df.to_csv(config.ARTIFACTS_DIR / f"{model_name}_oof.csv", index=False)
    print(f"Saved OOF predictions to {config.ARTIFACTS_DIR / f'{model_name}_oof.csv'}")

if __name__ == "__main__":
    # Pode ser parametrizado via argparse
    train_pipeline(model_name="lightgbm", mode="classifier")
