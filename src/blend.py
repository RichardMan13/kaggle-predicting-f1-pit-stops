import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

from src import config


def blend_pipeline():
    print("Iniciando Pipeline de Blending Dinamico (Robust Rank Ensemble)...")

    # 1. Detectar OOFs disponiveis na pasta de artefatos
    # Restringido para rodar exclusivamente xgboost e knn (ignorando o restante dos OOFs salvos na pasta)
    supported_models = ["xgboost", "knn"]
    oof_dfs = {}

    for model in supported_models:
        path = config.ARTIFACTS_DIR / f"{model}_oof.csv"
        if path.exists():
            oof_dfs[model] = pd.read_csv(path)

    if len(oof_dfs) < 2:
        print(
            "\n[ERRO] Sao necessarios pelo menos 2 modelos treinados com arquivos OOF para realizar o blending!"
        )
        print(f"Modelos suportados: {supported_models}")
        print(f"Modelos encontrados atualmente: {list(oof_dfs.keys())}")
        return

    models_list = list(oof_dfs.keys())
    print(f"Modelos detectados para blending: {models_list}")

    y_true = oof_dfs[models_list[0]][config.TARGET_COL].values

    # Solução B: Conversao das probabilidades brutas para Ranks Percentuais
    # Isso neutraliza problemas de escalas e calibrações discrepantes entre famílias.
    ranks = []
    for model in models_list:
        pred = oof_dfs[model]["oof_pred"].values
        rank = pd.Series(pred).rank(pct=True).values
        ranks.append(rank)
        print(f"  * {model} OOF AUC: {roc_auc_score(y_true, pred):.5f}")

    # 2. Otimizar os pesos usando SLSQP baseando-se nos Ranks com M-1 variáveis de liberdade
    # O último peso é determinado deterministicamente para garantir soma 1.0.
    M = len(models_list)

    def objective(x):
        w = list(x)
        w_last = 1.0 - sum(w)

        # A única regra inquebrável: o último peso não pode ser negativo.
        # Se a soma dos pesos anteriores for > 1.0, w_last fica negativo.
        if w_last < 0.0:
            # Retorna um valor ruim (positivo) proporcional ao erro,
            # guiando o otimizador suavemente de volta para a área válida.
            return 1.0 + abs(w_last)

        w.append(w_last)

        blend_preds = np.zeros_like(ranks[0])
        for i in range(M):
            blend_preds += w[i] * ranks[i]

        return -roc_auc_score(y_true, blend_preds)  # Minimizar negativo = Maximizar AUC

    # Chute inicial uniforme
    init_weights = [1.0 / M] * (M - 1)
    bounds = [(0.0, 1.0)] * (M - 1)

    # Mudança Chave: Uso de algoritmo livre de derivadas (Gradient-Free) Powell
    # Essencial para otimizar superficies não-suaves e degraus como a métrica do AUC.
    res = minimize(objective, init_weights, method="Powell", bounds=bounds)

    w_opt = list(res.x)
    w_last_opt = 1.0 - sum(w_opt)
    w_opt.append(w_last_opt)

    # Normalizar para garantir soma exata a 1.0 por seguranca numerica
    w_opt = np.array(w_opt) / np.sum(w_opt)
    best_auc = -res.fun

    print("\n========================================")
    print("Otimizacao de Blending Robust Rank Ensemble Completa!")
    print(f"Pesos Ideais:")
    for i, model in enumerate(models_list):
        print(f"  * {model}: {w_opt[i]:.4f}")
    print(f"Consolidado OOF AUC Blended (Ranks): {best_auc:.5f}")
    print("========================================")

    # 3. Carregar submissoes de teste e combinar usando Ranks Percentuais
    sub_dfs = {}
    for model in models_list:
        path = config.SUBMISSIONS_DIR / f"submission_{model}.csv"
        if path.exists():
            sub_dfs[model] = pd.read_csv(path)

    if len(sub_dfs) < M:
        print("\n[AVISO] Nem todas as submissoes de teste foram encontradas.")
        print(f"Esperadas submissoes para: {models_list}")
        print("Inferencia de teste incompleta. Previsoes OOF otimizadas com sucesso!")
        return

    # Unificar predicoes via Rank Blending
    sub_blend = sub_dfs[models_list[0]].copy()
    blend_preds_test = np.zeros(len(sub_blend))

    for i, model in enumerate(models_list):
        pred_test = sub_dfs[model][config.TARGET_COL].values
        rank_test = pd.Series(pred_test).rank(pct=True).values
        blend_preds_test += w_opt[i] * rank_test

    sub_blend[config.TARGET_COL] = blend_preds_test

    blend_path = config.SUBMISSIONS_DIR / "submission_blend.csv"
    sub_blend.to_csv(blend_path, index=False)
    print(f"Submissao robusta rank-blended criada com sucesso em: {blend_path}!")


if __name__ == "__main__":
    blend_pipeline()
