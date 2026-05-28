import pandas as pd
import numpy as np

def engineer_features(df, is_train=True):
    """
    Aplica transformações puras e engenharia de features no DataFrame.
    Garante o mesmo processamento para dados de treino e teste.
    """
    df = df.copy()
    
    # -------------------------------------------------------------
    # TODO: Implemente suas features customizadas aqui
    # Exemplo:
    # df['feature_comb'] = df['feat_1'] * df['feat_2']
    # df['log_feature'] = np.log1p(df['numeric_feat'])
    # -------------------------------------------------------------
    
    return df
