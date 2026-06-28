import pandas as pd

# Dicionarios de limites medios obtidos no EDA
COMPOUND_LIMITS = {
    "SOFT": 12.6,
    "MEDIUM": 17.1,
    "HARD": 21.4,
    "WET": 13.0,
    "INTERMEDIATE": 18.0,
}


def engineer_features(df, is_train=True):
    """
    Aplica transformacoes puras e engenharia de features no DataFrame.
    Garante o mesmo processamento para dados de treino e teste.
    """
    df = df.copy()

    # 1. Desgaste Relativo por Composto (wear_ratio)
    df["wear_ratio"] = df["TyreLife"] / df["Compound"].map(COMPOUND_LIMITS).fillna(15.0)

    # 2. Interacao de Fim de Corrida (stint_progress_interaction)
    df["stint_progress_interaction"] = df["Stint"] * (1.0 - df["RaceProgress"])

    # 3. Ajuste Anomalo de Temporada (is_2023_season)
    df["is_2023_season"] = (df["Year"] == 2023).astype(int)

    # 4. Lag Features Temporais (Media Movel de 3 voltas)
    # Ordenamos o dataframe localmente por Stint e LapNumber para garantir corretude temporal.
    # GroupBy por Race e Driver (garantindo que nao vaze tempo de um piloto para outro).
    df = df.sort_values(by=["Race", "Driver", "Stint", "LapNumber"])

    # Adicionando lags de 3 periodos fechados no Stint
    df["LapTime_Delta_roll_3"] = df.groupby(["Race", "Driver", "Stint"])[
        "LapTime_Delta"
    ].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["Cum_Degradation_roll_3"] = df.groupby(["Race", "Driver", "Stint"])[
        "Cumulative_Degradation"
    ].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # Retornar o DataFrame a ordem original pelo indice para nao quebrar a alinhacao com y
    df = df.sort_index()

    # Converter colunas de texto/objeto para tipo 'category' do pandas
    # permitindo que LightGBM e XGBoost as tratem nativamente
    cat_cols = ["Driver", "Compound", "Race"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df
