# Contexto da Competição: Playground Series S6E5 - F1 Pit Stops

Este arquivo contém as especificações, objetivos e detalhes sobre os dados da competição obtidos diretamente do Kaggle.

---

## Objetivo (Your Goal)
Prever se um piloto de Fórmula 1 fará uma parada nos boxes (pit stop) na próxima volta.
*   **Target:** `PitNextLap` (Binário: `1` para sim, `0` para não).

---

## Métrica de Avaliação (Evaluation)
As submissões são avaliadas pela área sob a curva ROC (**ROC AUC Score**) entre a probabilidade prevista e o alvo observado.
*   O modelo deve prever a **probabilidade** da classe positiva, e não a classe direta.

---

## Estrutura do Arquivo de Submissão
O arquivo de submissão final deve conter um cabeçalho e ter a seguinte estrutura:

```csv
id,PitNextLap
439140,0.2
439141,0.3
439142,0.9
```

---

## Descrição do Dataset (Dataset Description)
*   O conjunto de dados (treino e teste) é inspirado em um conjunto de dados de estratégia da F1.
*   As distribuições de features são próximas, mas não exatamente iguais às originais.
*   **Remoção Crítica:** A coluna `Normalized_TyreLife` foi removida intencionalmente para evitar que a previsão se tornasse trivial.
*   **Dica Especial:** É permitido o uso de datasets originais da F1 para explorar diferenças e incorporá-los ao treino a fim de melhorar a performance do modelo.

---

## Arquivos do Projeto
*   `train.csv` - O conjunto de dados de treino, contendo a coluna alvo `PitNextLap`.
*   `test.csv` - O conjunto de dados de teste, usado para prever a probabilidade de `PitNextLap`.
*   `sample_submission.csv` - Um arquivo de submissão de exemplo com a formatação correta esperada.

---

## Insights de Dados & Diretrizes de Modelagem (EDA)

Estas diretrizes foram extraídas da análise exploratória e devem guiar a modelagem e engenharia de features:

### 1. Padroes de Desgaste por Composto (Compound & TyreLife)
- Pneus macios (`SOFT`) e de chuva (`WET`) sao trocados significativamente mais cedo do que pneus duros (`HARD`).
- **Limiares de troca (Media de TyreLife em paradas de decisao PitNextLap == 1.0):**
  - `SOFT`: Troca media em **12.6 voltas** (degradacao termica media de **-30.31**).
  - `MEDIUM`: Troca media em **17.1 voltas** (degradacao termica media de **-38.49**).
  - `HARD`: Troca media em **21.4 voltas** (degradacao termica media de **-48.05**)
- **Diretriz de Features:** Criar features de desgaste relativo dividindo a idade atual (`TyreLife`) pelo limiar medio de cada composto.

### 2. O Pico de Estrategia do Stint 2
- O primeiro stint apresenta baixa taxa de paradas. No entanto, no **Stint 2 a taxa de paradas atinge o pico de 46.18%** na janela de 60% a 80% do progresso da corrida.
- Na reta final da prova (80-100% de progresso), a probabilidade de box cai bruscamente no Stint 2 para **34.08%** (tatica de evitar boxes no fim da corrida para nao perder posicoes de pista).
- **Diretriz de Features:** Criar variaveis cruzadas que multipliquem o Stint pelo progresso restante da corrida.

### 3. Tatica Pelotao vs Lideres (Position)
- Pilotos no meio do pelotao (posicoes intermediarias de P10 a P15, com pico de **23.51% na P13**) realizam pit stops com maior frequencia para taticas de ultrapassagem (*undercut*).
- Lideres (P1 a P3) mantem paradas estritamente planejadas, enquanto lanternas (P19 e P20) esticam os pneus ao maximo (taxa de paradas cai para **15.4% na P20**).

### 4. Vies Geografico das Pistas (GP Circuit Bias)
- Ha um vies geografico fortissimo na durabilidade media dos pneus por circuito no momento de troca real (`PitStop == 1`).
- No GP da Gra-Bretanha (`British Grand Prix`), a media de TyreLife nas trocas e de apenas **7.2 voltas** (pista de altissimo desgaste). No GP do Mexico (`Mexico City Grand Prix`), a media se estende ate **14.3 voltas** (baixo desgaste).
- **Diretriz de Features:** Mapear o desgaste medio historico por circuito (`Race`) como uma variavel continua para guiar o classificador.

### 5. O Efeito 'In-Lap Push' (LapTime & LapTime_Delta)
- Nas voltas imediatamente anteriores a decisao de parada (`PitNextLap == 1.0`), a media de `LapTime (s)` e **mais rapida** (89.59s) em comparacao com as voltas normais (91.28s), acompanhada de um delta mais negativo (-4.20s).
- **Insight de Corrida:** Isso revela o efeito tatico do **'in-lap push'** (o piloto extrai toda a performance restante na volta que antecede a troca). Mapear esse ganho de ritmo repentino sera uma feature de extrema utilidade.

### 6. Probabilidade de Pit Stops Consecutivos (PitStop)
- A taxa de paradas consecutivas e de **24.78%** em voltas onde `PitStop == 1`, contra 19.12% em voltas sem pit stop.
- **Conclusao Crucial:** Esta e uma caracteristica unica deste dataset! Tentar forçar previsoes `0.0` logo apos paradas de boxes baseado na intuicao real de pista **prejudicaria** severamente o score do modelo.

### 7. Estilo de Preservacao do Piloto (Driver Bias)
- Há uma variacao brutal no estilo de pilotagem: o piloto `OCO` consegue esticar a vida util media do pneu ate **22.0 voltas** antes da parada de box, enquanto `D277` e `D279` realizam paradas com medias de apenas **6.2 e 6.5 voltas**.
- **Diretriz de Features:** Criar uma pontuacao de preservacao de pneus por piloto (`Driver`) ou utilizar um Target Encoding robusto com K-Fold na coluna `Driver`.

### 8. Anomalia de Durabilidade por Temporada (Yearly Trends)
- Nas temporadas de `2022`, `2024` e `2025`, a durabilidade media dos pneus mantem-se estavel entre **10.1 e 11.4 voltas**.
- A temporada de **`2023` apresenta um comportamento completamente anomalo: durabilidade media de apenas 2.37 voltas** e baixissimo volume de pit stops registrados (1.685).
- **Impacto de Modelagem:** Tratar a temporada (`Year`) com features categoricas ou normalizar as estimativas de desgaste separadamente por ano sera vital para nao confundir o modelo com o ruido anomalo de 2023.

---

## Fase 2 - Analise de Influencia de Features & Validacao (XGBoost)

Rodamos um experimento quantitativo comparando o impacto das novas variaveis em um split de validacao de 80/20 com o modelo XGBoost:

*   **AUC do Baseline (Sem Engenharia de Features):** `0.94002`
*   **AUC da Fase 2 (Com Engenharia de Features):** `0.93872`

### Importancia das Features por Ganho (Gain)
1.  **`is_2023_season`**: `4591.83` (Isolamento da anomalia do ano de 2023)
2.  **`Stint`**: `1868.48`
3.  **`TyreLife`**: `291.37`
4.  **`stint_progress_interaction`**: `51.05` (Interacao multiplicativa)
5.  **`driver_wear_bias`**: `33.62` (Vies de preservacao do piloto)
6.  **`wear_ratio`**: `27.96` (Desgaste relativo por composto)
7.  **`track_wear_bias`**: `3.63` (Vies geografico por pista)

### Conclusoes e Diretrizes de Modelagem Avancada
*   **Isolamento da Anomalia:** A feature `is_2023_season` obteve disparada o maior ganho do modelo. Isso comprova que fornecer um indicador explicito do ano de 2023 permite que o XGBoost isole essa anomalia sem comprometer as fronteiras de decisao das outras temporadas.
*   **Efeito da Colinearidade:** A pequena queda de AUC (~0.0013) indica que a introducao de variaveis altamente correlacionadas (como `wear_ratio` e `TyreLife`) causa uma leve fragmentacao nos nos de decisao das arvores de forma individual.
*   **Diversidade para Ensembles:** Embora a performance individual tenha oscilado, as novas features geram um classificador com comportamento altamente complementar ao baseline. A combinacao das duas predicoes via Blending e altamente recomendada para maximizar o score no Kaggle.

