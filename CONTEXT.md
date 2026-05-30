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
