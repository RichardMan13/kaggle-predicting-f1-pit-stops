# Kaggle Express Template

Uma arquitetura de pipeline leve, modular, determinística e focada em validação cruzada robusta para competições do Kaggle.

## Estrutura de Pastas

```text
├── data/
│   ├── external/           # Dados de fontes externas/APIs públicas
│   ├── interim/            # Dados intermediários transformados
│   ├── processed/          # Dados finais processados prontos para modelagem
│   └── raw/                # Arquivos originais da competição (train.csv, test.csv)
├── artifacts/              # Modelos treinados (.pkl), previsões Out-of-Fold (OOF) e logs
├── notebooks/              # Exclusivo para EDA rápida e testes de hipóteses
│   └── 01_exploratory_analysis.ipynb
├── src/                    # O coração estruturado do seu pipeline
│   ├── config.py           # Paths, SEED, Hiperparâmetros e variáveis globais
│   ├── data_loader.py      # Leitura dos dados e configuração do Cross-Validation (K-Fold)
│   ├── features.py         # Funções puras de Engenharia de Features
│   ├── models.py           # Definição e arquitetura dos modelos
│   ├── train.py            # Script principal: roda o CV, calcula métricas, salva artefatos e gera OOF
│   └── predict.py          # Script de inferência: lê os modelos treinados e gera a submissão final
├── submissions/            # Armazena os arquivos de submissão finais (.csv) prontos para envio
```

## Como Começar um Novo Desafio

1. **Configurar o Ambiente com Conda**:
   Crie o ambiente virtual utilizando o arquivo de configuração `environment.yml` que já contém todas as dependências pré-configuradas em **Python 3.10** e instale a pasta `src` local em modo editável:
   ```bash
   conda env create -f environment.yml
   conda activate cookiecutter-kaggle
   pip install -e .
   ```

2. **Configurar as credenciais da Kaggle API**:
   * Acesse sua conta no site do Kaggle e vá em **Settings** > **API** > **Create New Token** para baixar o arquivo `kaggle.json`.
   * Salve este arquivo na pasta correta do seu computador:
     * No Windows: `C:\Users\SeuUsuario\.kaggle\kaggle.json`
     * No Linux/macOS: `~/.kaggle/kaggle.json`
   * Aceite os termos de regras no site da própria competição do Kaggle antes de prosseguir.

3. **Baixar os dados da competição de forma automática**:
   ```bash
   inv download-data --competition=nome-da-competicao
   ```

4. **Configurar suas variáveis e caminhos** em `src/config.py`.

5. **Executar o pipeline de treino** (com limpeza automática de caches):
   ```bash
   inv train
   ```

6. **Gerar previsões de submissão** usando os modelos do CV:
   ```bash
   inv predict
   ```

---

## Automação com Invoke (`tasks.py`)

Em vez de usar `Makefile`, este template utiliza a biblioteca **Invoke** escrita em Python puro, funcionando em Windows, macOS e Linux.

* **Listar tarefas disponíveis**:
  ```bash
  inv --list
  ```
* **Limpar cache e arquivos temporários**:
  ```bash
  inv clean
  ```
* **Formatar código com Ruff**:
  ```bash
  inv format
  ```
* **Executar linter (Ruff check)**:
  ```bash
  inv lint
  ```
* **Executar todas as verificações consecutivamente (Formatação + Lint)**:
  ```bash
  inv check
  ```