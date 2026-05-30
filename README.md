# Predict F1 Pit Stops (Kaggle Playground S6E5)

A lightweight, high-performance, and modular pipeline architecture focused on robust cross-validation to predict whether a Formula 1 driver will pit on the next lap.

---

## Goal and Metrics

*   **Goal:** Predict the probability that a driver will make a pit stop on the next lap (target `PitNextLap`).
*   **Evaluation Metric:** Area under the ROC curve (ROC AUC Score).
*   **Domain Context:** The dataset is inspired by a real F1 race strategy dataset. The feature `Normalized_TyreLife` was intentionally removed to prevent trivial predictions, requiring robust feature engineering to estimate tire degradation.

---

## Project Structure

```text
├── .agents/
│   └── skills/             # Custom AI Agent Skills (EDA, CV, Feature Ideation, etc.)
├── data/
│   ├── external/           # External F1 data (e.g., Ergast API / FastF1)
│   ├── interim/            # Transformed intermediate data
│   ├── processed/          # Final processed data ready for modeling
│   └── raw/                # Original competition files (train.csv, test.csv, sample_submission.csv)
├── artifacts/              # Trained models (.pkl), Out-of-Fold (OOF) predictions, and logs
├── notebooks/              # Sandbox for quick exploratory data analysis (EDA) and prototypes
├── src/                    # The structured heart of the pipeline
│   ├── config.py           # Paths, reproducibility seeds, model parameters, and API keys
│   ├── data_loader.py      # Data loading and robust cross-validation fold splitting
│   ├── features.py         # Domain-specific F1 feature engineering (tire wear, safety cars, etc.)
│   ├── models.py           # Machine learning model architectures (LightGBM, XGBoost, CatBoost, PyTorch)
│   ├── train.py            # Cross-validation execution, metric logging, and OOF generation
│   └── predict.py          # Final inference on test.csv and submission validation
├── submissions/            # Final prediction CSV files ready for Kaggle submission
└── tasks.py                # Command orchestration via Invoke
```

---

## How to Get Started

### 1. Configure the Environment with Conda
Create the virtual environment containing all machine learning dependencies and GPU support (CUDA 12.1):
```bash
# Create the virtual environment
conda env create -f environment.yml

# Activate the environment
conda activate cookiecutter-kaggle

# Install the src package in editable mode
pip install -e .
```

### 2. Kaggle API Authentication (Recommended Method)
This repository is configured to natively read the new Kaggle API Token format:
1. Log in to your Kaggle account -> Settings -> click "Create New Token" under the "API Tokens (Recommended)" section.
2. On Windows, save the downloaded token (`KGAT_...`) exactly into the file:
   `C:\Users\ricar\.kaggle\access_token`
3. Alternatively, create a `.env` file from the `.env.template` in the project root and add:
   ```env
   KAGGLE_API_TOKEN=your_token_here
   ```

### 3. Download the Competition Data Automatically
```bash
inv download-data --competition=playground-series-s6e5
```
*This command automatically downloads and extracts the competition files directly into `data/raw/`.*

---

## AI Agent Skills

This repository includes a `.agents/skills/` directory containing structured workflows to guide developers and AI coding agents.

Available skills:
*   `/kaggle-eda-loop` — Systematic, bug-free exploratory data analysis checklist.
*   `/kaggle-cv-guardrails` — Strict prevention of data leakage during pre-processing and training.
*   `/kaggle-feature-ideation` — F1 domain-specific feature engineering formulas (tire wear, safety car, pit lane deltas).
*   `/kaggle-model-tuner` — Stable, robust hyperparameter tuning using Optuna.
*   `/kaggle-submission-sanity` — Final checklist to validate prediction files before uploading.

---

## Automation and Useful Commands (tasks.py)

We use the Invoke library to orchestrate pipeline tasks across platforms.

*   **List tasks:** `inv --list`
*   **Clean temporary files:** `inv clean`
*   **Format code:** `inv format`
*   **Run linter (Ruff):** `inv lint`
*   **Run training pipeline:** `inv train` (runs cross-validation training for all configured models)
*   **Run inference pipeline:** `inv predict` (generates the final submission file in `submissions/`)