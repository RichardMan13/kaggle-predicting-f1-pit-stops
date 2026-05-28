import os
import shutil
import zipfile
from pathlib import Path
from invoke import task

BASE_DIR = Path(__file__).resolve().parent

@task
def clean(c):
    """Limpa arquivos temporários do Python, caches de Jupyter, logs e builds desnecessários."""
    print("Iniciando limpeza do workspace...")
    
    # 1. Limpar caches do Python (__pycache__, *.pyc, etc.)
    pycache_count = 0
    for p in BASE_DIR.rglob("__pycache__"):
        shutil.rmtree(p)
        pycache_count += 1
    
    pyc_count = 0
    for p in BASE_DIR.rglob("*.py[co]"):
        p.unlink()
        pyc_count += 1
        
    # 2. Limpar checkpoints de Jupyter Notebooks
    jupyter_count = 0
    for p in BASE_DIR.rglob(".ipynb_checkpoints"):
        shutil.rmtree(p)
        jupyter_count += 1
        
    print(f"Removidos {pycache_count} diretórios __pycache__ e {pyc_count} arquivos .pyc/.pyo.")
    print(f"Removidos {jupyter_count} diretórios de checkpoints do Jupyter.")

@task
def format(c):
    """Formata o código-fonte nas pastas src/ e notebooks/ utilizando o Ruff."""
    print("Formatando código com Ruff...")
    c.run("ruff format src/ notebooks/ tasks.py", warn=True)

@task
def lint(c):
    """Executa a verificação estática de código com o Ruff."""
    print("Executando análise estática com Ruff...")
    c.run("ruff check src/ notebooks/ tasks.py", warn=True)

@task(pre=[format, lint])
def check(c):
    """Executa a formatação e a verificação estática consecutivamente com o Ruff."""
    print("Verificação completa com Ruff concluída com sucesso!")

@task
def download_data(c, competition):
    """
    Baixa os dados da competição via API do Kaggle e descompacta na pasta data/raw/.
    Exemplo: inv download-data --competition=titanic
    """
    raw_dir = BASE_DIR / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Iniciando download dos dados da competicao '{competition}' via Kaggle API...")
    c.run(f"kaggle competitions download -c {competition} -p {raw_dir}")
    
    zip_file = raw_dir / f"{competition}.zip"
    if zip_file.exists():
        print(f"Descompactando {zip_file.name} em {raw_dir}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        zip_file.unlink()
        print("Dados baixados e descompactados com sucesso!")
    else:
        # Se a API baixou os arquivos CSV ou outros tipos diretamente sem zipar
        print("Download concluido! Nenhum arquivo ZIP principal precisou ser extraido.")

@task(pre=[clean])
def train(c, model="lightgbm", mode="classifier"):
    """
    Roda o pipeline principal de treino cruzado (CV).
    Exemplo de uso: inv train --model=xgboost --mode=classifier
    """
    print(f"Disparando pipeline de treino para o modelo '{model}' no modo '{mode}'...")
    # Executa o módulo train com python
    c.run(f"python src/train.py", pty=True)

@task
def predict(c, model="lightgbm", mode="classifier"):
    """
    Roda o pipeline de inferência final com os modelos treinados.
    Exemplo de uso: inv predict --model=lightgbm
    """
    print(f"Disparando pipeline de inferência...")
    c.run(f"python src/predict.py", pty=True)
