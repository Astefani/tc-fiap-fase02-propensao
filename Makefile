.PHONY: help install lint test repro metrics push pull mlflow-ui clean

# Sem alvo explícito, `make` mostra a ajuda em vez de executar o primeiro alvo.
.DEFAULT_GOAL := help

help:               ## Mostra esta ajuda
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:            ## Instala o ambiente (produção + dev)
	poetry install

lint:               ## Ruff sobre o código
	poetry run ruff check src tests

test:               ## Suíte de testes
	poetry run pytest

repro:              ## Executa o pipeline DVC (pula estágios não afetados)
	poetry run dvc repro

metrics:            ## Métricas do último run e diferença para o commit anterior
	poetry run dvc metrics show
	poetry run dvc metrics diff

push:               ## Envia dados e artefatos para o remote do DVC
	poetry run dvc push

pull:               ## Traz dados e artefatos do remote do DVC
	poetry run dvc pull

mlflow-ui:          ## Abre a UI do MLflow em http://localhost:5000
	poetry run mlflow ui --backend-store-uri sqlite:///mlflow-store/mlflow.db

clean:              ## Remove artefatos gerados (o DVC reconstrói com `make repro`)
	rm -rf models/*.joblib data/processed/*.parquet metrics/*.json
