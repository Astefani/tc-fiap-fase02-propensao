.PHONY: help install lint test data repro metrics push pull mlflow-ui clean

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

data:               ## Baixa o dataset cru do UCI (alternativa ao `dvc pull`, sem credencial)
	@mkdir -p data/raw
	curl -sL -o /tmp/shoppers.zip "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
	unzip -o -j /tmp/shoppers.zip online_shoppers_intention.csv -d data/raw/
	@rm -f /tmp/shoppers.zip
	@echo "--- conferindo contra o ponteiro versionado ---"
	poetry run dvc status data/raw/online_shoppers_intention.csv.dvc

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
