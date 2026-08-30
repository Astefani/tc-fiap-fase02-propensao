"""Métricas de avaliação e estágio ``evaluate`` do pipeline DVC.

A métrica principal é **PR-AUC** (``average_precision``): com 15,5% de sessões
convertendo, acurácia é enganosa — prever "ninguém compra" acerta 84,5%.
"""

import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from propensao import config, data, tracking
from propensao.logging_config import configurar_logging

logger = configurar_logging()


def calcular_metricas(
    y_verdadeiro: pd.Series, probabilidades: np.ndarray, limiar: float = 0.5
) -> dict[str, float]:
    """Calcula as métricas do hold-out.

    Args:
        y_verdadeiro: rótulos observados (0/1).
        probabilidades: probabilidade predita da classe positiva.
        limiar: corte usado para as métricas que exigem decisão binária.

    Returns:
        Dicionário de métricas, com ``pr_auc`` como principal.
    """
    predicao = (probabilidades >= limiar).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_verdadeiro, probabilidades)),
        "roc_auc": float(roc_auc_score(y_verdadeiro, probabilidades)),
        "precisao": float(precision_score(y_verdadeiro, predicao, zero_division=0)),
        "recall": float(recall_score(y_verdadeiro, predicao, zero_division=0)),
        "f1": float(f1_score(y_verdadeiro, predicao, zero_division=0)),
    }


def avaliar(pipeline: Pipeline, teste: pd.DataFrame, limiar: float) -> dict[str, float]:
    """Aplica o pipeline no hold-out e devolve as métricas."""
    X, y = data.separar_alvo(teste)
    probabilidades = pipeline.predict_proba(X)[:, 1]
    return calcular_metricas(y, probabilidades, limiar)


def gravar_metricas(metricas: dict[str, float], caminho: Path) -> None:
    """Grava as métricas em JSON — formato que o DVC lê para ``dvc metrics diff``."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(metricas, indent=2) + "\n", encoding="utf-8")


def anexar_ao_run(metricas: dict[str, float], limiar: float) -> None:
    """Anexa métricas e limiar ao run que o estágio ``train`` abriu.

    Retomar o run em vez de abrir outro é o que mantém params, modelo e
    métricas na mesma linha da tabela de comparação do MLflow.

    Não interrompe o pipeline se não houver run: o ``metrics.json`` é entregável
    do DVC e precisa sair mesmo sem rastreamento disponível.
    """
    refs = tracking.ler_referencias()
    if refs is None:
        logger.warning("Sem referências do train — métricas não foram para o MLflow")
        return
    run_id = refs["run_id"]

    tracking.configurar()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("limiar", limiar)
        mlflow.log_metrics(metricas)
    logger.info("Métricas anexadas ao run %s", run_id)


def main() -> None:
    """Carrega modelo e hold-out, avalia e grava ``metrics/metrics.json``."""
    params = config.carregar_params()
    pipeline = joblib.load(config.CAMINHO_MODELO)
    teste = data.ler_parquet(config.CAMINHO_TESTE)

    limiar = params["evaluate"]["limiar"]
    metricas = avaliar(pipeline, teste, limiar)
    gravar_metricas(metricas, config.CAMINHO_METRICAS)
    anexar_ao_run(metricas, limiar)

    logger.info("PR-AUC: %.4f | ROC-AUC: %.4f", metricas["pr_auc"], metricas["roc_auc"])


if __name__ == "__main__":
    main()
