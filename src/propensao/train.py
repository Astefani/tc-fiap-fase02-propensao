"""Estágio ``train`` do pipeline DVC: partição de treino → ``models/pipeline.joblib``.

Abre o run do MLflow que o ``evaluate`` vai completar com as métricas.
"""

from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from propensao import config, data, tracking
from propensao.logging_config import configurar_logging
from propensao.model import criar_modelo_dos_params
from propensao.preprocess import construir_pipeline

logger = configurar_logging()


def treinar_com_rastreamento(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, params: dict[str, Any]
) -> None:
    """Ajusta o pipeline dentro de um run do MLflow e registra o que descreve o treino.

    O run fica aberto durante o ``fit`` para que uma falha de treino apareça no
    MLflow como run interrompido, em vez de sumir sem rastro.

    Args:
        pipeline: pipeline não treinado.
        X: features da partição de treino.
        y: alvo da partição de treino.
        params: conteúdo do ``params.yaml``.
    """
    tracking.configurar()
    with mlflow.start_run(run_name=params["train"]["algoritmo"]) as run:
        mlflow.log_params(tracking.params_de_treino(params))
        mlflow.set_tags(tracking.tags_de_proveniencia())

        logger.info("Treinando %s em %s sessões", params["train"]["algoritmo"], f"{len(X):,}")
        pipeline.fit(X, y)

        # signature e input_example viajam COM o modelo: quem o carregar do
        # Registry descobre o formato de entrada sem precisar deste código.
        info = mlflow.sklearn.log_model(
            pipeline,
            name="modelo",
            signature=infer_signature(X, pipeline.predict(X)),
            input_example=X.head(3),
            skops_trusted_types=tracking.TIPOS_CONFIAVEIS,
        )
        tracking.salvar_referencias(run.info.run_id, info.model_uri)
        logger.info("Run do MLflow: %s | modelo: %s", run.info.run_id, info.model_uri)


def main() -> None:
    """Treina o pipeline completo na partição de treino e salva o artefato."""
    params = config.carregar_params()
    treino = data.ler_parquet(config.CAMINHO_TREINO)
    X, y = data.separar_alvo(treino)

    modelo = criar_modelo_dos_params(params)
    pipeline = construir_pipeline(modelo, usar_page_values=params["train"]["usar_page_values"])
    treinar_com_rastreamento(pipeline, X, y, params)

    config.CAMINHO_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.CAMINHO_MODELO)
    logger.info("Artefato salvo em %s", config.CAMINHO_MODELO.relative_to(config.RAIZ))


if __name__ == "__main__":
    main()
