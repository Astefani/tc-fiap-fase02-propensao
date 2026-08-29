"""Estágio ``train`` do pipeline DVC: partição de treino → ``models/pipeline.joblib``.

O rastreamento no MLflow entra neste módulo no bloco B7.
"""

import joblib

from propensao import config, data
from propensao.logging_config import configurar_logging
from propensao.model import criar_modelo_dos_params
from propensao.preprocess import construir_pipeline

logger = configurar_logging()


def main() -> None:
    """Treina o pipeline completo na partição de treino e salva o artefato."""
    params = config.carregar_params()
    treino = data.ler_parquet(config.CAMINHO_TREINO)
    X, y = data.separar_alvo(treino)

    modelo = criar_modelo_dos_params(params)
    pipeline = construir_pipeline(modelo, usar_page_values=params["train"]["usar_page_values"])

    logger.info("Treinando %s em %s sessões", params["train"]["algoritmo"], f"{len(X):,}")
    pipeline.fit(X, y)

    config.CAMINHO_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.CAMINHO_MODELO)
    logger.info("Artefato salvo em %s", config.CAMINHO_MODELO.relative_to(config.RAIZ))


if __name__ == "__main__":
    main()
