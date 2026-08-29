"""Estágio ``prepare`` do pipeline DVC: dado cru → partições de treino e teste."""

from propensao import config, data
from propensao.logging_config import configurar_logging

logger = configurar_logging()


def main() -> None:
    """Lê o CSV cru, remove duplicatas e grava as partições estratificadas."""
    params = config.carregar_params()
    brutos = data.carregar_csv(config.DADOS_BRUTOS)
    logger.info("Dataset cru: %s sessões", f"{len(brutos):,}")

    if params["prepare"]["remover_duplicatas"]:
        brutos = data.remover_duplicatas(brutos)
        logger.info("Após remover duplicatas: %s sessões", f"{len(brutos):,}")

    treino, teste = data.dividir_treino_teste(
        brutos, test_size=params["prepare"]["test_size"], seed=params["seed"]
    )
    data.gravar_parquet(treino, config.CAMINHO_TREINO)
    data.gravar_parquet(teste, config.CAMINHO_TESTE)
    logger.info("Treino: %s | Teste: %s", f"{len(treino):,}", f"{len(teste):,}")


if __name__ == "__main__":
    main()
