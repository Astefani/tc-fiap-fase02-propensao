"""Promoção de um modelo no MLflow Model Registry — CLI, **fora** do pipeline.

Por que não é um estágio do ``dvc.yaml``: o DVC reproduz transformações
determinísticas, e promover um modelo é uma **decisão**. Como estágio, todo
``dvc repro`` promoveria algo — inclusive um experimento ruim.

A decisão em si já está registrada no Git: ``train.algoritmo`` no ``params.yaml``
diz qual candidato foi escolhido, e o pipeline treinou aquele. Este script só
executa o que o repositório já declara, lendo as referências que o ``train``
gravou.

Uso::

    python -m propensao.register                      # promove como @champion
    python -m propensao.register --alias challenger   # registra sem despromover
"""

import argparse

import mlflow
from mlflow import MlflowClient

from propensao import config, tracking
from propensao.logging_config import configurar_logging

logger = configurar_logging()


def montar_descricao(params: dict[str, str], metricas: dict[str, float]) -> str:
    """Escreve a descrição que aparece na versão dentro do Registry."""
    return (
        f"Algoritmo: {params.get('algoritmo')} · "
        f"PR-AUC {metricas.get('pr_auc', float('nan')):.4f} · "
        f"ROC-AUC {metricas.get('roc_auc', float('nan')):.4f} · "
        f"limiar {params.get('limiar')} · "
        f"PageValues fora do modelo (near-target leakage)."
    )


def promover(alias: str) -> None:
    """Registra o modelo do estado atual do pipeline e aponta ``alias`` para ele.

    Args:
        alias: nome do ponteiro no Registry. ``champion`` é o modelo vigente;
            outros valores permitem registrar sem despromover o atual.

    Raises:
        FileNotFoundError: se o ``train`` ainda não rodou neste workspace.
    """
    refs = tracking.ler_referencias()
    if refs is None:
        raise FileNotFoundError(
            f"{config.CAMINHO_REFS_MLFLOW} não existe — rode `make repro` antes de promover."
        )

    tracking.configurar()
    cliente = MlflowClient()
    nome = config.ambiente.mlflow_registered_model
    run = cliente.get_run(refs["run_id"])

    versao = mlflow.register_model(refs["model_uri"], nome)
    logger.info("Registrado %s versão %s", nome, versao.version)

    # Tags de proveniência NA VERSÃO, não só no run: quem abrir o Registry vê de
    # qual código e de quais dados o modelo saiu, sem precisar caçar o run.
    for chave, valor in {**tracking.tags_de_proveniencia(), **run.data.params}.items():
        cliente.set_model_version_tag(nome, versao.version, chave, str(valor))

    cliente.update_model_version(
        name=nome,
        version=versao.version,
        description=montar_descricao(run.data.params, run.data.metrics),
    )

    cliente.set_registered_model_alias(nome, alias, versao.version)
    logger.info("Alias @%s agora aponta para a versão %s", alias, versao.version)
    logger.info('Consumo: mlflow.sklearn.load_model("models:/%s@%s")', nome, alias)


def main() -> None:
    """Lê os argumentos da linha de comando e promove."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--alias",
        default="champion",
        help="ponteiro a mover no Registry (padrão: champion)",
    )
    promover(parser.parse_args().alias)


if __name__ == "__main__":
    main()
