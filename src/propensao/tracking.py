"""Rastreamento de experimentos no MLflow.

Concentra aqui tudo que fala com o MLflow, para que ``train`` e ``evaluate``
continuem sendo sobre treinar e avaliar.

**Por que existe um arquivo de ``run_id``:** os parâmetros e o modelo nascem no
estágio ``train``, mas as métricas nascem no ``evaluate`` — que o DVC executa
como um processo separado. Para que tudo caia no *mesmo* run do MLflow, o
``train`` grava o identificador e o ``evaluate`` retoma o run por ele.
"""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import mlflow

from propensao import config
from propensao.features import ConstrutorDeFeatures
from propensao.logging_config import configurar_logging

logger = configurar_logging()

#: O MLflow 3 serializa modelos sklearn com **skops**, que — ao contrário do
#: pickle — se recusa a desserializar classes que não conhece. Nosso transformer
#: custom precisa ser declarado confiável, ou o ``log_model`` falha.
#: Derivado da própria classe, para que um rename não quebre isto em silêncio.
TIPOS_CONFIAVEIS = [f"{ConstrutorDeFeatures.__module__}.{ConstrutorDeFeatures.__qualname__}"]


def configurar() -> None:
    """Aponta o MLflow para o backend do ``.env`` e garante o experimento.

    Idempotente: chamar de novo não cria experimento duplicado.
    """
    config.aplicar_credenciais_mlflow()
    mlflow.set_tracking_uri(config.ambiente.mlflow_tracking_uri)
    mlflow.set_experiment(config.ambiente.mlflow_experiment_name)
    logger.info("MLflow em %s", config.ambiente.mlflow_tracking_uri)


def _hash_do_arquivo(caminho: Path) -> str:
    """Devolve o md5 de um arquivo, ou ``"ausente"`` se ele não existir."""
    if not caminho.exists():
        return "ausente"
    return hashlib.md5(caminho.read_bytes()).hexdigest()


def _commit_atual() -> str:
    """Devolve o SHA do HEAD, ou ``"desconhecido"`` fora de um repositório."""
    try:
        resultado = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.RAIZ,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconhecido"
    return resultado.stdout.strip()


def tags_de_proveniencia() -> dict[str, str]:
    """Tags que costuram o run do MLflow ao estado do repositório.

    É esse par que responde "de qual código e de quais dados saiu este modelo?":
    o commit identifica o código, e o hash do ``dvc.lock`` identifica os dados e
    os artefatos de cada estágio.
    """
    return {
        "git_commit": _commit_atual(),
        "dvc_lock_hash": _hash_do_arquivo(config.RAIZ / "dvc.lock"),
    }


def params_de_treino(params: dict[str, Any]) -> dict[str, Any]:
    """Achata o ``params.yaml`` no conjunto que descreve ESTE treino.

    Só os hiperparâmetros do algoritmo escolhido entram — logar os quatro
    candidatos em todo run encheria a tabela de comparação de colunas vazias.
    """
    algoritmo = params["train"]["algoritmo"]
    achatado: dict[str, Any] = {
        "seed": params["seed"],
        "algoritmo": algoritmo,
        "usar_page_values": params["train"]["usar_page_values"],
        "remover_duplicatas": params["prepare"]["remover_duplicatas"],
        "test_size": params["prepare"]["test_size"],
    }
    for chave, valor in params["train"].get(algoritmo, {}).items():
        achatado[f"{algoritmo}.{chave}"] = valor
    return achatado


def salvar_referencias(run_id: str, model_uri: str) -> None:
    """Grava as referências do MLflow para os estágios seguintes.

    Guarda também a ``model_uri`` devolvida pelo ``log_model``: no MLflow 3 o
    modelo é uma entidade própria (``LoggedModel``, id ``m-…``), e não um
    artefato do run — a URI é o endereço confiável para registrá-lo depois.
    """
    config.CAMINHO_REFS_MLFLOW.parent.mkdir(parents=True, exist_ok=True)
    conteudo = {"run_id": run_id, "model_uri": model_uri}
    config.CAMINHO_REFS_MLFLOW.write_text(json.dumps(conteudo, indent=2) + "\n", encoding="utf-8")


def ler_referencias() -> dict[str, str] | None:
    """Lê as referências gravadas pelo ``train``; ``None`` se não houver."""
    if not config.CAMINHO_REFS_MLFLOW.exists():
        return None
    return json.loads(config.CAMINHO_REFS_MLFLOW.read_text(encoding="utf-8"))
