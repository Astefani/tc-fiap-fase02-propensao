"""Fábrica de estimadores — traduz o ``params.yaml`` em um objeto scikit-learn.

Trocar de algoritmo é editar uma linha do ``params.yaml``; o DVC detecta a mudança
e re-executa apenas os estágios afetados.
"""

from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

FABRICAS: dict[str, type[ClassifierMixin]] = {
    "dummy": DummyClassifier,
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}


def criar_modelo(algoritmo: str, hiperparametros: dict[str, Any], seed: int) -> ClassifierMixin:
    """Instancia o estimador escolhido no ``params.yaml``.

    Args:
        algoritmo: chave em :data:`FABRICAS`.
        hiperparametros: argumentos do estimador, vindos do ``params.yaml``.
        seed: semente propagada a todos os estimadores, para reprodutibilidade.

    Returns:
        Estimador pronto para entrar no Pipeline.

    Raises:
        ValueError: se o algoritmo não estiver registrado.
    """
    if algoritmo not in FABRICAS:
        conhecidos = ", ".join(sorted(FABRICAS))
        raise ValueError(f"Algoritmo desconhecido: {algoritmo!r}. Conhecidos: {conhecidos}.")

    return FABRICAS[algoritmo](random_state=seed, **hiperparametros)


def criar_modelo_dos_params(params: dict[str, Any]) -> ClassifierMixin:
    """Atalho que lê o algoritmo, seus hiperparâmetros e a seed do ``params.yaml``."""
    algoritmo = params["train"]["algoritmo"]
    return criar_modelo(algoritmo, params["train"].get(algoritmo, {}), params["seed"])
