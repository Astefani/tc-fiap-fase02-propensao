"""Construção do Pipeline scikit-learn — a espinha dorsal do projeto.

Um único ``Pipeline`` serializável recebe dados **crus** e devolve probabilidade.
É ele que é salvo em ``models/pipeline.joblib`` e que serve treino e inferência,
garantindo que o mesmo pré-processamento seja aplicado nos dois momentos.
"""

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from propensao.config import (
    COLUNA_TEM_PAGE_VALUE,
    COLUNAS_ASSIMETRICAS,
    COLUNAS_CATEGORICAS,
    COLUNAS_DERIVADAS,
    COLUNAS_NUMERICAS,
)
from propensao.features import ConstrutorDeFeatures

COLUNA_PAGE_VALUES = "PageValues"


def _colunas_assimetricas(usar_page_values: bool) -> list[str]:
    """Lista as colunas de cauda longa, com ou sem ``PageValues``."""
    colunas = [*COLUNAS_ASSIMETRICAS, *COLUNAS_DERIVADAS]
    if usar_page_values:
        return colunas
    return [coluna for coluna in colunas if coluna != COLUNA_PAGE_VALUES]


def _colunas_numericas(usar_page_values: bool) -> list[str]:
    """Lista as colunas de escala curta, incluindo o indicador binário se houver."""
    if usar_page_values:
        return [*COLUNAS_NUMERICAS, COLUNA_TEM_PAGE_VALUE]
    return list(COLUNAS_NUMERICAS)


def construir_preprocessador(usar_page_values: bool = True) -> ColumnTransformer:
    """Monta o ``ColumnTransformer`` com um tratamento por grupo de colunas.

    Args:
        usar_page_values: quando ``False``, ``PageValues`` e seu indicador ficam de
            fora — usado para medir quanto do desempenho depende dessa variável.

    Returns:
        O transformador pronto para entrar no Pipeline.
    """
    log_e_escala = Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("escala", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("assimetricas", log_e_escala, _colunas_assimetricas(usar_page_values)),
            ("numericas", StandardScaler(), _colunas_numericas(usar_page_values)),
            # sparse_output=False porque o HistGradientBoosting exige matriz densa.
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                COLUNAS_CATEGORICAS,
            ),
        ]
    )


def construir_pipeline(modelo: ClassifierMixin, usar_page_values: bool = True) -> Pipeline:
    """Encadeia features derivadas, pré-processamento e estimador.

    Args:
        modelo: estimador scikit-learn já configurado.
        usar_page_values: repassado ao construtor de features e ao pré-processador.

    Returns:
        Pipeline treinável e serializável que aceita o DataFrame cru.
    """
    return Pipeline(
        [
            ("features", ConstrutorDeFeatures(usar_page_values=usar_page_values)),
            ("preprocessador", construir_preprocessador(usar_page_values)),
            ("modelo", modelo),
        ]
    )
