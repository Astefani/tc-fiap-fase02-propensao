"""Fixtures compartilhadas.

A suíte roda contra ``tests/data/shoppers_sample.csv``, que é **commitado** — e é
por isso que ela funciona num clone sem ``dvc pull``. O dado de verdade é grande e
mora no DVC; os testes não deveriam depender de rede nem de credencial.
"""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from propensao import data

CAMINHO_AMOSTRA = Path(__file__).parent / "data" / "shoppers_sample.csv"


@pytest.fixture(scope="session")
def amostra() -> pd.DataFrame:
    """Dataset cru reduzido, 300 sessões estratificadas pelo alvo."""
    return data.carregar_csv(CAMINHO_AMOSTRA)


@pytest.fixture(scope="session")
def X_y(amostra: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features e alvo separados, como o pipeline recebe."""
    return data.separar_alvo(amostra)


@pytest.fixture
def estimador() -> LogisticRegression:
    """Estimador barato — os testes verificam encanamento, não desempenho."""
    return LogisticRegression(max_iter=200, random_state=42)
