"""Transformador de features derivadas do comportamento de navegação."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from propensao.config import COLUNA_TEM_PAGE_VALUE


class ConstrutorDeFeatures(BaseEstimator, TransformerMixin):
    """Cria features derivadas a partir das colunas cruas de navegação.

    É **stateless**: ``fit`` não aprende nada dos dados, então não há risco de
    vazamento entre treino e teste. Vive dentro do Pipeline para que o artefato
    serializado saiba transformar dados crus sozinho, sem código externo.

    Features criadas:

    - ``total_paginas``: soma das três contagens de página da sessão.
    - ``duracao_total``: soma das três durações.
    - ``duracao_por_pagina``: tempo médio por página — distingue quem lê de quem
      passa rápido, o que as contagens sozinhas não capturam.
    - ``tem_page_value``: indicador de ``PageValues > 0``. Só é criada quando
      ``usar_page_values`` é verdadeiro; caso contrário reintroduziria pela porta
      dos fundos o sinal que se quis remover.

    Args:
        usar_page_values: se ``False``, não cria o indicador de ``PageValues``.
    """

    CONTAGENS = ["Administrative", "Informational", "ProductRelated"]
    DURACOES = [
        "Administrative_Duration",
        "Informational_Duration",
        "ProductRelated_Duration",
    ]

    def __init__(self, usar_page_values: bool = True) -> None:
        self.usar_page_values = usar_page_values

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ConstrutorDeFeatures":
        """Não aprende nada; existe para cumprir o contrato do scikit-learn."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Devolve uma cópia do DataFrame com as features derivadas."""
        dados = X.copy()
        dados["total_paginas"] = dados[self.CONTAGENS].sum(axis=1)
        dados["duracao_total"] = dados[self.DURACOES].sum(axis=1)
        dados["duracao_por_pagina"] = dados["duracao_total"] / (dados["total_paginas"] + 1)

        if self.usar_page_values:
            dados[COLUNA_TEM_PAGE_VALUE] = (dados["PageValues"] > 0).astype(int)

        return dados
