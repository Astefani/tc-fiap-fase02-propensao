"""Testes do pré-processamento — em especial, da exclusão de ``PageValues``."""

import pandas as pd
from sklearn.linear_model import LogisticRegression

from propensao.features import ConstrutorDeFeatures
from propensao.preprocess import construir_pipeline, construir_preprocessador


def _nomes_de_saida(dados: pd.DataFrame, usar_page_values: bool) -> list[str]:
    """Ajusta o pré-processador e devolve os nomes das colunas que ele produz."""
    com_features = ConstrutorDeFeatures(usar_page_values=usar_page_values).fit_transform(dados)
    preprocessador = construir_preprocessador(usar_page_values).fit(com_features)
    return list(preprocessador.get_feature_names_out())


def test_page_values_fica_fora_quando_desligado(X_y: tuple[pd.DataFrame, pd.Series]) -> None:
    """Nenhuma coluna de saída pode derivar de ``PageValues``.

    Esta é a trava da decisão de 29/08 (commit ``c7e533d``): ``PageValues`` é
    *near-target leakage* e custou 50% de PR-AUC para sair. Um ``remainder``
    esquecido no default a traria de volta em silêncio.
    """
    nomes = _nomes_de_saida(X_y[0], usar_page_values=False)

    assert not [nome for nome in nomes if "PageValues" in nome or "tem_page_value" in nome]


def test_page_values_entra_quando_ligado(X_y: tuple[pd.DataFrame, pd.Series]) -> None:
    """O caminho oposto — serve para provar que o teste acima não passa à toa."""
    nomes = _nomes_de_saida(X_y[0], usar_page_values=True)

    assert [nome for nome in nomes if "PageValues" in nome]
    assert [nome for nome in nomes if "tem_page_value" in nome]


def test_remainder_descarta_o_que_nao_foi_declarado(
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Colunas fora dos três grupos não sobrevivem ao ``remainder='drop'``."""
    X = X_y[0].copy()
    X["coluna_intrusa"] = 1

    nomes = _nomes_de_saida(X, usar_page_values=False)

    assert not [nome for nome in nomes if "coluna_intrusa" in nome]


def test_categoria_desconhecida_nao_quebra(X_y: tuple[pd.DataFrame, pd.Series]) -> None:
    """``handle_unknown='ignore'``: um mês não visto no treino não derruba a inferência."""
    X, y = X_y
    pipeline = construir_pipeline(LogisticRegression(max_iter=200), usar_page_values=False)
    pipeline.fit(X, y)

    inedito = X.head(1).copy()
    inedito["Month"] = "Mesinexistente"

    assert pipeline.predict_proba(inedito).shape == (1, 2)
