"""Testes do ``ConstrutorDeFeatures``."""

import pandas as pd

from propensao.config import COLUNA_TEM_PAGE_VALUE
from propensao.features import ConstrutorDeFeatures


def test_cria_as_tres_features_derivadas(amostra: pd.DataFrame) -> None:
    """As três derivadas aparecem, independente de ``usar_page_values``."""
    resultado = ConstrutorDeFeatures(usar_page_values=False).fit_transform(amostra)

    for coluna in ("total_paginas", "duracao_total", "duracao_por_pagina"):
        assert coluna in resultado.columns


def test_duracao_por_pagina_nao_divide_por_zero() -> None:
    """Sessão sem página nenhuma não pode virar divisão por zero.

    A entrada é construída de propósito: a amostra commitada **não** contém esse
    caso de borda, e um teste que depende dela passaria por sorte de amostragem.
    É o ``+1`` no denominador que sustenta esta garantia.
    """
    sessao_vazia = pd.DataFrame(
        {
            "Administrative": [0],
            "Informational": [0],
            "ProductRelated": [0],
            "Administrative_Duration": [0.0],
            "Informational_Duration": [0.0],
            "ProductRelated_Duration": [0.0],
        }
    )

    resultado = ConstrutorDeFeatures(usar_page_values=False).fit_transform(sessao_vazia)

    assert resultado["total_paginas"].iloc[0] == 0
    assert resultado["duracao_por_pagina"].notna().all()
    assert resultado["duracao_por_pagina"].iloc[0] == 0.0


def test_indicador_de_page_values_respeita_a_decisao(amostra: pd.DataFrame) -> None:
    """O indicador só existe quando ``PageValues`` está em uso.

    É a trava da decisão de 29/08: sem ela, ``tem_page_value`` reintroduziria pela
    porta dos fundos exatamente o sinal com vazamento que se quis remover.
    """
    com = ConstrutorDeFeatures(usar_page_values=True).fit_transform(amostra)
    sem = ConstrutorDeFeatures(usar_page_values=False).fit_transform(amostra)

    assert COLUNA_TEM_PAGE_VALUE in com.columns
    assert COLUNA_TEM_PAGE_VALUE not in sem.columns


def test_e_stateless(amostra: pd.DataFrame) -> None:
    """``fit`` não aprende nada — é o que garante ausência de vazamento.

    Ajustar na metade de cima e transformar a de baixo tem que dar o mesmo que
    transformar sem ajuste nenhum.
    """
    metade = len(amostra) // 2
    transformador = ConstrutorDeFeatures(usar_page_values=False)

    ajustado_na_metade = transformador.fit(amostra.iloc[:metade]).transform(amostra)
    sem_ajuste = ConstrutorDeFeatures(usar_page_values=False).transform(amostra)

    pd.testing.assert_frame_equal(ajustado_na_metade, sem_ajuste)


def test_nao_altera_o_dataframe_de_entrada(amostra: pd.DataFrame) -> None:
    """``transform`` trabalha sobre cópia — a fixture é de sessão e é reusada."""
    antes = amostra.copy()
    ConstrutorDeFeatures(usar_page_values=True).fit_transform(amostra)

    pd.testing.assert_frame_equal(amostra, antes)
