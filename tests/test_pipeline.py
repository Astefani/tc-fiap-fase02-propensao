"""Testes do Pipeline servível — o artefato que vai para produção."""

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline

from propensao.preprocess import construir_pipeline


def test_aceita_dados_crus(X_y: tuple[pd.DataFrame, pd.Series], estimador: ClassifierMixin) -> None:
    """O Pipeline recebe o DataFrame cru, sem pré-processamento manual.

    É a garantia central do desenho: quem consome o artefato não precisa repetir
    nenhuma transformação, então treino e inferência não podem divergir.
    """
    X, y = X_y
    pipeline = construir_pipeline(estimador, usar_page_values=False)
    pipeline.fit(X, y)

    probabilidades = pipeline.predict_proba(X)

    assert probabilidades.shape == (len(X), 2)
    assert np.all((probabilidades >= 0) & (probabilidades <= 1))


def test_tem_as_tres_etapas(estimador: ClassifierMixin) -> None:
    """Features derivadas, pré-processamento e estimador, nesta ordem."""
    pipeline = construir_pipeline(estimador, usar_page_values=False)

    assert [nome for nome, _ in pipeline.steps] == ["features", "preprocessador", "modelo"]


def test_sobrevive_a_dump_e_load(
    X_y: tuple[pd.DataFrame, pd.Series], estimador: ClassifierMixin, tmp_path
) -> None:
    """Serializar e recarregar tem que devolver as MESMAS probabilidades.

    Sem isto, o modelo publicado pode divergir do avaliado — e nenhuma métrica
    registrada no MLflow valeria nada.
    """
    X, y = X_y
    pipeline = construir_pipeline(estimador, usar_page_values=False)
    pipeline.fit(X, y)
    antes = pipeline.predict_proba(X)[:, 1]

    caminho = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, caminho)
    recarregado: Pipeline = joblib.load(caminho)

    np.testing.assert_array_equal(antes, recarregado.predict_proba(X)[:, 1])
