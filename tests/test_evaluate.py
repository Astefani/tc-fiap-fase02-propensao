"""Testes das métricas e da gravação do ``metrics.json``."""

import json

import numpy as np
import pandas as pd

from propensao.evaluate import calcular_metricas, gravar_metricas

CHAVES = {"pr_auc", "roc_auc", "precisao", "recall", "f1"}


def test_separacao_perfeita_leva_as_metricas_ao_maximo() -> None:
    """Com separação perfeita, PR-AUC e ROC-AUC valem 1."""
    y = pd.Series([0, 0, 1, 1])
    probabilidades = np.array([0.01, 0.02, 0.98, 0.99])

    metricas = calcular_metricas(y, probabilidades, limiar=0.5)

    assert set(metricas) == CHAVES
    assert metricas["pr_auc"] == 1.0
    assert metricas["roc_auc"] == 1.0
    assert metricas["f1"] == 1.0


def test_limiar_troca_recall_por_precisao() -> None:
    """Subir o limiar reduz o recall — é o trade-off que `evaluate.limiar` arbitra."""
    y = pd.Series([0, 1, 1, 1])
    probabilidades = np.array([0.10, 0.40, 0.60, 0.90])

    permissivo = calcular_metricas(y, probabilidades, limiar=0.30)
    restritivo = calcular_metricas(y, probabilidades, limiar=0.70)

    assert permissivo["recall"] > restritivo["recall"]


def test_metricas_independentes_de_limiar_nao_se_movem() -> None:
    """PR-AUC e ROC-AUC ignoram o corte — por isso comparam candidatos.

    É o que sustenta a leitura da tabela de candidatos: precisão, recall e F1
    não são comparáveis entre modelos com limiares distintos; estas duas são.
    """
    y = pd.Series([0, 1, 1, 1])
    probabilidades = np.array([0.10, 0.40, 0.60, 0.90])

    baixo = calcular_metricas(y, probabilidades, limiar=0.30)
    alto = calcular_metricas(y, probabilidades, limiar=0.70)

    assert baixo["pr_auc"] == alto["pr_auc"]
    assert baixo["roc_auc"] == alto["roc_auc"]


def test_grava_json_legivel_pelo_dvc(tmp_path) -> None:
    """O arquivo tem que ser JSON válido — é o que o ``dvc metrics diff`` lê."""
    metricas = {"pr_auc": 0.3563, "roc_auc": 0.7741}
    caminho = tmp_path / "sub" / "metrics.json"

    gravar_metricas(metricas, caminho)

    assert json.loads(caminho.read_text(encoding="utf-8")) == metricas
