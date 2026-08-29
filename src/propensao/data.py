"""Entrada e saída de dados e separação treino/teste.

Este módulo só carrega, separa e grava. Nenhuma transformação de feature acontece
aqui — isso é responsabilidade do pipeline, para que o artefato salvo saiba lidar
sozinho com dados crus.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from propensao.config import ALVO


def carregar_csv(caminho: Path) -> pd.DataFrame:
    """Lê o dataset cru a partir de um CSV."""
    return pd.read_csv(caminho)


def ler_parquet(caminho: Path) -> pd.DataFrame:
    """Lê uma partição processada."""
    return pd.read_parquet(caminho)


def gravar_parquet(dados: pd.DataFrame, caminho: Path) -> None:
    """Grava um DataFrame em parquet, criando o diretório se necessário."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    dados.to_parquet(caminho, index=False)


def remover_duplicatas(dados: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas exatamente iguais.

    São 125 no dataset original. Duas sessões idênticas são plausíveis na vida real,
    mas duplicatas exatas vazam entre treino e teste no split.
    """
    return dados.drop_duplicates()


def separar_alvo(dados: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa as features do alvo, convertendo o alvo booleano para inteiro."""
    return dados.drop(columns=ALVO), dados[ALVO].astype(int)


def dividir_treino_teste(
    dados: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o dataset preservando a proporção do alvo nas duas partições.

    Args:
        dados: dataset completo, com a coluna alvo.
        test_size: fração destinada ao hold-out.
        seed: semente para tornar o split reproduzível.

    Returns:
        Tupla ``(treino, teste)``.
    """
    return train_test_split(
        dados, test_size=test_size, stratify=dados[ALVO], random_state=seed
    )
