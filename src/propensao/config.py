"""Configuração central do projeto: caminhos, grupos de colunas e ambiente.

Três camadas, com fronteiras deliberadas:

- ``params.yaml`` — decisões de modelagem; mudá-las invalida o modelo e o DVC re-executa
  os estágios afetados.
- ``.env`` — configuração de ambiente (URIs, nível de log); não afeta o modelo.
- este módulo — constantes de domínio e caminhos, que são código.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

RAIZ = Path(__file__).resolve().parents[2]


class Ambiente(BaseSettings):
    """Variáveis de ambiente lidas do ``.env``, com defaults utilizáveis.

    Os defaults existem para que o projeto rode num clone limpo, sem ``.env``.
    """

    mlflow_tracking_uri: str = "sqlite:///mlflow-store/mlflow.db"
    mlflow_experiment_name: str = "propensao-compra"
    mlflow_registered_model: str = "propensao-compra"

    #: Só usados quando o tracking é um servidor remoto (DagsHub). Vazios com o
    #: default sqlite, que não autentica.
    mlflow_tracking_username: str = ""
    mlflow_tracking_password: str = ""

    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=RAIZ / ".env", extra="ignore")


ambiente = Ambiente()


def aplicar_credenciais_mlflow() -> None:
    """Exporta as credenciais de tracking para o ambiente do processo.

    O MLflow lê ``MLFLOW_TRACKING_USERNAME`` e ``MLFLOW_TRACKING_PASSWORD`` de
    ``os.environ``, não de um objeto de configuração — e o pydantic-settings
    carrega o ``.env`` para o objeto ``ambiente``, sem tocar em ``os.environ``.
    Esta função é a ponte entre os dois.

    Não faz nada quando as credenciais estão vazias, que é o caso do default
    sqlite. Nunca sobrescreve o que já veio do ambiente: variável exportada no
    shell ou injetada pelo container tem precedência sobre o ``.env``.
    """
    credenciais = {
        "MLFLOW_TRACKING_USERNAME": ambiente.mlflow_tracking_username,
        "MLFLOW_TRACKING_PASSWORD": ambiente.mlflow_tracking_password,
    }
    for chave, valor in credenciais.items():
        if valor and chave not in os.environ:
            os.environ[chave] = valor

# --- Caminhos ---------------------------------------------------------------
CAMINHO_PARAMS = RAIZ / "params.yaml"
DADOS_BRUTOS = RAIZ / "data" / "raw" / "online_shoppers_intention.csv"
DIR_PROCESSADO = RAIZ / "data" / "processed"
CAMINHO_TREINO = DIR_PROCESSADO / "treino.parquet"
CAMINHO_TESTE = DIR_PROCESSADO / "teste.parquet"
DIR_MODELOS = RAIZ / "models"
CAMINHO_MODELO = DIR_MODELOS / "pipeline.joblib"

#: Ponte entre os estágios `train`, `evaluate` e o `register`: eles rodam em
#: processos separados, e é por este arquivo que as métricas chegam ao run certo
#: e que a promoção sabe qual modelo registrar.
CAMINHO_REFS_MLFLOW = DIR_MODELOS / "mlflow_run.json"
DIR_METRICAS = RAIZ / "metrics"
CAMINHO_METRICAS = DIR_METRICAS / "metrics.json"

# --- Colunas ----------------------------------------------------------------
ALVO = "Revenue"

#: Contagens e durações de navegação: caudas longas (assimetria de 2 a 7,6) → log1p.
COLUNAS_ASSIMETRICAS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "PageValues",
]

#: Já limitadas a faixas curtas: só padronização.
COLUNAS_NUMERICAS = ["BounceRates", "ExitRates", "SpecialDay"]

#: ``OperatingSystems``, ``Browser``, ``Region`` e ``TrafficType`` são inteiros que
#: identificam categorias, não ordens — por isso entram no one-hot.
COLUNAS_CATEGORICAS = [
    "Month",
    "VisitorType",
    "Weekend",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

#: Features derivadas criadas pelo ConstrutorDeFeatures.
COLUNAS_DERIVADAS = ["total_paginas", "duracao_total", "duracao_por_pagina"]

#: Criada só quando PageValues está em uso (ver ConstrutorDeFeatures).
COLUNA_TEM_PAGE_VALUE = "tem_page_value"


def carregar_params(caminho: Path = CAMINHO_PARAMS) -> dict[str, Any]:
    """Lê o ``params.yaml``.

    Args:
        caminho: arquivo de parâmetros; o default é o da raiz do projeto.

    Returns:
        Dicionário com as seções ``seed``, ``prepare``, ``train`` e ``evaluate``.
    """
    with caminho.open(encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)
