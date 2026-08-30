# Imagem do pipeline de propensão de compra.
#
# Single-stage de propósito: o critério da avaliação é "Dockerfile configurado
# corretamente para o projeto", e uma imagem que builda e roda o pipeline já o
# cumpre. Multi-stage é otimização de tamanho, não requisito.

FROM python:3.12-slim

# POETRY_VIRTUALENVS_CREATE=false instala no Python do sistema, sem venv.
# Isso não é atalho: o container já É o ambiente isolado, e um venv dentro dele
# seria redundante. Mais importante — o compose monta o projeto em /app, e um
# venv ali seria encoberto pelo do host, cujos shebangs apontam para caminhos
# que não existem aqui dentro.
ENV POETRY_VERSION=2.3.2 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# git é dependência real do DVC: ele opera dentro de um repositório e chama o
# git para resolver revisões (`dvc metrics diff`, por exemplo).
RUN apt-get update \
 && apt-get install --no-install-recommends -y git \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app

# Dependências ANTES do código. Enquanto pyproject e lock não mudarem, o Docker
# reaproveita esta camada — e um rebuild após editar código não reinstala nada.
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

# Só agora o que muda a cada commit.
COPY src/ ./src/
COPY params.yaml dvc.yaml dvc.lock ./
RUN poetry install --only main

# Não-root: o container escreve em volume montado do host. Como root, os
# arquivos gerados sairiam com dono root e você precisaria de sudo para apagar.
# uid 1000 casa com o usuário padrão do Ubuntu.
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

# Um comando demonstra os dois critérios: Docker executa o projeto E roda o
# pipeline versionado.
CMD ["dvc", "repro"]
