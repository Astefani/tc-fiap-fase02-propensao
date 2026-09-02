# Propensão de Compra em E-commerce — Tech Challenge Fase 02

Pipeline de Machine Learning **reproduzível e containerizado** que estima a probabilidade de uma
sessão de navegação terminar em compra, do dado cru ao modelo promovido no Model Registry.
Projeto desenvolvido para o Tech Challenge da Fase 02 da Pós Tech ML Engineering (FIAP + Alura).

**Autor:** Alessandro Stefani  
**Modelo central:** RandomForest (scikit-learn) num Pipeline servível único, com dados versionados em DVC, experimentos e Model Registry no MLflow, e execução containerizada em Docker.  
**Dataset:** [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (UCI) — 12.330 sessões

[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![Poetry](https://img.shields.io/badge/deps-poetry%202.3-blueviolet)]()
[![DVC](https://img.shields.io/badge/data-DVC%203.67-945dd6)]()
[![MLflow](https://img.shields.io/badge/tracking-MLflow%203.15-0194e2)]()
[![Docker](https://img.shields.io/badge/container-docker-2496ed)]()

> 🎥 **Vídeo de apresentação (5 min):** <https://youtu.be/zR_Lred9zJ8>  
> 📊 **Experimentos e Model Registry:** <https://dagshub.com/Astefani/tc-fiap-fase02-propensao.mlflow>  
> 💾 **Repositório:** <https://github.com/Astefani/tc-fiap-fase02-propensao>

---

## O que este projeto é — e o que ele deliberadamente não é

O enunciado é explícito: *"o foco deste desafio **não é a complexidade matemática do modelo**, mas
sim a **Engenharia de Machine Learning**"*. Os critérios de avaliação confirmam — 90% do peso está
em Clean Code, ambiente reprodutível, rastreamento de experimentos, containerização e versionamento
de dados; 10% na modelagem.

Este repositório foi construído nessa proporção. **Um modelo simples bem empacotado vale mais que
um modelo sofisticado mal empacotado** — e a decisão mais importante aqui foi jogar fora metade do
poder preditivo, de propósito e por um bom motivo. Está em
[Decisões](#decisões-e-o-porquê-de-cada-uma).

---

## Começando

### Pré-requisitos

| Ferramenta | Versão | Para quê |
|---|---|---|
| Python | **3.12** (o `pyproject` exige `>=3.12,<3.13`) | rodar o projeto |
| Poetry | 2.x | instalar o ambiente a partir do `poetry.lock` |
| Git | qualquer | clonar — e o DVC o usa em tempo de execução |
| `make`, `curl`, `unzip` | do sistema | atalhos do Makefile e download do dataset |
| Docker | opcional | rodar o pipeline containerizado |

Não precisa de GPU nem de credencial. O pipeline completo roda em menos de dois minutos numa
máquina comum.

### Reproduzir o pipeline — sem credencial nenhuma

```bash
git clone https://github.com/Astefani/tc-fiap-fase02-propensao.git
cd tc-fiap-fase02-propensao
cp .env.example .env
poetry install
make data      # baixa o CSV cru do UCI e confere contra o ponteiro .dvc
make repro     # prepare -> train -> evaluate
```

O `make data` existe para que **nada precise de token**. O dataset é público, e o md5 do arquivo
baixado bate com o que o ponteiro `.dvc` versiona — o próprio comando confere:

```
$ make data
...
--- conferindo contra o ponteiro versionado ---
data/raw/online_shoppers_intention.csv.dvc:
        changed outs:
                not in cache:       data/raw/online_shoppers_intention.csv
```

**`not in cache` não é erro.** O md5 do arquivo baixado é idêntico ao que o ponteiro versiona
(`cc6ec1db03b4f10f8de52c56ff48b085`); o que falta é a cópia no *cache* local do DVC, que só o
`dvc pull` traz — e o `make data` existe justamente para dispensá-lo. O `dvc repro` confere a
fonte antes de usá-la (`Verifying data sources in stage: ...csv.dvc`) e pararia se o hash
divergisse.

Com o `.env.example` copiado sem alterações, o MLflow grava numa SQLite local e o pipeline roda
offline. Nada sai da máquina.

### Ver a ajuda

```
$ make
  help        Mostra esta ajuda
  install     Instala o ambiente (produção + dev)
  lint        Ruff sobre o código
  test        Suíte de testes
  data        Baixa o dataset cru do UCI (alternativa ao `dvc pull`, sem credencial)
  repro       Executa o pipeline DVC (pula estágios não afetados)
  metrics     Métricas do último run e diferença para o commit anterior
  register    Promove o modelo atual no Model Registry como @champion
  push        Envia dados e artefatos para o remote do DVC
  pull        Traz dados e artefatos do remote do DVC
  mlflow-ui   Abre a UI do MLflow em http://localhost:5000
  clean       Remove artefatos gerados (o DVC reconstrói com `make repro`)
```

### Puxar os dados versionados (opcional, requer token)

```bash
poetry run dvc remote modify --local dagshub access_key_id      <SEU_TOKEN>
poetry run dvc remote modify --local dagshub secret_access_key  <SEU_TOKEN>
poetry run dvc pull
```

Token gratuito em [dagshub.com](https://dagshub.com) → *Settings* → *Tokens*.
**`dvc pull` anônimo não funciona** — dá `Unable to locate credentials`. Quem só quer reproduzir
usa `make data`, que dispensa isso.

---

## Arquitetura

```
data/raw/*.csv           ← versionado pelo DVC (o Git guarda só o ponteiro .dvc)
        │
   [ prepare ]           remove duplicatas · split estratificado 80/20
        │
data/processed/*.parquet
        │
   [ train ]             Pipeline sklearn único · abre o run no MLflow
        │
models/pipeline.joblib + models/mlflow_run.json
        │
   [ evaluate ]          métricas no hold-out · anexa ao mesmo run
        │
metrics/metrics.json     ← versionado no Git (é o histórico de experimentos)
```

O grafo real, direto da ferramenta:

```
$ poetry run dvc dag
+--------------------------------------------+
| data/raw/online_shoppers_intention.csv.dvc |
+--------------------------------------------+
                       *
                  +---------+
                  | prepare |
                  +---------+
                  *         **
         +-------+               *
         | train |             **
         +-------+            *
                  *         **
                 +----------+
                 | evaluate |
                 +----------+
```

### Um único Pipeline serializável

O artefato salvo recebe **dados crus** e devolve probabilidade. Não há pré-processamento manual a
repetir na inferência — é o que impede treino e produção de divergirem.

```python
Pipeline([
    ("features",       ConstrutorDeFeatures()),       # derivadas, stateless
    ("preprocessador", ColumnTransformer(...)),       # log1p+escala · escala · one-hot
    ("modelo",         RandomForestClassifier(...)),  # trocável por uma linha do params.yaml
])
```

Trocar de algoritmo é editar `train.algoritmo` no `params.yaml`; o `model.py` traduz a string em
estimador e o DVC re-executa só o que foi afetado.

### Onde cada decisão mora

| Tipo | Arquivo | Versionado? |
|---|---|---|
| Decisões de modelagem (seed, split, hiperparâmetros, limiar) | `params.yaml` | Git — e o DVC as rastreia |
| Configuração de ambiente (URIs, credenciais, log) | `.env` | **não** — só o `.env.example` |
| Constantes de domínio (grupos de colunas, caminhos) | `src/propensao/config.py` | Git, como código |

A regra: se mudar o valor **invalida o modelo**, vai para o `params.yaml` e o DVC re-executa o que
foi afetado.

---

## Reprodutibilidade

### O pipeline só refaz o que mudou

```
$ make repro
'data/raw/online_shoppers_intention.csv.dvc' didn't change, skipping
Stage 'prepare' didn't change, skipping
Stage 'train' didn't change, skipping
Stage 'evaluate' didn't change, skipping
Data and pipelines are up to date.
```

E o efeito de uma mudança é mensurável contra o commit anterior — aqui, a troca de campeão
(HistGB com limiar 0,45 → RandomForest com limiar 0,40):

```
$ poetry run dvc params diff HEAD~1
Path         Param            HEAD~1                  workspace
params.yaml  evaluate.limiar  0.45                    0.4
params.yaml  train.algoritmo  hist_gradient_boosting  random_forest

$ poetry run dvc metrics diff HEAD~1
Path                  Metric    HEAD~1     workspace    Change
metrics/metrics.json  f1        0.41361    0.43316      0.01955
metrics/metrics.json  pr_auc    0.35633    0.37409      0.01776
metrics/metrics.json  precisao  0.31429    0.32838      0.01409
metrics/metrics.json  recall    0.60471    0.63613      0.03141
metrics/metrics.json  roc_auc   0.77408    0.7906       0.01652
```

Quando muda **só** o `evaluate.limiar`, `prepare` e `train` são pulados e apenas o `evaluate`
re-executa — o limiar não toca no modelo, só na decisão. Nesse caso `pr_auc` e `roc_auc` não
aparecem no `metrics diff`, porque **independem de limiar**. É por isso que são elas que comparam
candidatos.

### Verificado num clone limpo do GitHub

O caminho documentado em [Começando](#começando) foi executado do zero — `git clone` → `make data`
→ `make repro` → `make test`:

```
16 passed in 0.67s

$ cat metrics/metrics.json
{
  "pr_auc": 0.3740913137621523,
  "roc_auc": 0.7906013441181481,
  "precisao": 0.32837837837837835,
  "recall": 0.6361256544502618,
  "f1": 0.43315508021390375
}
```

**Byte a byte idêntico** ao `metrics/metrics.json` commitado neste repositório.

Depois do `repro`, o `git status` do clone acusa **`M dvc.lock`** — e só isso. A única linha que
muda é o md5 de `models/mlflow_run.json`: cada execução abre um run novo no MLflow, e o id do run
entra no hash. O `models/pipeline.joblib` sai com o **mesmo** md5 — na mesma arquitetura, o treino
é determinístico, e é para isso que a `seed` do `params.yaml` existe.

### Provado em três ambientes

Mesmo commit, mesmo `dvc.lock`, retreinando do zero:

| Ambiente | HistGB — PR-AUC | RandomForest — PR-AUC |
|---|---|---|
| Ubuntu x86 (host) | 0,3563329947804378 | 0,3740913137621523 |
| Container Docker (x86) | **idêntico** | **idêntico** |
| macOS ARM (clone limpo) | **idêntico** | 0,3764050381711980 |

O container reproduz o host **até o último decimal** — é exatamente o que ele existe para fazer.

> ⚠️ **A honestidade que costuma faltar nessas tabelas.** O `RandomForest` **não** é reprodutível
> bit-a-bit entre **arquiteturas**: x86 dá 0,374091 e ARM dá 0,376405, com o mesmo commit e o mesmo
> `dvc.lock`. O `HistGradientBoosting` é. A causa não é o DVC nem o Docker — é a ordem das reduções
> em ponto flutuante das bibliotecas numéricas de cada arquitetura. O `Dockerfile` fixa bibliotecas
> e sistema operacional, **mas não a arquitetura**: fechar essa lacuna exigiria `platform:
> linux/amd64` no compose, ao custo de emulação no Mac. O custo foi medido, aceito e está
> registrado no `params.yaml`.

> ⚠️ **`dvc repro` sozinho num clone recém-`pull`ado não executa nada** — os hashes já batem e ele
> reporta "up to date". O teste real de reprodutibilidade exige `--force`, senão passa sem ter
> testado nada.

O `md5` do `pipeline.joblib` também diverge entre máquinas mesmo quando as métricas são idênticas:
o `joblib` grava dtypes, alinhamento de memória e compressão. **O pipeline reproduz o resultado,
não o binário** — propriedade do pickle, não limitação do DVC.

---

## Rastreamento de experimentos

Cada execução do estágio `train` abre um run no MLflow com os params do `params.yaml`, tags de
proveniência e o modelo com `signature` e `input_example`. O `evaluate` **retoma o mesmo run** para
anexar as métricas.

> Por que isso é menos trivial do que parece: os dois estágios rodam como **processos separados**.
> Sem uma ponte, params e métricas cairiam em runs distintos e a tabela de comparação seria inútil.
> O `train` grava `models/mlflow_run.json` — output dele, dependência do `evaluate`.

### Os quatro candidatos

```
algoritmo                  PR-AUC  ROC-AUC   limiar   precisão   recall       F1
--------------------------------------------------------------------------------
dummy                      0.1565   0.5000     0.45     0.0000   0.0000   0.0000
logreg                     0.3390   0.7640     0.45     0.2592   0.8272   0.3948
hist_gradient_boosting     0.3563   0.7741     0.45     0.3143   0.6047   0.4136
random_forest              0.3741   0.7906     0.40     0.3284   0.6361   0.4332
```

**O `dummy` valida o arranjo.** ROC-AUC exatamente 0,5000 e PR-AUC 0,1565 — a taxa base de
positivos. Se ele pontuasse bem, haveria vazamento em algum lugar.

**Compare pela PR-AUC e pela ROC-AUC.** Precisão, recall e F1 dependem do limiar, e o limiar é
recalibrado por candidato (0,45 para o HistGB, 0,40 para o RF) — as três últimas colunas não são
comparáveis entre linhas.

### Proveniência

Cada run leva as tags `git_commit` e `dvc_lock_hash`. Juntas respondem *"de qual código e de quais
dados saiu este modelo?"* — o commit identifica o código, o hash do `dvc.lock` identifica os dados e
os artefatos de cada estágio. As mesmas tags são gravadas **na versão do Registry**, para quem abre
o Registry não precisar caçar o run.

### Model Registry

```bash
make register                                                # registra e move o alias @champion
poetry run python -m propensao.register --alias challenger   # registra sem despromover
```

Promoção é **decisão**, não transformação determinística — por isso o `register.py` é uma CLI e fica
**fora** do `dvc.yaml`. Como estágio, todo `dvc repro` promoveria algo, inclusive um experimento
ruim. A decisão em si já está no Git: `train.algoritmo` no `params.yaml` diz qual candidato foi
escolhido, e o script só executa o que o repositório declara.

Consumo, independente de versão:

```python
modelo = mlflow.sklearn.load_model("models:/propensao-compra@champion")
```

O alias é uma indireção: promover um modelo novo é mover o ponteiro, sem tocar em código.

---

## Docker

```bash
docker compose run --rm train      # roda o pipeline dentro do container
```

O compose monta o projeto inteiro em `/app` porque o DVC precisa de `.git/` e `.dvc/` em tempo de
execução — que o `.dockerignore` deixa fora da imagem de propósito. O `user:` fixa uid/gid do host,
para os artefatos gerados no volume não saírem com dono `root`.

O `.env` **nunca entra na imagem**: chega pelo mount, em tempo de execução. A imagem publicada sai
limpa. As dependências são instaladas **antes** do código no `Dockerfile`, então editar um `.py` não
reinstala nada — só a última camada é refeita.

**Imagem: 2,2 GB**, single-stage. Medido com `docker history` e `du`: `pyarrow` 152 MB, `scipy` 110,
`numpy` 58, `pandas` 45, `mlflow` 44, `sklearn` 32 — quase tudo é biblioteca científica legítima.
Multi-stage economizaria ~150 MB (Poetry e cache do pip) e está registrado como melhoria pendente:
o critério é "Dockerfile configurado corretamente", e otimização de tamanho não é requisito.

---

## Qualidade

```bash
make test    # 16 testes
make lint    # ruff sobre src e tests
```

```
$ make test
16 passed in 0.68s
```

A suíte roda num **clone sem `dvc pull`** — é para isso que `tests/data/shoppers_sample.csv` existe
(300 sessões estratificadas, 28 KB, commitado). Nenhum teste exige rede ou credencial, então quem
avalia consegue rodar tudo.

Os testes travam garantias, não trivialidades:

| teste | o que quebraria sem ele |
|---|---|
| `PageValues` fora quando desligado | um `remainder` no default reintroduziria em silêncio a variável vazada |
| `ConstrutorDeFeatures` é stateless | vazamento entre treino e teste |
| `dump`/`load` preserva probabilidades | o modelo publicado divergir do avaliado |
| PR-AUC/ROC-AUC imunes ao limiar | a leitura da tabela de candidatos |
| categoria desconhecida não quebra | inferência caindo com um `Month` inédito |

Convenções de código: funções curtas com docstring Google-style, nomes em português (o domínio é
descrito em português), `ruff` com `line-length = 100` e ordenação de imports. Todo comentário no
repositório responde **por que**, não **o quê** — o "o quê" está no código.

---

## Decisões e o porquê de cada uma

### `PageValues` está fora do modelo — e custou metade do desempenho

A variável correlaciona 0,49 com o alvo. Sessões com `PageValues > 0` convertem a **56,3%**; com
zero, a **3,9%**.

Parece a melhor feature do dataset. É a pior.

A documentação a define como *"o valor médio de uma página que o usuário visitou **antes de
completar uma transação**"* — ela é construída a partir do desfecho que se quer prever. É
**near-target leakage**: não é o alvo entrando no treino por erro de split, é uma feature cuja
própria definição depende da conversão ter ocorrido.

**Custo de removê-la, medido no HistGB: PR-AUC 0,7390 → 0,3563.** Metade do poder preditivo.

Foi removida mesmo assim. Um modelo que aprende comportamento de navegação vale mais que um número
inflado por informação do futuro — e num TC cuja modelagem vale 10%, entregar 0,74 com vazamento
seria trocar rigor por aparência.

### O limiar de decisão é escolhido *out-of-fold*, e recalculado a cada troca de modelo

Por F1 máximo em predições *out-of-fold* (StratifiedKFold, 5 folds) sobre a **partição de treino**.
O hold-out nunca foi usado para escolher o limiar: ele existe para medir, não para calibrar.

| algoritmo | limiar | observação |
|---|---|---|
| `hist_gradient_boosting` | 0,45 | em amostra o ótimo aparecia em 0,65 — o HistGB memoriza o treino e chega a recall 1,000; fora de amostra cai para 0,45, com a curva plana entre 0,35 e 0,50 |
| `random_forest` | 0,40 | ótimo 0,41 na grade de 0,01; zona plana entre 0,39 e 0,43 |

O histórico de cada recalibração — com o valor, a grade varrida e a largura da zona plana — está
nos comentários de `evaluate.limiar`, no `params.yaml`.

**O limiar não viaja com o algoritmo.** Quando o RandomForest passou a liderar, promovê-lo exigiu
recalcular o corte antes: o 0,45 tinha sido calibrado para o HistGB, e aplicá-lo ao RF fazia o
candidato melhor parecer pior em F1. Trocar de modelo sem recalibrar o limiar é um erro silencioso —
as métricas de ranking melhoram e as de decisão pioram.

### O campeão é o `random_forest`, e a troca foi uma decisão consciente

Com o limiar recalibrado, o RF vence o HistGB em **todas** as métricas do hold-out (PR-AUC 0,3741
contra 0,3563). Custo conhecido e aceito: o RF **não** é reprodutível bit-a-bit entre arquiteturas,
enquanto o HistGB é — ver [Reprodutibilidade](#reprodutibilidade).

### O remote do DVC é externo, não local

A primeira versão usava um diretório local. **Não sobrevive a um clone.** O DVC resolve URL relativo
de remote a partir de `.dvc/`, não da raiz do repositório, então num clone em outro caminho ele
aponta para um diretório inexistente — comprovado num clone de teste. Trocado por um endpoint
S3-compatível no DagsHub, com `endpointurl` commitado no `.dvc/config` (é o que torna o remote
resolvível por quem clonar) e credenciais em `.dvc/config.local`, que fica fora do Git.

---

## Resultados

**Modelo:** `RandomForestClassifier` (400 árvores, `min_samples_leaf=5`,
`class_weight='balanced_subsample'`), limiar 0,40, `PageValues` fora.

| Métrica | Valor |
|---|---|
| PR-AUC | **0,3741** |
| ROC-AUC | 0,7906 |
| Precisão | 0,3284 |
| Recall | 0,6361 |
| F1 | 0,4332 |

**PR-AUC é a métrica principal.** Com 15,5% de conversão, acurácia é enganosa: prever "ninguém
compra" acerta 84,5% e não serve para nada. O baseline honesto é a taxa base, 0,1565 — o modelo
mais que dobra isso.

Em termos de negócio: a 0,40, o modelo captura **64% das sessões que convertem**, com 33% de
precisão. Para acionar uma campanha de retenção, esse é o trade-off que o limiar arbitra.

---

## Limitações

- **`PageValues` fora** limita o teto de desempenho. Consciente e defendido acima.
- **RandomForest não é bit-reproduzível entre arquiteturas.** Fechável com `platform: linux/amd64`.
- **Sem validação cruzada na seleção final** — hold-out único. Modelagem vale 10%; o orçamento foi
  para engenharia.
- **Sem tuning de hiperparâmetros.** Valores razoáveis no `params.yaml`.
- **Imagem de 2,2 GB.** Multi-stage pendente.
- **`dvc pull` exige token.** O `make data` contorna, baixando a fonte pública.
- **Sem API de inferência e sem deploy** — não estão nos critérios desta fase.

---

## Stack

Python 3.12 · scikit-learn 1.9 · pandas 2.3 · Poetry 2.3 · DVC 3.67 · MLflow 3.15 · Docker ·
pytest 8 · ruff

## Estrutura

```
├── src/propensao/        config · data · features · preprocess · model
│                         prepare · train · evaluate · register · tracking
├── tests/                16 testes + amostra commitada (roda sem dvc pull)
├── notebooks/            01-eda-baseline.ipynb
├── params.yaml           decisões de modelagem (rastreadas pelo DVC)
├── dvc.yaml / dvc.lock   o pipeline e os hashes que o provam
├── metrics/metrics.json  métricas do hold-out, versionadas no Git
├── .env.example          configuração de ambiente (o .env fica fora do Git)
├── Dockerfile            single-stage, não-root
├── docker-compose.yml    serviço `train`
└── Makefile              atalhos (`make` mostra a ajuda)
```
