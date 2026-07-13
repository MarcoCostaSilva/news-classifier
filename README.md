# News Classifier API

Classificador de categorias de notícias da Folha de S.Paulo, desenvolvido como teste técnico para a posição de Cientista de Dados Jr. — AeC Centro de Contatos.

![CI](https://github.com/MarcoCostaSilva/news-classifier/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Sumário

- [Visão geral](#visão-geral)
- [Decisões de projeto](#decisões-de-projeto)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como rodar localmente](#como-rodar-localmente)
- [Como rodar com Docker](#como-rodar-com-docker)
- [Uso da API](#uso-da-api)
- [Pipeline de dados e modelagem](#pipeline-de-dados-e-modelagem)
- [Resultados e avaliação](#resultados-e-avaliação)
- [Testes automatizados](#testes-automatizados)
- [Integração contínua (CI)](#integração-contínua-ci)
- [Notebooks](#notebooks)
- [Limitações conhecidas e trabalho futuro](#limitações-conhecidas-e-trabalho-futuro)

## Visão geral

O projeto recebe notícias da Folha de S.Paulo (título e texto) e prevê a categoria editorial correspondente (ex. `esporte`, `mercado`, `poder`), usando um pipeline de **TF-IDF + Regressão Logística**, disponibilizado via uma **API REST (FastAPI)**.

O dataset utilizado é o [News of the Brazilian Newspaper](https://www.kaggle.com/datasets/marlesson/news-of-the-site-folhauol) (Kaggle), com 167.053 notícias coletadas entre 2015 e 2017.

## Decisões de projeto

O enunciado do teste prioriza explicitamente: **1) clareza de código, 2) entrega funcional, 3) modelagem, 4) EDA** — nessa ordem. Todas as decisões abaixo foram tomadas com essa priorização em mente.

### Modelagem: TF-IDF + Regressão Logística, não Transformers

Avaliamos formalmente a alternativa de embeddings semânticos pré-treinados (`sentence-transformers`) contra o TF-IDF tradicional (ver [`notebooks/02_experimento_transformers.ipynb`](notebooks/02_experimento_transformers.ipynb)). Numa comparação direta na mesma amostra de teste, o TF-IDF superou os embeddings pré-treinados (F1 macro 0,73 vs. 0,43), além de ser mais simples de produtizar, mais rápido de treinar e **interpretável** — dá para auditar exatamente quais palavras o modelo associa a cada categoria (ver [`notebooks/03_avaliacao_e_interpretabilidade.ipynb`](notebooks/03_avaliacao_e_interpretabilidade.ipynb)).

### Alvo do classificador: `category`, não `subcategory`

A coluna `subcategory` tem 82% de valores ausentes no dataset original — insuficiente para servir como alvo confiável de um modelo supervisionado.

### Limpeza de categorias raras (dois critérios)

1. Categorias com menos de 200 exemplos no total foram excluídas (não agrupadas em "outras", por serem semanticamente heterogêneas entre si).
2. Após o split temporal, identificamos que algumas categorias ficavam com poucos exemplos especificamente na janela de teste, mesmo tendo volume total suficiente — adicionamos um segundo filtro (mínimo de 30 exemplos no teste) para garantir métricas de avaliação confiáveis.

O dataset final de modelagem tem **22 categorias** e **162.860 registros** (99% do volume original).

### Split temporal, não aleatório

Treino com notícias até 2017-04-01, teste com o período seguinte (até 2017-10-01). Isso simula um cenário realista de produção (o modelo é avaliado em notícias "futuras" que nunca viu) e evita vazamento de dados entre notícias republicadas/atualizadas em datas próximas.

### Feature de entrada: título + texto combinados

O título sozinho é altamente discriminativo, mas curto (~10 palavras em média); o texto sozinho é mais rico, mas 0,46% das notícias não têm texto preenchido. A combinação aproveita o melhor dos dois e garante que sempre há sinal disponível (título está presente em 100% dos casos).

Toda a análise exploratória que fundamenta essas decisões está documentada em [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

## Estrutura do repositório

```
news-classifier/
├── api/                        # Aplicação FastAPI
│   ├── main.py                 # Endpoints e ciclo de vida da API
│   └── schemas.py              # Modelos Pydantic de entrada/saída
├── src/news_classifier/        # Código de produção (pipeline de dados e modelagem)
│   ├── config.py                # Constantes e decisões de projeto centralizadas
│   ├── preprocessing.py         # Limpeza e preparação dos dados
│   ├── split.py                 # Divisão treino/teste (temporal)
│   ├── train.py                 # Treinamento do modelo
│   └── evaluate.py              # Avaliação (F1 macro/weighted, relatório por classe)
├── notebooks/
│   ├── 01_eda.ipynb                          # Análise exploratória e decisões de limpeza
│   ├── 02_experimento_transformers.ipynb     # Comparação TF-IDF vs. Sentence Embeddings
│   └── 03_avaliacao_e_interpretabilidade.ipynb  # Matriz de confusão e interpretabilidade
├── tests/                      # Testes automatizados (pytest)
├── models/                     # Modelo treinado serializado (model_v1.joblib)
├── data/raw/                   # Dataset bruto (não versionado, ver instruções abaixo)
├── reports/figures/            # Gráficos gerados pela EDA e avaliação
├── Dockerfile / .dockerignore  # Containerização da API
├── .github/workflows/ci.yml    # Pipeline de CI (GitHub Actions)
├── requirements.txt
└── pyproject.toml
```

## Como rodar localmente

### Pré-requisitos

- Python 3.11+
- O dataset bruto: baixe em [Kaggle - News of the Brazilian Newspaper](https://www.kaggle.com/datasets/marlesson/news-of-the-site-folhauol) e coloque o arquivo `articles.csv` em `data/raw/articles.csv`.

### Setup

```bash
# Clone o repositório
git clone https://github.com/MarcoCostaSilva/news-classifier.git
cd news-classifier

# Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\Activate.ps1  # Windows PowerShell

# Instale as dependências e o pacote em modo editável
pip install -r requirements.txt
pip install -e .
```

### Treinar o modelo (opcional — o repositório já inclui `models/model_v1.joblib` treinado)

```bash
python -m news_classifier.train
```

### Avaliar o modelo

```bash
python -m news_classifier.evaluate
```

### Rodar a API

```bash
uvicorn api.main:app --reload
```

Acesse a documentação interativa (Swagger) em **http://127.0.0.1:8000/docs**.

## Como rodar com Docker

```bash
docker build -t news-classifier-api .
docker run -d -p 8000:8000 --name news-classifier-container news-classifier-api
```

A API estará disponível em **http://localhost:8000/docs**.

> **Nota**: a imagem atual inclui as dependências do notebook de experimento (`sentence-transformers`/`torch`), o que a torna maior do que estritamente necessário para servir a API. Uma versão otimizada (`requirements-docker.txt` enxuto, sem essas dependências) é um refinamento planejado como próximo passo.

## Uso da API

### `GET /health`

Verifica se a API está no ar e se o modelo foi carregado com sucesso.

```json
{"status": "ok", "modelo_carregado": true}
```

### `POST /predict`

Classifica uma notícia em uma das 22 categorias editoriais treinadas.

**Request:**
```json
{
  "title": "Governo anuncia novas medidas econômicas para 2026",
  "text": "O ministério da economia anunciou hoje um pacote de medidas..."
}
```

**Response:**
```json
{
  "categoria_prevista": "mercado",
  "confianca": 0.87,
  "top_3_categorias": {
    "mercado": 0.87,
    "poder": 0.06,
    "cotidiano": 0.03
  }
}
```

O campo `text` é opcional — a API funciona apenas com `title`. Entradas inválidas (título vazio) retornam erro 422 com detalhamento.

## Pipeline de dados e modelagem

1. **Carregamento e limpeza** (`preprocessing.py`): remoção de categorias inválidas, filtro de categorias raras, combinação de título + texto.
2. **Split temporal** (`split.py`): divisão treino/teste por data de corte, com filtro adicional de volume mínimo por categoria no teste.
3. **Vetorização**: TF-IDF (unigramas, 20.000 features, `min_df=3`, `max_df=0.9`).
4. **Classificação**: Regressão Logística com `class_weight="balanced"` (compensa o desbalanceamento remanescente entre categorias).
5. **Serialização**: pipeline completo (vetorizador + classificador) salvo como um único artefato `joblib`, consumido diretamente pela API.

## Resultados e avaliação

Avaliação no conjunto de teste (22.911 exemplos, split temporal):

| Métrica | Valor |
|---|---|
| F1 macro | 0,696 |
| F1 weighted | 0,825 |
| Acurácia | 0,82 |

Usamos F1 macro e F1 weighted como métricas principais — não acurácia isolada — porque o dataset é desbalanceado entre as 22 categorias; acurácia isolada seria dominada pelas classes majoritárias e esconderia o desempenho real nas categorias menores.

**Análise de erro**: a matriz de confusão (ver [`notebooks/03_avaliacao_e_interpretabilidade.ipynb`](notebooks/03_avaliacao_e_interpretabilidade.ipynb)) mostra que os principais pontos de confusão do modelo são entre categorias editorialmente próximas (`serafina` ↔ `ilustrada`, ambas de comportamento/estilo), e não erros aleatórios — um sinal de que os erros refletem sobreposição semântica real entre editorias, não falhas de implementação.

**Interpretabilidade**: por ser um modelo linear, é possível extrair as palavras mais associadas a cada categoria (ex. "clube", "jogador", "estádio" → `esporte`), confirmando que o modelo aprendeu padrões semanticamente coerentes.

## Testes automatizados

12 testes cobrindo pré-processamento, split e API:

```bash
pytest tests/ -v
```

- `tests/test_preprocessing.py`: regras de limpeza de dados.
- `tests/test_split.py`: lógica de split temporal e filtro de volume mínimo no teste.
- `tests/test_api.py`: endpoints `/health` e `/predict`, incluindo validação de erro (título vazio).

## Integração contínua (CI)

Todo push/pull request na branch `main` dispara automaticamente a suíte de testes em um ambiente Linux limpo, via GitHub Actions (ver [`.github/workflows/ci.yml`](.github/workflows/ci.yml)) — garantindo que o projeto é reprodutível fora do ambiente de desenvolvimento original.

## Notebooks

| Notebook | Conteúdo |
|---|---|
| `01_eda.ipynb` | Análise exploratória completa: distribuição de categorias, qualidade dos dados, decisões de limpeza justificadas por evidência quantitativa. |
| `02_experimento_transformers.ipynb` | Comparação empírica entre TF-IDF e Sentence Embeddings pré-treinados, justificando a escolha do modelo de produção. |
| `03_avaliacao_e_interpretabilidade.ipynb` | Matriz de confusão e análise de interpretabilidade do modelo final. |

## Limitações conhecidas e trabalho futuro

- **Imagem Docker não otimizada**: inclui dependências do notebook de experimento (torch), desnecessárias para a API em produção — planejado enxugamento com `requirements-docker.txt` dedicado.
- **`seminariosfolha` e categorias de baixo desempenho**: algumas categorias (ex. `serafina`, `seminariosfolha`) têm F1 mais baixo, refletindo sobreposição editorial genuína — não foram forçadas melhorias artificiais nessas classes para não distorcer o comportamento real do modelo.
- **Sem fine-tuning de Transformers**: o experimento com embeddings usou um modelo pré-treinado sem ajuste fino no domínio jornalístico; um fine-tuning completo (com volume de dados proporcional) é uma direção válida de trabalho futuro, mas fora do escopo deste teste.
- **Deriva temporal do vocabulário**: o modelo foi treinado com dados de 2015-2017; a análise de interpretabilidade identificou termos ligados a eventos específicos do período (ex. "caminhoneiros", relativo à greve de 2018) — em um cenário de produção real, seria necessário retreinamento periódico para manter relevância.

## Autor

Marco Costa Silva — [GitHub](https://github.com/MarcoCostaSilva)

## Licença

Este projeto está licenciado sob a licença MIT — ver [LICENSE](LICENSE) para detalhes.
