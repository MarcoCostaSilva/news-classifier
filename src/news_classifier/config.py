"""
Configurações centrais do projeto: paths, parâmetros de limpeza e modelagem.

Centralizar essas constantes aqui evita "números mágicos" espalhados pelo código e
torna explícitas as decisões tomadas durante a análise exploratória (ver notebooks/01_eda.ipynb).
"""
from pathlib import Path

# --- Paths ---
# Raiz do projeto, calculada de forma relativa a este arquivo (robusto independente
# de onde o código é executado: notebook, script, API, testes).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "articles.csv"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# --- Limpeza de dados (decisões da EDA, ver notebooks/01_eda.ipynb, seções 3-5) ---

# Valores de 'category' identificados como erro de parsing na coleta original,
# não são categorias editoriais reais.
CATEGORIAS_INVALIDAS = ["2016", "2015"]

# Número mínimo de exemplos por categoria para que ela seja mantida no treino.
# Categorias abaixo disso são excluídas (não agrupadas), pois são semanticamente
# heterogêneas entre si (ver EDA, seção 5, para a justificativa completa).
LIMIAR_MINIMO_CATEGORIA = 200

# --- Split treino/teste ---

# Split temporal: notícias até esta data vão para treino, a partir dela para teste.
# Escolhido para simular um cenário realista de produção e evitar vazamento de dados
# entre notícias similares publicadas em datas próximas (ver EDA, seção 7).
DATA_CORTE_TREINO_TESTE = "2017-04-01"

# --- Modelagem ---
RANDOM_STATE = 42

# Número mínimo de exemplos por categoria especificamente na janela de teste.
# Necessário porque o split temporal pode concentrar uma categoria no início do
# período, deixando poucos (ou nenhum) exemplos no teste mesmo que o volume total
# seja suficiente — o que inviabiliza métricas confiáveis para essa classe.
LIMIAR_MINIMO_TESTE = 30