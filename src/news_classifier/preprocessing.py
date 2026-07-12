"""
Funções de carregamento e limpeza de dados.

A lógica aqui reproduz, em código de produção, as decisões validadas na análise
exploratória (ver notebooks/01_eda.ipynb, seções 3-6).
"""
import pandas as pd

from news_classifier.config import (
    DATA_RAW_PATH,
    CATEGORIAS_INVALIDAS,
    LIMIAR_MINIMO_CATEGORIA,
)


def carregar_dados_brutos(path=DATA_RAW_PATH) -> pd.DataFrame:
    """Carrega o dataset bruto de notícias a partir do CSV original."""
    return pd.read_csv(path)


def remover_categorias_invalidas(df: pd.DataFrame) -> pd.DataFrame:
    """Remove registros cuja 'category' é um valor de erro de parsing (ex. '2015', '2016')."""
    return df[~df["category"].isin(CATEGORIAS_INVALIDAS)].copy()


def filtrar_categorias_raras(
    df: pd.DataFrame, limiar_minimo: int = LIMIAR_MINIMO_CATEGORIA
) -> pd.DataFrame:
    """
    Remove registros de categorias com menos de `limiar_minimo` exemplos.

    Categorias raras são excluídas (não agrupadas em uma classe "outras") porque são
    semanticamente heterogêneas entre si — agrupá-las criaria uma classe artificial sem
    padrão aprendível (ver EDA, seção 5, para a justificativa completa).
    """
    contagem = df["category"].value_counts()
    categorias_validas = contagem[contagem >= limiar_minimo].index
    return df[df["category"].isin(categorias_validas)].copy()


def combinar_titulo_e_texto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a coluna 'texto_completo', combinando título e texto como entrada do classificador.

    Valores nulos em 'text' são tratados como string vazia, já que o título sozinho
    (presente em 100% dos casos) garante que sempre há sinal disponível (ver EDA, seção 6).
    """
    df = df.copy()
    df["texto_completo"] = df["title"] + " " + df["text"].fillna("")
    return df


def preparar_dataset(path=DATA_RAW_PATH) -> pd.DataFrame:
    """
    Pipeline completo de carregamento e limpeza, aplicando em sequência todas as
    decisões validadas na EDA: remoção de categorias inválidas, filtro de categorias
    raras e combinação de título + texto.
    """
    df = carregar_dados_brutos(path)
    df = remover_categorias_invalidas(df)
    df = filtrar_categorias_raras(df)
    df = combinar_titulo_e_texto(df)
    return df