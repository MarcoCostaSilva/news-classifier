"""
Divisão dos dados em treino e teste, usando corte temporal.

Um split temporal (em vez de aleatório) simula um cenário realista de produção: o
modelo é treinado com dados do "passado" e avaliado em notícias "futuras" que nunca
viu, evitando vazamento de dados entre notícias similares publicadas em datas
próximas (ver notebooks/01_eda.ipynb, seção 7, para a justificativa completa).
"""
import pandas as pd

from news_classifier.config import DATA_CORTE_TREINO_TESTE, LIMIAR_MINIMO_TESTE


def split_temporal(df: pd.DataFrame, data_corte: str = DATA_CORTE_TREINO_TESTE):
    """Divide o dataframe em treino e teste com base em uma data de corte."""
    datas = pd.to_datetime(df["date"])
    corte = pd.to_datetime(data_corte)

    df_treino = df[datas < corte].copy()
    df_teste = df[datas >= corte].copy()

    return df_treino, df_teste


def filtrar_categorias_com_teste_insuficiente(
    df_treino: pd.DataFrame, df_teste: pd.DataFrame, limiar_minimo: int = LIMIAR_MINIMO_TESTE
):
    """
    Remove, de treino e teste, categorias com poucos exemplos na janela de teste.

    Categorias abaixo do limiar aqui têm métricas de avaliação não-confiáveis (poucos
    exemplos derrubam precision/recall de forma instável) e por isso são removidas de
    ambos os conjuntos, mantendo consistência entre as classes usadas em treino e teste.
    """
    contagem_teste = df_teste["category"].value_counts()
    categorias_validas = contagem_teste[contagem_teste >= limiar_minimo].index

    df_treino_filtrado = df_treino[df_treino["category"].isin(categorias_validas)].copy()
    df_teste_filtrado = df_teste[df_teste["category"].isin(categorias_validas)].copy()

    return df_treino_filtrado, df_teste_filtrado


def preparar_split(df: pd.DataFrame, data_corte: str = DATA_CORTE_TREINO_TESTE):
    """Pipeline completo: split temporal seguido de filtro de categorias com teste insuficiente."""
    df_treino, df_teste = split_temporal(df, data_corte)
    df_treino, df_teste = filtrar_categorias_com_teste_insuficiente(df_treino, df_teste)
    return df_treino, df_teste 