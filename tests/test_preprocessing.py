"""
Testes unitários para as funções de limpeza de dados em news_classifier.preprocessing.

Usamos dataframes fictícios pequenos (não o dataset real) para testar cada regra de
limpeza de forma isolada e rápida.
"""
import pandas as pd

from news_classifier.preprocessing import (
    remover_categorias_invalidas,
    filtrar_categorias_raras,
    combinar_titulo_e_texto,
)


def test_remover_categorias_invalidas():
    """Categorias '2015' e '2016' (erro de parsing) devem ser removidas."""
    df = pd.DataFrame({
        "category": ["esporte", "2015", "mercado", "2016"],
        "title": ["a", "b", "c", "d"],
    })

    resultado = remover_categorias_invalidas(df)

    assert len(resultado) == 2
    assert "2015" not in resultado["category"].values
    assert "2016" not in resultado["category"].values


def test_filtrar_categorias_raras():
    """Categorias com menos exemplos que o limiar mínimo devem ser removidas."""
    df = pd.DataFrame({
        "category": ["esporte"] * 5 + ["nicho"] * 2,
    })

    resultado = filtrar_categorias_raras(df, limiar_minimo=3)

    assert "esporte" in resultado["category"].values
    assert "nicho" not in resultado["category"].values
    assert len(resultado) == 5


def test_combinar_titulo_e_texto():
    """A coluna 'texto_completo' deve combinar título e texto, tratando nulos em 'text'."""
    df = pd.DataFrame({
        "title": ["Título A", "Título B"],
        "text": ["Corpo da notícia A", None],
    })

    resultado = combinar_titulo_e_texto(df)

    assert resultado.loc[0, "texto_completo"] == "Título A Corpo da notícia A"
    assert resultado.loc[1, "texto_completo"] == "Título B "  # texto nulo vira string vazia


def test_combinar_titulo_e_texto_preserva_colunas_originais():
    """A função não deve remover nenhuma coluna original, apenas adicionar 'texto_completo'."""
    df = pd.DataFrame({"title": ["A"], "text": ["B"]})

    resultado = combinar_titulo_e_texto(df)

    assert "title" in resultado.columns
    assert "text" in resultado.columns
    assert "texto_completo" in resultado.columns