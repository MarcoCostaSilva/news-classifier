"""
Testes unitários para as funções de split em news_classifier.split.
"""
import pandas as pd

from news_classifier.split import split_temporal, filtrar_categorias_com_teste_insuficiente


def test_split_temporal_separa_por_data():
    """Notícias antes da data de corte vão para treino, a partir dela para teste."""
    df = pd.DataFrame({
        "date": ["2017-01-01", "2017-03-01", "2017-05-01", "2017-06-01"],
        "category": ["esporte", "esporte", "mercado", "mercado"],
    })

    treino, teste = split_temporal(df, data_corte="2017-04-01")

    assert len(treino) == 2
    assert len(teste) == 2
    assert treino["date"].tolist() == ["2017-01-01", "2017-03-01"]
    assert teste["date"].tolist() == ["2017-05-01", "2017-06-01"]


def test_split_temporal_nao_perde_linhas():
    """A soma de treino + teste deve ser igual ao total original (nenhuma linha perdida)."""
    df = pd.DataFrame({
        "date": ["2017-01-01", "2017-02-01", "2017-03-01", "2017-04-01", "2017-05-01"],
        "category": ["a", "b", "c", "d", "e"],
    })

    treino, teste = split_temporal(df, data_corte="2017-03-15")

    assert len(treino) + len(teste) == len(df)


def test_filtrar_categorias_com_teste_insuficiente_remove_categoria_rara_no_teste():
    """Categorias com poucos exemplos no teste devem ser removidas de treino E teste,
    mesmo que tenham volume suficiente no treino."""
    df_treino = pd.DataFrame({
        "category": ["esporte"] * 10 + ["nicho"] * 10,  # 'nicho' tem bom volume no treino
    })
    df_teste = pd.DataFrame({
        "category": ["esporte"] * 5 + ["nicho"] * 1,  # mas quase nenhum exemplo no teste
    })

    treino_filtrado, teste_filtrado = filtrar_categorias_com_teste_insuficiente(
        df_treino, df_teste, limiar_minimo=3
    )

    assert "nicho" not in treino_filtrado["category"].values
    assert "nicho" not in teste_filtrado["category"].values
    assert "esporte" in treino_filtrado["category"].values
    assert "esporte" in teste_filtrado["category"].values


def test_filtrar_categorias_com_teste_insuficiente_mantem_categoria_com_volume_ok():
    """Categorias com volume suficiente no teste devem ser mantidas em ambos os conjuntos."""
    df_treino = pd.DataFrame({"category": ["esporte"] * 10})
    df_teste = pd.DataFrame({"category": ["esporte"] * 5})

    treino_filtrado, teste_filtrado = filtrar_categorias_com_teste_insuficiente(
        df_treino, df_teste, limiar_minimo=3
    )

    assert len(treino_filtrado) == 10
    assert len(teste_filtrado) == 5