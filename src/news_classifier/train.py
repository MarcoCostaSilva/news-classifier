"""
Treinamento do classificador de categorias de notícias.

Pipeline: TF-IDF (vetorização de texto) + Regressão Logística (classificação).
Essa combinação foi escolhida em detrimento de modelos baseados em Transformers
(BERTimbau, etc.) por oferecer o melhor custo-benefício para este projeto: treino
rápido, sem dependência de GPU, interpretável, e com desempenho historicamente forte
em classificação de texto por categoria editorial (vocabulário bem discriminativo
entre classes como "mercado" e "esporte").
"""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from news_classifier.config import MODELS_DIR, RANDOM_STATE
from news_classifier.preprocessing import preparar_dataset
from news_classifier.split import preparar_split

# Stopwords em português (evita dependência do NLTK/spaCy só para esta lista simples;
# TfidfVectorizer aceita uma lista de palavras diretamente).
STOPWORDS_PT = [
    "a", "o", "e", "de", "da", "do", "das", "dos", "em", "um", "uma", "para", "com",
    "não", "por", "mais", "as", "os", "se", "na", "no", "que", "é", "ao", "aos", "às",
    "como", "mas", "foi", "ao", "ele", "ela", "eles", "elas", "seu", "sua", "seus",
    "suas", "ou", "já", "também", "só", "pelo", "pela", "até", "isso", "num", "numa",
    "nos", "nas", "esse", "essa", "esses", "essas", "este", "esta", "estes", "estas",
]


def construir_pipeline() -> Pipeline:
    """
    Monta o pipeline de vetorização TF-IDF + classificador de Regressão Logística.

    Versão com vocabulário completo (max_features=20_000, bigramas incluídos),
    usada como comparação de referência contra a versão otimizada para baixa
    memória (ver histórico do projeto / README para a comparação de resultados).
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            stop_words=STOPWORDS_PT,
            min_df=3,
            max_df=0.9,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def treinar_modelo():
    """Executa o pipeline completo: carrega dados, treina e salva o modelo."""
    print("Carregando e preparando dados...")
    df = preparar_dataset()
    df_treino, df_teste = preparar_split(df)

    print(f"Treino: {df_treino.shape[0]} exemplos | Teste: {df_teste.shape[0]} exemplos")

    X_treino, y_treino = df_treino["texto_completo"], df_treino["category"]
    X_teste, y_teste = df_teste["texto_completo"], df_teste["category"]

    print("Treinando pipeline (TF-IDF + Regressão Logística) - versão completa...")
    pipeline = construir_pipeline()
    pipeline.fit(X_treino, y_treino)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    caminho_modelo = MODELS_DIR / "model_v2_full.joblib"
    joblib.dump(pipeline, caminho_modelo)
    print(f"Modelo salvo em: {caminho_modelo}")

    return pipeline, X_teste, y_teste


if __name__ == "__main__":
    treinar_modelo()