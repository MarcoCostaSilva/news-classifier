"""
Avaliação do classificador de categorias de notícias.

Usamos F1 macro e F1 weighted como métricas principais (não acurácia), pois o
dataset é desbalanceado entre as 22 categorias finais: acurácia seria dominada
pelas classes majoritárias (ex. "poder", "colunas", "mercado") e esconderia o
desempenho real em categorias menores.
"""
import sys

import joblib
from sklearn.metrics import classification_report, f1_score

from news_classifier.config import MODELS_DIR
from news_classifier.preprocessing import preparar_dataset
from news_classifier.split import preparar_split


def avaliar_modelo(caminho_modelo=None):
    """Carrega o modelo salvo, avalia no conjunto de teste e imprime métricas."""
    if caminho_modelo is None:
        caminho_modelo = MODELS_DIR / "model_v1.joblib"

    print(f"Carregando modelo: {caminho_modelo}")
    pipeline = joblib.load(caminho_modelo)

    print("Preparando dados de teste...")
    df = preparar_dataset()
    _, df_teste = preparar_split(df)
    X_teste, y_teste = df_teste["texto_completo"], df_teste["category"]

    print("Gerando predições...")
    y_pred = pipeline.predict(X_teste)

    f1_macro = f1_score(y_teste, y_pred, average="macro")
    f1_weighted = f1_score(y_teste, y_pred, average="weighted")

    print(f"\nF1 macro:    {f1_macro:.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}")
    print("\nRelatório por classe:")
    print(classification_report(y_teste, y_pred, zero_division=0))

    return y_teste, y_pred


if __name__ == "__main__":
    # Permite escolher o modelo pela linha de comando:
    # python -m news_classifier.evaluate models/model_v1_leve.joblib
    caminho = sys.argv[1] if len(sys.argv) > 1 else MODELS_DIR / "model_v1.joblib"
    avaliar_modelo(caminho)