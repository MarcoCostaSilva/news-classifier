"""
API de classificação de categorias de notícias.

Expõe um endpoint /predict que recebe título e texto de uma notícia e retorna a
categoria prevista pelo modelo treinado (TF-IDF + Regressão Logística), além da
confiança da predição e as 3 categorias mais prováveis.
"""
import logging
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException

from news_classifier.config import MODELS_DIR
from api.schemas import NoticiaInput, PredicaoOutput, HealthOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Estado global simples para armazenar o modelo carregado em memória.
modelo_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo treinado em memória na inicialização da API (forma moderna,
    substitui o antigo @app.on_event("startup"), que está depreciado no FastAPI)."""
    global modelo_pipeline
    caminho_modelo = MODELS_DIR / "model_v1.joblib"
    try:
        modelo_pipeline = joblib.load(caminho_modelo)
        logger.info(f"Modelo carregado com sucesso de {caminho_modelo}")
    except FileNotFoundError:
        logger.error(f"Modelo não encontrado em {caminho_modelo}")
        modelo_pipeline = None
    yield
    # Nenhuma limpeza necessária no shutdown por enquanto.


app = FastAPI(
    title="News Classifier API",
    description="Classifica notícias da Folha de S.Paulo em categorias editoriais.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthOutput)
def health():
    """Verifica se a API está no ar e se o modelo foi carregado com sucesso."""
    return HealthOutput(
        status="ok",
        modelo_carregado=modelo_pipeline is not None,
    )


@app.post("/predict", response_model=PredicaoOutput)
def predict(noticia: NoticiaInput):
    """Classifica uma notícia em uma das categorias editoriais treinadas."""
    if modelo_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não está carregado. Verifique os logs da API.",
        )

    texto_completo = f"{noticia.title} {noticia.text}".strip()

    if not texto_completo:
        raise HTTPException(
            status_code=422,
            detail="O texto combinado de título e corpo não pode estar vazio.",
        )

    try:
        probabilidades = modelo_pipeline.predict_proba([texto_completo])[0]
        classes = modelo_pipeline.classes_

        pares_ordenados = sorted(zip(classes, probabilidades), key=lambda x: x[1], reverse=True)

        categoria_prevista, confianca = pares_ordenados[0]
        top_3 = {categoria: float(prob) for categoria, prob in pares_ordenados[:3]}

        return PredicaoOutput(
            categoria_prevista=categoria_prevista,
            confianca=float(confianca),
            top_3_categorias=top_3,
        )
    except Exception as e:
        logger.error(f"Erro ao gerar predição: {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao processar a predição.")