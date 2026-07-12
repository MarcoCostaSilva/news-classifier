"""
Testes de integração para os endpoints da API (api.main).

Uso o TestClient do FastAPI como context manager (`with TestClient(app) as client`),
o que garante que os eventos de lifespan (carregamento do modelo) sejam executados
antes dos testes rodarem, sem isso, o modelo não seria carregado em memória.
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """Cliente de teste que dispara o lifespan da aplicação (carrega o modelo)."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_retorna_status_ok(client):
    """O endpoint /health deve responder 200 e indicar que o modelo está carregado."""
    resposta = client.get("/health")

    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["status"] == "ok"
    assert dados["modelo_carregado"] is True


def test_predict_com_input_valido_retorna_categoria(client):
    """O endpoint /predict deve classificar uma notícia válida e retornar os campos esperados."""
    payload = {
        "title": "Time vence o campeonato após final emocionante",
        "text": "O time venceu o jogo decisivo do campeonato nacional neste domingo.",
    }

    resposta = client.post("/predict", json=payload)

    assert resposta.status_code == 200
    dados = resposta.json()
    assert "categoria_prevista" in dados
    assert "confianca" in dados
    assert "top_3_categorias" in dados
    assert 0 <= dados["confianca"] <= 1
    assert len(dados["top_3_categorias"]) == 3


def test_predict_com_titulo_vazio_retorna_erro_422(client):
    """O endpoint /predict deve rejeitar título vazio com erro de validação (422)."""
    payload = {"title": "", "text": "Algum texto"}

    resposta = client.post("/predict", json=payload)

    assert resposta.status_code == 422


def test_predict_sem_texto_ainda_funciona(client):
    """O endpoint /predict deve funcionar mesmo sem o campo 'text' (usa apenas o título)."""
    payload = {"title": "Governo anuncia novo pacote econômico"}

    resposta = client.post("/predict", json=payload)

    assert resposta.status_code == 200