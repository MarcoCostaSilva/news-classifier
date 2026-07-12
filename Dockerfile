# Imagem base enxuta com Python 3.12 
FROM python:3.12-slim

WORKDIR /app

# Copia primeiro os arquivos de dependência (aproveita cache de camadas do Docker:
# se o código mudar mas as dependências não, essa camada não é reconstruída)
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código
COPY src/ src/
COPY api/ api/
COPY models/ models/

# Instala o pacote local (news_classifier) em modo editável
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]