"""
Modelos Pydantic para validação de entrada e formatação de saída da API.
"""
from pydantic import BaseModel, ConfigDict, Field


class NoticiaInput(BaseModel):
    """Dados de entrada para classificação de uma notícia."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Governo anuncia novas medidas econômicas para 2026",
                "text": "O ministério da economia anunciou hoje um pacote de medidas...",
            }
        }
    )

    title: str = Field(..., min_length=1, description="Título da notícia")
    text: str = Field(default="", description="Corpo/texto da notícia (opcional)")


class PredicaoOutput(BaseModel):
    """Resultado da classificação de uma notícia."""

    categoria_prevista: str = Field(..., description="Categoria com maior probabilidade")
    confianca: float = Field(..., description="Probabilidade da categoria prevista (0 a 1)")
    top_3_categorias: dict[str, float] = Field(
        ..., description="As 3 categorias mais prováveis, com suas probabilidades"
    )


class HealthOutput(BaseModel):
    """Status de saúde da API."""

    status: str
    modelo_carregado: bool