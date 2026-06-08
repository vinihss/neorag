from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class EmbedderConfig(BaseModel):
    type: Literal["bge-m3", "sentence-transformer"] = "bge-m3"
    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"
    normalize: bool = True


class QdrantConfig(BaseModel):
    host: str = "qdrant"
    port: int = Field(default=6333, ge=1, le=65535)
    prefer_grpc: bool = True
    collection_name: str = "documents"


class LLMConfig(BaseModel):
    type: Literal["ollama", "openai-compatible"] = "ollama"
    base_url: str = "http://ollama:11434"
    model: str = "llama3.2:8b"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)


class RetrievalConfig(BaseModel):
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k_retrieve: int = Field(default=20, ge=1)
    top_k_rerank: int = Field(default=5, ge=1)
    min_score: float = Field(default=0.6, ge=0.0, le=1.0)


class SessionConfig(BaseModel):
    storage: Literal["sqlite", "redis"] = "sqlite"
    db_path: str = "data/sessions.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
