from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "GraphRAG OCR Research Workbench"
    app_env: str = "development"
    data_dir: Path = Path("./data")
    database_url: str = "sqlite:///./backend/ocr_graphrag.db"
    max_upload_mb: int = 25
    confidence_threshold: float = 0.90
    candidate_count: int = 5
    ocr_engine: str = "auto"
    paddle_lang: str = "en"
    paddle_ocr_version: str = "PP-OCRv5"
    paddle_device: str = "cpu"
    embedding_model: str = "BAAI/bge-m3"
    embedding_gguf_model_path: Path | None = None
    retrieval_top_k: int = 3
    chunk_size: int = 80
    doc_layout_model_path: Path | None = None
    doc_layout_device: str = "cpu"
    doc_layout_confidence: float = 0.20
    neo4j_uri: str | None = None
    neo4j_user: str = "neo4j"
    neo4j_password: str | None = None
    slm_model_path: str | None = None
    llama_server_executable: Path | None = None
    llama_server_url: str = "http://127.0.0.1:8091"
    slm_gpu_layers: int = 99
    slm_context_size: int = 2048
    slm_max_new_tokens: int = 120

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def ensure_directories(self) -> None:
        for name in ("input", "processed", "output", "embeddings", "graphs"):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    value = Settings()
    value.ensure_directories()
    return value
