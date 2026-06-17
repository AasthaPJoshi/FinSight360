"""Centralised settings — reads from environment / .env file."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass


class _Settings:
    # ── AI / LLM ─────────────────────────────────────────────────────────
    @property
    def gemini_api_key(self) -> str:
        return os.environ.get("GEMINI_API_KEY", "")

    @property
    def openai_api_key(self) -> str:
        return os.environ.get("OPENAI_API_KEY", "")

    @property
    def llm_model(self) -> str:
        return os.environ.get("LLM_MODEL", "gpt-4o-mini")

    @property
    def embedding_model(self) -> str:
        return os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    @property
    def max_context_tokens(self) -> int:
        return int(os.environ.get("MAX_CONTEXT_TOKENS", "8000"))

    @property
    def chunk_size(self) -> int:
        return int(os.environ.get("CHUNK_SIZE", "1000"))

    @property
    def chunk_overlap(self) -> int:
        return int(os.environ.get("CHUNK_OVERLAP", "200"))

    # ── Vector store ─────────────────────────────────────────────────────
    @property
    def chroma_persist_dir(self) -> str:
        return os.environ.get("CHROMA_PERSIST_DIR", "data/chromadb")

    # ── Graph DB ─────────────────────────────────────────────────────────
    @property
    def neo4j_uri(self) -> str:
        return os.environ.get("NEO4J_URI", "bolt://localhost:7687")

    @property
    def neo4j_user(self) -> str:
        return os.environ.get("NEO4J_USER", "neo4j")

    @property
    def neo4j_password(self) -> str:
        return os.environ.get("NEO4J_PASSWORD", "finsight360")

    # ── Database ─────────────────────────────────────────────────────────
    @property
    def duckdb_path(self) -> str:
        return os.environ.get("DUCKDB_PATH", "data/finsight360.duckdb")

    # ── Ingestion ────────────────────────────────────────────────────────
    @property
    def edgar_user_agent(self) -> str:
        return os.environ.get(
            "EDGAR_USER_AGENT", "FinSight360 ajoshi8879@sdsu.edu"
        )

    @property
    def data_source(self) -> str:
        return os.environ.get("DATA_SOURCE", "edgar")

    # ── MLflow ───────────────────────────────────────────────────────────
    @property
    def mlflow_tracking_uri(self) -> str:
        return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")

    # ── Logging ──────────────────────────────────────────────────────────
    @property
    def log_level(self) -> str:
        return os.environ.get("LOG_LEVEL", "INFO")

    @property
    def log_format(self) -> str:
        return os.environ.get("LOG_FORMAT", "console")


settings = _Settings()
