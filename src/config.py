"""Single source of configuration; the rest of the code reads `settings`, never os.getenv."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

load_dotenv(ROOT_DIR / ".env")


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    alpha_vantage_api_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    news_provider: str = os.getenv("NEWS_PROVIDER", "tavily").lower()
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    serpapi_api_key: str = os.getenv("SERPAPI_API_KEY", "")

    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "1800"))
    use_mock_data: bool = _bool("USE_MOCK_DATA")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    # Cosine score below which a chunk is treated as off-topic (on-topic hits measured ≥ 0.5).
    rag_min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.45"))
    router_min_confidence: float = float(os.getenv("ROUTER_MIN_CONFIDENCE", "0.5"))

    cache_dir: Path = DATA_DIR / "cache"
    mock_dir: Path = DATA_DIR / "mock"
    knowledge_dir: Path = DATA_DIR / "knowledge_base"
    index_dir: Path = DATA_DIR / "faiss_index"


settings = Settings()
