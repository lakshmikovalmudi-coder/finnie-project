"""Load the FAISS index once and return the top-k chunks for a query."""

import asyncio
import json
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.rag.ingest import build_index


@lru_cache(maxsize=1)
def _load() -> tuple[faiss.Index, list[dict], SentenceTransformer]:
    index_path = settings.index_dir / "index.faiss"
    if not index_path.exists():
        build_index()
    index = faiss.read_index(str(index_path))
    chunks = json.loads((settings.index_dir / "chunks.json").read_text())
    return index, chunks, SentenceTransformer(settings.embedding_model)


def search(query: str, k: int | None = None) -> list[dict]:
    """Return chunks as {title, file, text, score}, best first."""
    index, chunks, model = _load()
    vec = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(vec, dtype="float32"), k or settings.rag_top_k)
    return [
        {**chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0], strict=True) if i != -1
    ]


async def asearch(query: str, k: int | None = None) -> list[dict]:
    return await asyncio.to_thread(search, query, k)


def relevant(hits: list[dict], min_score: float | None = None) -> list[dict]:
    """Drop weak matches, so an off-topic chunk never poses as a source."""
    threshold = settings.rag_min_score if min_score is None else min_score
    return [h for h in hits if h["score"] >= threshold]


def format_context(hits: list[dict]) -> str:
    """Render hits as numbered context blocks for the LLM prompt."""
    return "\n\n".join(f"[{n}] {h['title']}\n{h['text']}" for n, h in enumerate(hits, 1))
