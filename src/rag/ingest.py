"""Build the FAISS index from data/knowledge_base/*.md.

Run: python -m src.rag.ingest
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)

CHUNK_CHARS = 800
CHUNK_OVERLAP = 150


def _chunk(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_CHARS])
        start += CHUNK_CHARS - CHUNK_OVERLAP
    return chunks


def _load_articles(kb_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(kb_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        text = path.read_text()
        title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        for i, chunk in enumerate(_chunk(text)):
            docs.append(
                {"id": f"{path.stem}#{i}", "title": title, "file": path.name, "text": chunk}
            )
    return docs


def build_index(kb_dir: Path | None = None, out_dir: Path | None = None) -> int:
    """Embed every chunk and write index.faiss + chunks.json. Returns chunk count."""
    kb_dir = kb_dir or settings.knowledge_dir
    out_dir = out_dir or settings.index_dir
    docs = _load_articles(kb_dir)
    if not docs:
        raise RuntimeError(f"No articles found in {kb_dir}")

    model = SentenceTransformer(settings.embedding_model)
    vectors = model.encode([d["text"] for d in docs], normalize_embeddings=True)
    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product on unit vectors = cosine
    index.add(np.asarray(vectors, dtype="float32"))

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    (out_dir / "chunks.json").write_text(json.dumps(docs, indent=2))
    return len(docs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = build_index()
    print(f"Indexed {count} chunks into {settings.index_dir}")
