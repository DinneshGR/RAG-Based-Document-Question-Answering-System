"""
Local embedding module.

Wraps a Sentence Transformers model so the rest of the application never
talks to the model directly. Swapping the embedding model only requires
changing EMBEDDING_MODEL in the environment/config - no other code
changes needed.
"""
from __future__ import annotations

import logging
import threading
from typing import List

from sentence_transformers import SentenceTransformer

from app.config import settings
from app.exceptions import EmbeddingModelError

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Lazily load and cache the Sentence Transformers model (singleton)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # double-checked locking
                try:
                    logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
                    _model = SentenceTransformer(settings.EMBEDDING_MODEL)
                except Exception as exc:
                    raise EmbeddingModelError(
                        f"Failed to load embedding model '{settings.EMBEDDING_MODEL}': {exc}"
                    ) from exc
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts into dense vectors."""
    if not texts:
        return []
    try:
        model = get_embedding_model()
        vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return vectors.tolist()
    except EmbeddingModelError:
        raise
    except Exception as exc:
        raise EmbeddingModelError(f"Failed to generate embeddings: {exc}") from exc


def embed_query(text: str) -> List[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]


class LocalEmbeddingFunction:
    """Adapter exposing the embedding model in ChromaDB's expected interface."""

    def name(self) -> str:  # required by newer chromadb versions
        return f"local-sentence-transformers-{settings.EMBEDDING_MODEL}"

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        return embed_texts(list(input))
