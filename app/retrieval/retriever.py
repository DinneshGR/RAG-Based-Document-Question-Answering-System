"""
Retrieval module.

Exposes a single `retrieve` function that supports both plain semantic
similarity search and MMR (Maximum Marginal Relevance), using the
configured TOP_K / FETCH_K / LAMBDA_MULT values by default.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal

from app.config import settings
from app.exceptions import EmptyQuestionError, NoRelevantContextError
from app.vectorstore.chroma_store import mmr_search, similarity_search

logger = logging.getLogger(__name__)

RetrievalMode = Literal["similarity", "mmr"]


def retrieve(
    question: str,
    mode: RetrievalMode = "mmr",
    top_k: int | None = None,
    fetch_k: int | None = None,
    lambda_mult: float | None = None,
) -> List[Dict[str, Any]]:
    """Retrieve the most relevant chunks for a question.

    Returns a list of dicts with keys: text, document_name, document_id,
    page_number, chunk_index, similarity.
    """
    if not question or not question.strip():
        raise EmptyQuestionError("Question must not be empty.")

    top_k = top_k or settings.TOP_K
    fetch_k = fetch_k or settings.FETCH_K
    lambda_mult = lambda_mult if lambda_mult is not None else settings.LAMBDA_MULT

    if mode == "mmr":
        hits = mmr_search(question, top_k=top_k, fetch_k=fetch_k, lambda_mult=lambda_mult)
    else:
        hits = similarity_search(question, top_k=top_k)

    logger.info("Retrieved %d chunks for question (mode=%s)", len(hits), mode)

    if not hits:
        raise NoRelevantContextError(
            "No relevant context found. Please upload documents before asking questions."
        )
    return hits
