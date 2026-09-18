"""
ChromaDB vector store wrapper.

Provides a small, well-defined interface over ChromaDB: add chunks,
similarity search, MMR search, list documents, delete a document, and
duplicate detection via content hash. Keeping this logic in one module
means the rest of the app never imports chromadb directly.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.embeddings.embedding_model import LocalEmbeddingFunction, embed_query
from app.exceptions import VectorStoreError
from app.ingestion.chunking import Chunk

logger = logging.getLogger(__name__)

_client_lock = threading.Lock()
_client = None
_collection = None


def _get_collection():
    """Lazily create (or fetch) the persistent Chroma collection (singleton)."""
    global _client, _collection
    if _collection is None:
        with _client_lock:
            if _collection is None:
                try:
                    _client = chromadb.PersistentClient(
                        path=str(settings.CHROMA_DIR),
                        settings=ChromaSettings(anonymized_telemetry=False),
                    )
                    _collection = _client.get_or_create_collection(
                        name=settings.CHROMA_COLLECTION,
                        embedding_function=LocalEmbeddingFunction(),
                        metadata={"hnsw:space": "cosine"},
                    )
                except Exception as exc:
                    raise VectorStoreError(f"Failed to initialize ChromaDB: {exc}") from exc
    return _collection


def document_hash_exists(file_hash: str) -> Optional[str]:
    """Return the document_id if a document with this content hash is already indexed."""
    collection = _get_collection()
    try:
        result = collection.get(where={"file_hash": file_hash}, limit=1)
    except Exception as exc:
        raise VectorStoreError(f"Failed checking for duplicate document: {exc}") from exc
    if result and result.get("ids"):
        return result["metadatas"][0].get("document_id")
    return None


def add_chunks(chunks: List[Chunk], file_hash: str) -> None:
    """Embed and persist a batch of chunks into the vector store."""
    if not chunks:
        return
    collection = _get_collection()
    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas: List[Dict[str, Any]] = []
    for c in chunks:
        meta = dict(c.metadata)
        meta["file_hash"] = file_hash
        metadatas.append(meta)

    try:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    except Exception as exc:
        raise VectorStoreError(f"Failed to add chunks to ChromaDB: {exc}") from exc

    logger.info("Added %d chunks for document_id=%s to ChromaDB", len(chunks), chunks[0].document_id)


def similarity_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    """Plain vector similarity search (no diversity re-ranking)."""
    collection = _get_collection()
    try:
        query_embedding = embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise VectorStoreError(f"Similarity search failed: {exc}") from exc

    return _format_results(results)


def mmr_search(
    query: str,
    top_k: int,
    fetch_k: int,
    lambda_mult: float,
) -> List[Dict[str, Any]]:
    """Maximum Marginal Relevance search: balances relevance and diversity.

    Chroma has no native MMR, so we fetch `fetch_k` candidates by
    similarity and re-rank them with a manual MMR implementation over
    the embedding vectors, matching LangChain's standard approach.
    """
    import numpy as np

    collection = _get_collection()
    try:
        query_embedding = embed_query(query)
        candidates = collection.query(
            query_embeddings=[query_embedding],
            n_results=max(fetch_k, top_k),
            include=["documents", "metadatas", "distances", "embeddings"],
        )
    except Exception as exc:
        raise VectorStoreError(f"MMR search failed: {exc}") from exc

    doc_embeddings = candidates.get("embeddings", [[]])[0]
    documents = candidates.get("documents", [[]])[0]
    metadatas = candidates.get("metadatas", [[]])[0]
    distances = candidates.get("distances", [[]])[0]

    if len(documents) == 0:
        return []

    doc_embeddings = np.array(doc_embeddings)
    query_vec = np.array(query_embedding)

    def cosine_sim(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-10
        return float(np.dot(a, b) / denom)

    selected_indices: List[int] = []
    remaining_indices = list(range(len(documents)))

    # Precompute relevance of each candidate to the query.
    relevance_scores = [cosine_sim(query_vec, emb) for emb in doc_embeddings]

    while remaining_indices and len(selected_indices) < top_k:
        if not selected_indices:
            # First pick: the most relevant candidate.
            best_idx = max(remaining_indices, key=lambda i: relevance_scores[i])
        else:
            best_idx, best_score = None, float("-inf")
            for i in remaining_indices:
                diversity_penalty = max(
                    cosine_sim(doc_embeddings[i], doc_embeddings[j]) for j in selected_indices
                )
                mmr_score = (
                    lambda_mult * relevance_scores[i]
                    - (1 - lambda_mult) * diversity_penalty
                )
                if mmr_score > best_score:
                    best_score, best_idx = mmr_score, i
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    results = {
        "documents": [[documents[i] for i in selected_indices]],
        "metadatas": [[metadatas[i] for i in selected_indices]],
        "distances": [[distances[i] for i in selected_indices]],
    }
    return _format_results(results)


def _format_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize raw Chroma query output into a list of retrieval hits."""
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    hits: List[Dict[str, Any]] = []
    for text, meta, distance in zip(documents, metadatas, distances):
        similarity = round(1 - distance, 4) if distance is not None else None
        hits.append(
            {
                "text": text,
                "document_name": meta.get("document_name"),
                "document_id": meta.get("document_id"),
                "page_number": meta.get("page_number"),
                "chunk_index": meta.get("chunk_index"),
                "similarity": similarity,
            }
        )
    return hits


def list_documents() -> List[Dict[str, Any]]:
    """Return one summary entry per unique indexed document."""
    collection = _get_collection()
    try:
        all_items = collection.get(include=["metadatas"])
    except Exception as exc:
        raise VectorStoreError(f"Failed listing documents: {exc}") from exc

    summary: Dict[str, Dict[str, Any]] = {}
    for meta in all_items.get("metadatas", []):
        doc_id = meta.get("document_id")
        if doc_id not in summary:
            summary[doc_id] = {
                "document_id": doc_id,
                "document_name": meta.get("document_name"),
                "chunk_count": 0,
            }
        summary[doc_id]["chunk_count"] += 1

    return list(summary.values())


def delete_document(document_id: str) -> int:
    """Delete all chunks belonging to a document. Returns count deleted."""
    collection = _get_collection()
    try:
        existing = collection.get(where={"document_id": document_id})
        ids = existing.get("ids", [])
        if not ids:
            return 0
        collection.delete(ids=ids)
        return len(ids)
    except Exception as exc:
        raise VectorStoreError(f"Failed to delete document {document_id}: {exc}") from exc
