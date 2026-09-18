"""
RAG pipeline orchestration.

Ties together ingestion, chunking, embeddings, the vector store,
retrieval and the LLM into two high-level operations: `ingest_document`
and `answer_question`. This is the module the API routes call into.
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List

from app.config import settings
from app.exceptions import DuplicateDocumentError, UnsupportedFileTypeError
from app.ingestion.chunking import chunk_pages
from app.ingestion.loaders import load_document
from app.ingestion.metadata import compute_file_hash
from app.llm.ollama_client import generate_answer
from app.llm.prompts import build_rag_prompt
from app.retrieval.retriever import retrieve
from app.vectorstore.chroma_store import add_chunks, document_hash_exists

logger = logging.getLogger(__name__)


def ingest_document(upload_path: Path, original_filename: str) -> Dict[str, Any]:
    """Load, chunk, embed and store a document. Raises on duplicates/errors."""
    ext = Path(original_filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}'. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )

    file_hash = compute_file_hash(upload_path)
    existing_doc_id = document_hash_exists(file_hash)
    if existing_doc_id:
        raise DuplicateDocumentError(
            f"This document has already been indexed (document_id={existing_doc_id})."
        )

    document_id = str(uuid.uuid4())
    pages = load_document(upload_path)
    chunks = chunk_pages(pages, document_id=document_id, document_name=original_filename)
    add_chunks(chunks, file_hash=file_hash)

    logger.info("Ingested document '%s' as document_id=%s", original_filename, document_id)
    return {
        "document_id": document_id,
        "document_name": original_filename,
        "chunks_indexed": len(chunks),
    }


def save_upload(file_bytes: bytes, filename: str) -> Path:
    """Persist an uploaded file's bytes to the uploads directory."""
    safe_name = f"{uuid.uuid4().hex}_{Path(filename).name}"
    dest = settings.UPLOAD_DIR / safe_name
    with open(dest, "wb") as f:
        f.write(file_bytes)
    return dest


def answer_question(
    question: str,
    mode: str = "mmr",
    top_k: int | None = None,
    fetch_k: int | None = None,
    lambda_mult: float | None = None,
) -> Dict[str, Any]:
    """Run the full retrieve -> prompt -> generate pipeline for a question."""
    chunks: List[Dict[str, Any]] = retrieve(
        question, mode=mode, top_k=top_k, fetch_k=fetch_k, lambda_mult=lambda_mult
    )

    prompt = build_rag_prompt(question, chunks)
    answer = generate_answer(prompt["system"], prompt["user"])

    # Deduplicate sources (same document + page cited once).
    seen = set()
    sources = []
    for c in chunks:
        key = (c["document_name"], c.get("page_number"))
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "document_name": c["document_name"],
                    "page_number": c.get("page_number"),
                    "similarity": c.get("similarity"),
                }
            )

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_passages": chunks,
    }
