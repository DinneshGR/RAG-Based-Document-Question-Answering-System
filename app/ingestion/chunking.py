"""
Text chunking module.

Wraps LangChain's RecursiveCharacterTextSplitter with configurable
chunk size/overlap (from app.config) and attaches rich metadata
(document name, page number, chunk id, document id) to every chunk so
retrieval results can always be traced back to their source.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.ingestion.loaders import PageContent
from app.ingestion.metadata import new_chunk_id

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk of text ready for embedding, with full provenance."""

    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def build_splitter(
    chunk_size: int | None = None, chunk_overlap: int | None = None
) -> RecursiveCharacterTextSplitter:
    """Create a text splitter using configured (or overridden) values."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_pages(
    pages: List[PageContent],
    document_id: str,
    document_name: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Chunk]:
    """Split every page's text into overlapping chunks with metadata."""
    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks: List[Chunk] = []
    running_index = 0

    for page in pages:
        pieces = splitter.split_text(page.text)
        for piece in pieces:
            chunks.append(
                Chunk(
                    chunk_id=new_chunk_id(),
                    document_id=document_id,
                    document_name=document_name,
                    page_number=page.page_number,
                    text=piece,
                    chunk_index=running_index,
                    metadata={
                        "document_id": document_id,
                        "document_name": document_name,
                        "page_number": page.page_number,
                        "chunk_index": running_index,
                    },
                )
            )
            running_index += 1

    logger.info(
        "Chunked document '%s' into %d chunks (size=%s, overlap=%s)",
        document_name,
        len(chunks),
        chunk_size or settings.CHUNK_SIZE,
        chunk_overlap or settings.CHUNK_OVERLAP,
    )
    return chunks
