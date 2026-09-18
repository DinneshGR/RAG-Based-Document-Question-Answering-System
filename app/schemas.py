"""Pydantic request/response models for the FastAPI backend."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class UploadResponse(BaseModel):
    document_id: str
    document_name: str
    chunks_indexed: int
    message: str


class DocumentSummary(BaseModel):
    document_id: str
    document_name: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: List[DocumentSummary]


class DeleteResponse(BaseModel):
    document_id: str
    chunks_deleted: int
    message: str


class SourceCitation(BaseModel):
    document_name: str
    page_number: Optional[int] = None
    similarity: Optional[float] = None


class RetrievedPassage(BaseModel):
    text: str
    document_name: str
    page_number: Optional[int] = None
    similarity: Optional[float] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question")
    top_k: Optional[int] = Field(None, ge=1, le=50)
    fetch_k: Optional[int] = Field(None, ge=1, le=200)
    lambda_mult: Optional[float] = Field(None, ge=0.0, le=1.0)
    mode: Optional[str] = Field("mmr", pattern="^(mmr|similarity)$")


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    retrieved_passages: List[RetrievedPassage]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    ollama_reachable: bool
    model_available: bool
    model_name: str
    vector_store_ready: bool
