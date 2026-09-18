"""FastAPI routes for document upload, listing and deletion."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.exceptions import (
    CorruptedFileError,
    DuplicateDocumentError,
    EmptyDocumentError,
    RagAppError,
    UnsupportedFileTypeError,
    VectorStoreError,
)
from app.schemas import DeleteResponse, DocumentListResponse, DocumentSummary, UploadResponse
from app.services.rag_pipeline import ingest_document, save_upload
from app.vectorstore.chroma_store import delete_document, list_documents

logger = logging.getLogger(__name__)
router = APIRouter(tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(status_code=400, detail=f"File exceeds max size of {settings.MAX_UPLOAD_MB}MB.")

    saved_path = save_upload(file_bytes, file.filename)

    try:
        result = ingest_document(saved_path, file.filename)
        return UploadResponse(
            document_id=result["document_id"],
            document_name=result["document_name"],
            chunks_indexed=result["chunks_indexed"],
            message="Document uploaded and indexed successfully.",
        )
    except DuplicateDocumentError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (UnsupportedFileTypeError, EmptyDocumentError, CorruptedFileError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VectorStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RagAppError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/documents", response_model=DocumentListResponse)
async def get_documents() -> DocumentListResponse:
    try:
        docs = list_documents()
    except VectorStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return DocumentListResponse(
        documents=[DocumentSummary(**d) for d in docs]
    )


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def remove_document(document_id: str) -> DeleteResponse:
    try:
        deleted_count = delete_document(document_id)
    except VectorStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"No document found with id {document_id}")

    return DeleteResponse(
        document_id=document_id,
        chunks_deleted=deleted_count,
        message="Document deleted successfully.",
    )
