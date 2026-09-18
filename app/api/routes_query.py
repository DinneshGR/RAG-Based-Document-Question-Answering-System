"""FastAPI routes for question answering and health checks."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.exceptions import (
    EmptyQuestionError,
    ModelUnavailableError,
    NoRelevantContextError,
    OllamaUnavailableError,
    RagAppError,
    VectorStoreError,
)
from app.llm.ollama_client import check_ollama_health, is_model_available
from app.schemas import HealthResponse, QueryRequest, QueryResponse, RetrievedPassage, SourceCitation
from app.services.rag_pipeline import answer_question
from app.vectorstore.chroma_store import _get_collection  # noqa: F401 (used for health probe)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        result = answer_question(
            question=request.question,
            mode=request.mode or "mmr",
            top_k=request.top_k,
            fetch_k=request.fetch_k,
            lambda_mult=request.lambda_mult,
        )
    except EmptyQuestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NoRelevantContextError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VectorStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RagAppError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceCitation(**s) for s in result["sources"]],
        retrieved_passages=[
            RetrievedPassage(
                text=p["text"],
                document_name=p["document_name"],
                page_number=p.get("page_number"),
                similarity=p.get("similarity"),
            )
            for p in result["retrieved_passages"]
        ],
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    ollama_ok = check_ollama_health()
    model_ok = is_model_available() if ollama_ok else False

    vector_store_ok = True
    try:
        _get_collection()
    except Exception:
        vector_store_ok = False

    status = "ok" if (ollama_ok and model_ok and vector_store_ok) else "degraded"

    return HealthResponse(
        status=status,
        ollama_reachable=ollama_ok,
        model_available=model_ok,
        model_name=settings.OLLAMA_MODEL,
        vector_store_ready=vector_store_ok,
    )
