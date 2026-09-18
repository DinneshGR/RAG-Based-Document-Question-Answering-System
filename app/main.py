"""FastAPI application entrypoint for the RAG Document Q&A backend."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_documents import router as documents_router
from app.api.routes_query import router as query_router
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    logger.info("RAG Document Q&A API started. Ollama model=%s", settings.OLLAMA_MODEL)
    yield


app = FastAPI(
    title="RAG Document Q&A API",
    description=(
        "A local, fully open-source Retrieval-Augmented Generation API. "
        "Upload documents and ask grounded, cited questions about them."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(query_router)


@app.get("/")
async def root() -> dict:
    return {
        "message": "RAG Document Q&A API is running.",
        "docs": "/docs",
        "health": "/health",
    }
