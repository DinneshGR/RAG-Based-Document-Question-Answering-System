"""Custom exceptions used across the RAG pipeline.

Centralizing exceptions makes error handling in the API layer explicit
and lets us return clear, user-friendly messages instead of leaking
stack traces.
"""


class RagAppError(Exception):
    """Base class for all application-specific errors."""


class UnsupportedFileTypeError(RagAppError):
    """Raised when a user uploads a file type we don't support."""


class EmptyDocumentError(RagAppError):
    """Raised when a document contains no extractable text."""


class CorruptedFileError(RagAppError):
    """Raised when a file cannot be parsed/opened."""


class DuplicateDocumentError(RagAppError):
    """Raised when the same document (by content hash) is already indexed."""


class EmptyQuestionError(RagAppError):
    """Raised when the user submits a blank question."""


class NoRelevantContextError(RagAppError):
    """Raised when retrieval returns no chunks above any usable threshold."""


class VectorStoreError(RagAppError):
    """Raised when ChromaDB operations fail."""


class OllamaUnavailableError(RagAppError):
    """Raised when the Ollama server cannot be reached."""


class ModelUnavailableError(RagAppError):
    """Raised when the configured Ollama model is not pulled/available."""


class EmbeddingModelError(RagAppError):
    """Raised when the local embedding model fails to load or encode."""


class DocumentNotFoundError(RagAppError):
    """Raised when a requested document ID doesn't exist in the store."""
