"""
Helpers for computing document identity and metadata.

A content hash (rather than filename) is used as the document ID so that
re-uploading an identical file is detected as a duplicate even if the
filename differs, while re-uploading a file with the same name but
different content is treated as a new document.
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path


def compute_file_hash(path: Path) -> str:
    """Compute a stable SHA-256 hash of a file's contents."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def new_chunk_id() -> str:
    """Generate a unique chunk identifier."""
    return str(uuid.uuid4())
