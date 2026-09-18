"""Unit tests for app.ingestion.loaders."""
from pathlib import Path

import pytest

from app.exceptions import EmptyDocumentError, UnsupportedFileTypeError
from app.ingestion.loaders import load_document, load_txt


def test_load_txt_success(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello world.\nThis is a test document.")
    pages = load_txt(file_path)
    assert len(pages) == 1
    assert "Hello world" in pages[0].text
    assert pages[0].page_number == 1


def test_load_txt_empty_raises(tmp_path: Path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   \n\n  ")
    with pytest.raises(EmptyDocumentError):
        load_txt(file_path)


def test_load_document_unsupported_extension(tmp_path: Path):
    file_path = tmp_path / "file.xyz"
    file_path.write_text("data")
    with pytest.raises(UnsupportedFileTypeError):
        load_document(file_path)


def test_load_document_dispatches_txt(tmp_path: Path):
    file_path = tmp_path / "note.txt"
    file_path.write_text("Some content here.")
    pages = load_document(file_path)
    assert pages[0].text == "Some content here."
