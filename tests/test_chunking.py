"""Unit tests for app.ingestion.chunking."""
from app.ingestion.chunking import chunk_pages
from app.ingestion.loaders import PageContent


def test_chunk_pages_respects_size_and_overlap():
    long_text = "sentence one. " * 200  # long enough to force multiple chunks
    pages = [PageContent(page_number=1, text=long_text)]
    chunks = chunk_pages(pages, document_id="doc-1", document_name="test.txt", chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 250  # allow small overshoot from separator logic
        assert chunk.document_id == "doc-1"
        assert chunk.document_name == "test.txt"
        assert chunk.page_number == 1


def test_chunk_pages_preserves_page_numbers():
    pages = [
        PageContent(page_number=1, text="Page one content here."),
        PageContent(page_number=2, text="Page two content here."),
    ]
    chunks = chunk_pages(pages, document_id="doc-2", document_name="multi.pdf")
    page_numbers = {c.page_number for c in chunks}
    assert page_numbers == {1, 2}


def test_chunk_ids_are_unique():
    pages = [PageContent(page_number=1, text="Some text. " * 100)]
    chunks = chunk_pages(pages, document_id="doc-3", document_name="dup.txt", chunk_size=100, chunk_overlap=10)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
