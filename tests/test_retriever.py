"""Unit tests for app.retrieval.retriever, using a mocked vector store."""
from unittest.mock import patch

import pytest

from app.exceptions import EmptyQuestionError, NoRelevantContextError
from app.retrieval import retriever


FAKE_HITS = [
    {
        "text": "Revenue grew 18%.",
        "document_name": "report.pdf",
        "document_id": "doc-1",
        "page_number": 12,
        "chunk_index": 0,
        "similarity": 0.91,
    }
]


def test_retrieve_raises_on_empty_question():
    with pytest.raises(EmptyQuestionError):
        retriever.retrieve("   ")


def test_retrieve_mmr_mode_calls_mmr_search():
    with patch.object(retriever, "mmr_search", return_value=FAKE_HITS) as mock_mmr:
        results = retriever.retrieve("What is the revenue?", mode="mmr")
    mock_mmr.assert_called_once()
    assert results == FAKE_HITS


def test_retrieve_similarity_mode_calls_similarity_search():
    with patch.object(retriever, "similarity_search", return_value=FAKE_HITS) as mock_sim:
        results = retriever.retrieve("What is the revenue?", mode="similarity")
    mock_sim.assert_called_once()
    assert results == FAKE_HITS


def test_retrieve_raises_when_no_hits():
    with patch.object(retriever, "mmr_search", return_value=[]):
        with pytest.raises(NoRelevantContextError):
            retriever.retrieve("Anything?", mode="mmr")
