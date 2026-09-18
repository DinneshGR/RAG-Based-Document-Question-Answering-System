"""Unit tests for app.embeddings.embedding_model, using a lightweight mock model
so tests don't require downloading the real Sentence Transformers model."""
from unittest.mock import MagicMock, patch

import numpy as np

from app.embeddings import embedding_model


def _make_fake_model():
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    return fake_model


def test_embed_texts_returns_list_of_vectors():
    with patch.object(embedding_model, "get_embedding_model", return_value=_make_fake_model()):
        vectors = embedding_model.embed_texts(["hello", "world"])
    assert len(vectors) == 2
    assert all(isinstance(v, list) for v in vectors)


def test_embed_texts_empty_input_returns_empty_list():
    assert embedding_model.embed_texts([]) == []


def test_embed_query_returns_single_vector():
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    with patch.object(embedding_model, "get_embedding_model", return_value=fake_model):
        vector = embedding_model.embed_query("hello")
    assert isinstance(vector, list)
    assert len(vector) == 3
