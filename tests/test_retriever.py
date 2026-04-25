"""
Unit tests for src/retrieval/retriever.py

Uses unittest.mock to avoid requiring a live ChromaDB instance.
"""
from unittest.mock import patch, MagicMock
from src.retrieval.retriever import collection_exists


class TestCollectionExists:
    def test_returns_true_when_documents_exist(self):
        """collection_exists should return True when count > 0."""
        mock_store = MagicMock()
        mock_store._collection.count.return_value = 5

        with patch("src.retrieval.retriever.Chroma", return_value=mock_store), \
             patch("src.retrieval.retriever.load_embedding_model", return_value=MagicMock()):
            result = collection_exists("test_collection")

        assert result is True

    def test_returns_false_when_empty(self):
        """collection_exists should return False when count == 0."""
        mock_store = MagicMock()
        mock_store._collection.count.return_value = 0

        with patch("src.retrieval.retriever.Chroma", return_value=mock_store), \
             patch("src.retrieval.retriever.load_embedding_model", return_value=MagicMock()):
            result = collection_exists("empty_collection")

        assert result is False

    def test_returns_false_on_exception(self):
        """collection_exists should return False if ChromaDB raises an error."""
        with patch("src.retrieval.retriever.Chroma", side_effect=Exception("DB error")), \
             patch("src.retrieval.retriever.load_embedding_model", return_value=MagicMock()):
            result = collection_exists("broken_collection")

        assert result is False


class TestGetRetriever:
    def test_returns_retriever(self):
        """get_retriever should return a retriever object."""
        from src.retrieval.retriever import get_retriever

        mock_store = MagicMock()
        mock_retriever = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever

        with patch("src.retrieval.retriever.Chroma", return_value=mock_store), \
             patch("src.retrieval.retriever.load_embedding_model", return_value=MagicMock()):
            retriever = get_retriever("test_collection", top_k=3)

        mock_store.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        assert retriever is mock_retriever
