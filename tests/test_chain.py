# tests/test_chain.py

import pytest
from unittest.mock import MagicMock, patch
from src.generation.chain import query_with_citations, build_rag_chain
from langchain_core.documents import Document

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_qa_chain():
    """
    Mocks a QA chain that returns a fixed answer and source documents.
    Supports the .invoke() method used in the production code.
    """
    chain = MagicMock()
    # Configure the mock to return this dict when .invoke() is called
    chain.invoke.return_value = {
        "result": "Apple's revenue was $94.8B in Q1 2024.",
        "source_documents": [
            Document(
                page_content="Apple reported revenue of $94.8B in Q1 2024.",
                metadata={"source": "apple_report.pdf", "chunk_index": 3}
            ),
            Document(
                page_content="Revenue grew 5% compared to Q1 2023.",
                metadata={"source": "apple_report.pdf", "chunk_index": 7}
            )
        ]
    }
    return chain


@pytest.fixture
def mock_retriever():
    return MagicMock()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_query_with_citations_returns_answer(mock_qa_chain):
    result = query_with_citations(mock_qa_chain, "What was Apple's revenue?")
    assert result["answer"] == "Apple's revenue was $94.8B in Q1 2024."


def test_query_with_citations_returns_sources(mock_qa_chain):
    result = query_with_citations(mock_qa_chain, "What was Apple's revenue?")
    assert len(result["sources"]) == 2


def test_source_has_required_fields(mock_qa_chain):
    result   = query_with_citations(mock_qa_chain, "What was Apple's revenue?")
    source   = result["sources"][0]
    assert "source"      in source
    assert "chunk_index" in source
    assert "content"     in source


def test_source_content_is_preview(mock_qa_chain):
    """Content should be truncated to 200 chars."""
    result  = query_with_citations(mock_qa_chain, "What was Apple's revenue?")
    content = result["sources"][0]["content"]
    assert len(content) <= 200


def test_build_rag_chain_returns_runnable(mock_retriever):
    """
    Checks that build_rag_chain() returns a chain without hitting the LLM.
    Mocks ChatOpenAI so no API key is needed in CI.
    """
    with patch("src.generation.chain.ChatOpenAI") as mock_llm:
        mock_llm.return_value = MagicMock()
        chain = build_rag_chain(mock_retriever)
        assert chain is not None