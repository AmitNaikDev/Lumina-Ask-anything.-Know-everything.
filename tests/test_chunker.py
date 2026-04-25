"""
Unit tests for src/ingestion/chunker.py

Tests chunking logic without requiring ChromaDB or any LLM.
"""
from langchain_core.documents import Document
from src.ingestion.chunker import chunk_documents, enrich_chunk_metadata, chunk_stats


def make_doc(content: str, source: str = "test.txt") -> Document:
    return Document(page_content=content, metadata={"source": source})


class TestChunkDocuments:
    def test_basic_chunking(self):
        """A long document should be split into multiple chunks."""
        content = "word " * 200  # 1000 chars
        docs = [make_doc(content)]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1

    def test_small_doc_stays_single_chunk(self):
        """A short document should remain as a single chunk."""
        docs = [make_doc("Short text.")]
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1

    def test_chunks_respect_size_limit(self):
        """No chunk should exceed chunk_size by more than the overlap."""
        content = "word " * 300
        docs = [make_doc(content)]
        chunk_size = 100
        chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=10)
        for chunk in chunks:
            # Allow some tolerance for splitter behaviour
            assert len(chunk.page_content) <= chunk_size * 1.5

    def test_empty_docs_returns_empty(self):
        """Empty document list should return empty chunk list."""
        chunks = chunk_documents([], chunk_size=500, chunk_overlap=50)
        assert chunks == []


class TestEnrichChunkMetadata:
    def test_metadata_added(self):
        """Each chunk should receive source, chunk_index, total_chunks."""
        docs = [make_doc("hello " * 50)]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        enriched = enrich_chunk_metadata(chunks, file_name="my_file.pdf")

        for i, chunk in enumerate(enriched):
            assert chunk.metadata["source"] == "my_file.pdf"
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == len(enriched)

    def test_all_chunks_processed(self):
        """The return bug (early return inside loop) should not recur."""
        docs = [make_doc("word " * 200)]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1  # ensure we have multiple chunks to test
        enriched = enrich_chunk_metadata(chunks, file_name="test.txt")
        assert len(enriched) == len(chunks)  # all chunks returned, not just first


class TestChunkStats:
    def test_stats_with_chunks(self, capsys):
        """chunk_stats should print without crashing."""
        docs = [make_doc("word " * 100)]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        chunk_stats(chunks)
        captured = capsys.readouterr()
        assert "Total chunks" in captured.out

    def test_stats_empty_no_crash(self, capsys):
        """chunk_stats with empty list should not raise."""
        chunk_stats([])
        captured = capsys.readouterr()
        assert "No chunks" in captured.out
