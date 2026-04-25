import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from src.utils.config import settings

logger = logging.getLogger(__name__)

# Chunking configuration — driven by .env (CHUNK_SIZE, CHUNK_OVERLAP)
DEFAULT_CHUNK_SIZE = settings.CHUNK_SIZE
DEFAULT_CHUNK_OVERLAP = settings.CHUNK_OVERLAP


# Core Chunker Code

def chunk_documents(docs: List[Document], chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Document]:
    """Chunks documents into smaller segments with overlap to maintain context."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    logger.info("%d document(s) → %d chunks", len(docs), len(chunks))
    logger.info("Chunk size: %d | Overlap: %d", chunk_size, chunk_overlap) 

    return chunks

# Token Aware Chunker
def chunk_by_tokens(docs: List[Document], model_name: str = "all-MiniLM-L6-v2", tokens_per_chunk: int = 256, chunk_overlap: int = 32) -> List[Document]:
    """Splits by token count instead of character count."""
    splitter = SentenceTransformersTokenTextSplitter(
        model_name=model_name,
        tokens_per_chunk=tokens_per_chunk,
        chunk_overlap=chunk_overlap
    )
    # Note: split_documents is preferred for list of documents
    chunks = splitter.split_documents(docs)
    logger.info("Token-aware chunking → %d chunks", len(chunks))
    return chunks

# Metadata Enrichment
def enrich_chunk_metadata(chunks: List[Document], file_name: str) -> List[Document]:
    """Adds extra metadata to each chunk for citations."""
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "source": file_name,
            "chunk_index": i,
            "total_chunks": len(chunks)
        })
    return chunks

# Stats Helper 
def chunk_stats(chunks: List[Document]) -> None:
    if not chunks:
        logger.info("No chunks to analyze.")
        return
    sizes = [len(c.page_content) for c in chunks]
    logger.info("Chunk Stats — Total: %d | Min: %d | Max: %d | Avg: %d chars",
                len(chunks), min(sizes), max(sizes), sum(sizes) // len(sizes))