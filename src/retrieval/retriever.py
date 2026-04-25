import logging
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from src.utils.config import settings

logger = logging.getLogger(__name__)


def _get_chroma_store(collection_name: str, embedding_model: HuggingFaceEmbeddings) -> Chroma:
    """Internal helper — returns a Chroma store for a given collection."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=settings.CHROMA_PERSIST_DIR
    )


# Core Retriever
def get_retriever(
    collection_name: str,
    embedding_model: HuggingFaceEmbeddings,
    top_k: Optional[int] = None
) -> BaseRetriever:
    """Loads a ChromaDB collection and returns a LangChain similarity retriever."""
    k = top_k or settings.TOP_K_RESULTS

    vector_store = _get_chroma_store(collection_name, embedding_model)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    logger.info("Retriever ready — collection: '%s' | top_k: %d", collection_name, k)
    return retriever


# MMR Retriever
def get_mmr_retriever(
    collection_name: str,
    embedding_model: HuggingFaceEmbeddings,
    top_k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5
) -> BaseRetriever:
    """Returns an MMR retriever that balances relevance and diversity."""
    vector_store = _get_chroma_store(collection_name, embedding_model)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
    )
    logger.info(
        "MMR Retriever ready — collection: '%s' | top_k: %d | fetch_k: %d | lambda_mult: %.1f",
        collection_name, top_k, fetch_k, lambda_mult
    )
    return retriever


def similarity_search(
    query: str,
    collection_name: str,
    embedding_model: HuggingFaceEmbeddings,
    top_k: int = 4
) -> List[Document]:
    """Direct similarity search — returns Documents ranked by score."""
    k = top_k or settings.TOP_K_RESULTS

    vector_store = _get_chroma_store(collection_name, embedding_model)
    results = vector_store.similarity_search_with_score(query, k=k)

    logger.info("Top %d results for: '%s'", k, query)
    for i, (doc, score) in enumerate(results):
        logger.info(
            "  [%d] score=%.4f | source=%s | chunk=%s",
            i + 1, score,
            doc.metadata.get('source', 'unknown'),
            doc.metadata.get('chunk_index', '?')
        )

    return [doc for doc, _ in results]


def collection_exists(
    collection_name: str,
    embedding_model: HuggingFaceEmbeddings
) -> bool:
    """
    Checks if a ChromaDB collection exists and has documents in it.
    Uses the already-loaded embedding model instead of creating a new one.
    """
    try:
        store = _get_chroma_store(collection_name, embedding_model)
        count = store._collection.count()
        logger.info("Collection '%s' has %d vectors.", collection_name, count)
        return count > 0
    except Exception as e:
        logger.warning("Collection check failed: %s", e)
        return False
