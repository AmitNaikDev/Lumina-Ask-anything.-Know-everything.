import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.utils.config import settings

logger = logging.getLogger(__name__)


# Embedding model (standalone fallback — prefer dependency-injected model)
def get_embedding_model(model_name: str = settings.EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Loads a sentence-transformer embedding model locally."""
    logger.info("Loading embedding model: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# Chroma DB Store
def get_vector_store(
    collection_name: str,
    embedding_model: HuggingFaceEmbeddings,
    persist_dir: str = settings.CHROMA_PERSIST_DIR
) -> Chroma:
    """Connects to (or creates) a ChromaDB collection."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_dir
    )


def embed_and_store(
    chunks: List[Document],
    collection_name: str,
    embedding_model: Optional[HuggingFaceEmbeddings] = None
) -> Chroma:
    """Embeds chunks and stores data in ChromaDB."""
    if embedding_model is None:
        embedding_model = get_embedding_model()

    logger.info("Embedding %d chunks into collection: '%s'", len(chunks), collection_name)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=settings.CHROMA_PERSIST_DIR
    )

    if hasattr(vector_store, 'persist'):
        vector_store.persist()

    logger.info("Stored and persisted %d vectors to ChromaDB.", len(chunks))
    return vector_store


def delete_collection(
    collection_name: str,
    embedding_model: Optional[HuggingFaceEmbeddings] = None
) -> None:
    """Deletes a ChromaDB collection. Accepts optional embedding_model to avoid re-loading."""
    if embedding_model is None:
        embedding_model = get_embedding_model()
    store = get_vector_store(collection_name, embedding_model)
    store.delete_collection()
    logger.info("Deleted collection: '%s'", collection_name)