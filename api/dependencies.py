import logging
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.config import settings

logger = logging.getLogger(__name__)

app_state = {}

def load_resources() -> None:
    """Load the embedding model into memory at application startup."""
    logger.info("Loading embedding model...")
    app_state["embedding_model"] = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    logger.info("Embedding model ready: %s", settings.EMBEDDING_MODEL)

def clear_resources() -> None:
    app_state.clear()
    logger.info("Resources cleared.")

# Dependency injectors
def get_embedding_model() -> HuggingFaceEmbeddings:
    model = app_state.get("embedding_model")
    if model is None:
        raise RuntimeError("Embedding model not loaded. Check startup.")
    return model
