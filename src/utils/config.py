
import logging
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file explicitly
load_dotenv()

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent


# Settings
class Settings(BaseSettings):
    """Central configuration class that validates and loads settings from environment variables."""

    # LLM
    OPENAI_API_KEY:    str  = ""
    OPENROUTER_API_KEY: str  = ""
    GROQ_API_KEY:      str  = ""
    LLM_MODEL_NAME:    str  = "gpt-3.5-turbo"
    LLM_TEMPERATURE:   float = 0.0        # 0 = deterministic, best for Q&A
    LLM_MAX_TOKENS:    int  = 512

    # Embeddings
    EMBEDDING_MODEL:   str  = "all-MiniLM-L6-v2"

    # ChromaDB
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "chroma_store")

    # Chunking
    CHUNK_SIZE:        int  = 500
    CHUNK_OVERLAP:     int  = 50

    # Retrieval
    TOP_K_RESULTS:     int  = 4           # number of chunks to retrieve

    # API
    API_HOST:          str  = "0.0.0.0"
    API_PORT:          int  = 8000

    # App
    APP_ENV:           str  = "development"   # development | production
    LOG_LEVEL:         str  = "INFO"

    class Config:
        env_file     = ".env"
        env_file_encoding = "utf-8"
        extra        = "ignore"           # ignore unknown env vars silently


# Singleton Instance
# Every module does: from src.utils.config import settings
# Never instantiate Settings() again elsewhere
settings = Settings()


# Startup Validation
def validate_settings() -> None:
    """Validate critical configurations at startup to catch missing keys early."""
    errors = []

    if not settings.OPENAI_API_KEY and not settings.OPENROUTER_API_KEY and not settings.GROQ_API_KEY:
        errors.append("Neither OPENAI_API_KEY, OPENROUTER_API_KEY, nor GROQ_API_KEY is set.")

    if settings.CHUNK_OVERLAP >= settings.CHUNK_SIZE:
        errors.append(
            f"CHUNK_OVERLAP ({settings.CHUNK_OVERLAP}) must be "
            f"less than CHUNK_SIZE ({settings.CHUNK_SIZE})."
        )

    if errors:
        for err in errors:
            logger.error("CONFIG ERROR: %s", err)
        raise EnvironmentError(
            f"{len(errors)} config error(s) found. Check your .env file."
        )

    logger.info("All settings validated successfully.")


if __name__ == "__main__":
    validate_settings()
    print(f"\nActive settings:")
    print(f"  LLM Model      : {settings.LLM_MODEL_NAME}")
    print(f"  Embedding Model: {settings.EMBEDDING_MODEL}")
    print(f"  ChromaDB Dir   : {settings.CHROMA_PERSIST_DIR}")
    print(f"  Chunk Size     : {settings.CHUNK_SIZE}")
    print(f"  Top K Results  : {settings.TOP_K_RESULTS}")
    print(f"  Environment    : {settings.APP_ENV}")