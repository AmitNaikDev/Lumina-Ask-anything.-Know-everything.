import logging
import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from langchain_huggingface import HuggingFaceEmbeddings
from api.schemas import IngestResponse
from api.dependencies import get_embedding_model
from src.ingestion.loader import load_document, validate_document
from src.ingestion.chunker import chunk_documents, enrich_chunk_metadata
from src.ingestion.embedder import embed_and_store, delete_collection
from src.retrieval.retriever import collection_exists
from src.utils.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# Temp directory for uploaded files
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv"}


import re
import asyncio

@router.post("", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    embedding_model: HuggingFaceEmbeddings = Depends(get_embedding_model)
):
    """Validate, save, and ingest a document into ChromaDB via a background thread."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"

    # Generate a Chroma-safe collection name
    raw_name = Path(file.filename).stem.lower()
    # Replace spaces with underscores and remove invalid characters
    clean_name = re.sub(r'[^a-z0-9_-]', '', raw_name.replace(" ", "_"))
    # Must start and end with alphanumeric
    clean_name = re.sub(r'^[^a-z0-9]+', '', clean_name)
    clean_name = re.sub(r'[^a-z0-9]+$', '', clean_name)
    # Ensure length is between 3 and 63
    if len(clean_name) < 3:
        clean_name = clean_name.ljust(3, '0')
    collection_name = clean_name[:63]

    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info("Saved upload to: %s", temp_path)

        # Ingest Pipeline
        def run_ingestion():
            docs = load_document(str(temp_path))
            validate_document(docs)

            # Use chunk settings from config (respects .env overrides)
            local_chunks = chunk_documents(
                docs,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            local_chunks = enrich_chunk_metadata(local_chunks, file_name=file.filename)

            # If collection already exists, delete it first to avoid duplicates
            if collection_exists(collection_name, embedding_model):
                logger.info("Collection '%s' already exists — overwriting.", collection_name)
                delete_collection(collection_name, embedding_model)

            embed_and_store(
                chunks=local_chunks,
                collection_name=collection_name,
                embedding_model=embedding_model
            )
            return len(local_chunks)

        # Run CPU/IO heavy parsing and embedding in a threadpool to prevent blocking FastAPI I/O
        total_chunks = await asyncio.to_thread(run_ingestion)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Ingestion failed for file: %s", file.filename)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always clean up temp file
        if temp_path.exists():
            temp_path.unlink()
            logger.info("Cleaned up temp file: %s", temp_path)

    return IngestResponse(
        message="Document ingested successfully.",
        collection_name=collection_name,
        total_chunks=total_chunks,
        file_name=file.filename
    )
