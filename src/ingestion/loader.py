import logging
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader

logger = logging.getLogger(__name__)

# Supported File formats
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv"}

# Core Loading 

def load_document(file_path: str) -> List[Document]:
    """Loads a document from disk and returns a list of LangChain Document objects."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported File Type: {ext}.")
    
    logger.info("Loading %s file: %s", ext.upper(), path.name)

    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".txt":
        return load_text(file_path)
    elif ext == ".csv":
        return load_csv(file_path)
    
    return []

# File Format Specific Loaders 

def load_pdf(file_path: str) -> List[Document]:
    """Loads a PDF page by page."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    logger.info("Loaded %d pages from PDF.", len(docs))
    return docs


def load_text(file_path: str) -> List[Document]:
    """Loads a plain text file as a single Document."""
    loader = TextLoader(file_path)
    docs = loader.load()
    logger.info("Loaded TXT file — %d document(s).", len(docs))
    return docs


def load_csv(file_path: str) -> List[Document]:
    """Loads a CSV where each row becomes one Document."""
    loader = CSVLoader(file_path)
    docs = loader.load()
    logger.info("Loaded %d rows from CSV.", len(docs))
    return docs


def validate_document(docs: List[Document]) -> None:
    """Validate that documents are not empty and have content."""
    if not docs:
        raise ValueError("Document is empty - no content was loaded")

    empty_pages = [d for d in docs if not d.page_content.strip()]
    if empty_pages:
        logger.warning("Found %d empty pages/documents.", len(empty_pages))
    
    total_chars = sum(len(d.page_content) for d in docs)
    logger.info("Total characters loaded: %s", f"{total_chars:,}")


