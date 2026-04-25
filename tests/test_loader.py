import pytest 
from pathlib import Path
from langchain_core.documents import Document 
from src.ingestion.loader import load_document, validate_document 

# Fixtures 
@pytest.fixture
def create_sample_txt(tmp_path):
    """Creates a sample txt file for tests."""
    sample = tmp_path / "sample.txt"
    sample.write_text("This is a text document.\nIt has multiple lines\nLine three.")
    return sample 

# Tests 
def test_load_txt_returns_documents(create_sample_txt):
    docs = load_document(str(create_sample_txt))
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert isinstance(docs[0], Document)

def test_load_txt_has_content(create_sample_txt):
    docs = load_document(str(create_sample_txt))
    assert docs[0].page_content.strip() != ""

def test_load_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        load_document("nonexistent/path/file.txt")
    
def test_unsupported_extension_raises(tmp_path):
    bad_file_path = tmp_path / "file.xyz"
    bad_file_path.write_text("Some Content")
    with pytest.raises(ValueError, match="Unsupported File Type"):
        load_document(str(bad_file_path))

def test_validate_document_passes_on_valid(create_sample_txt):
    docs = load_document(str(create_sample_txt))
    # Should not raise
    validate_document(docs)

def test_validate_document_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_document([])