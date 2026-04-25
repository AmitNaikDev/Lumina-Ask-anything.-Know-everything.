from pydantic import BaseModel, Field
from typing import Optional, List, Union

# Ingest Schemas
class IngestResponse(BaseModel):
    """Returned after a document is successfully ingested."""
    message: str 
    collection_name: str 
    total_chunks: int 
    file_name: str 

# Query Schema 
class QueryRequest(BaseModel):     
    """Natural language question against an ingested document."""
    question: str = Field(..., min_length=3, description="Natural language question")
    collection_name: str = Field(..., description="Collection to query against")
    top_k: int = Field(4, ge=1, le=10, description="Number of chunks to retrieve")
    threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score filter")

class SourceChunk(BaseModel):
    """Specific chunk used as context for the answer."""
    source: str 
    chunk_index: Union[int, str]
    content: str 

class QueryResponse(BaseModel):
    """Structured RAG answer with source citations."""
    answer: str
    collection_name: str
    question: str
    sources: List[SourceChunk]
    total_sources: int

class HealthResponse(BaseModel):
    """API health status."""
    status: str
    environment: str
    llm_model: str
    embed_model: str

class CollectionStatusResponse(BaseModel):
    """Status of a specific vector collection."""
    collection_name: str
    exists: bool
    message: str