import logging
from fastapi import APIRouter, HTTPException, Depends
from langchain_huggingface import HuggingFaceEmbeddings
from api.schemas import QueryRequest, QueryResponse, SourceChunk
from api.dependencies import get_embedding_model
from src.retrieval.retriever import get_retriever, collection_exists
from src.generation.chain import build_qa_chain, query_with_citations

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
def query_document(
    request:         QueryRequest,
    embedding_model: HuggingFaceEmbeddings = Depends(get_embedding_model)
):
    """Execute RAG query against a collection and return answer with citations."""
    # Validate Collection
    if not collection_exists(request.collection_name, embedding_model):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{request.collection_name}' not found. "
                   f"Please ingest a document first."
        )

    # Build Retriever and Chain
    try:
        retriever = get_retriever(
            collection_name=request.collection_name,
            embedding_model=embedding_model,
            top_k=request.top_k
        )
        chain  = build_qa_chain(retriever)
        result = query_with_citations(chain, request.question)

    except Exception as e:
        logger.exception("Query failed for collection: %s", request.collection_name)
        raise HTTPException(status_code=500, detail=str(e))

    # Build Source Citations
    sources = [
        SourceChunk(
            source=      s["source"],
            chunk_index= s["chunk_index"],
            content=     s["content"]
        )
        for s in result["sources"]
    ]

    return QueryResponse(
        answer=          result["answer"],
        collection_name= request.collection_name,
        question=        request.question,
        sources=         sources,
        total_sources=   len(sources)
    )


@router.get("/status/{collection_name}", tags=["Query"])
def collection_status(
    collection_name: str,
    embedding_model: HuggingFaceEmbeddings = Depends(get_embedding_model)
):
    """Check if a collection exists and is ready to query."""
    exists = collection_exists(collection_name, embedding_model)
    return {
        "collection_name": collection_name,
        "exists":          exists,
        "message":         "Ready to query." if exists
                           else "Collection not found. Ingest a document first."
    }