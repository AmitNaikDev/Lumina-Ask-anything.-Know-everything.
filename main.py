import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.dependencies import load_resources, clear_resources
from api.routers import ingest, query
from api.schemas import HealthResponse
from src.utils.config import settings, validate_settings

# Logging Setup
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        validate_settings()
        load_resources()
        logger.info("Application started successfully.")
    except Exception:
        logger.exception("FATAL: Startup failed — see traceback above.")
        raise
    yield
    clear_resources()
    logger.info("Application shut down.")


# App Init
app = FastAPI(
    title="Lumina API",
    description="Lumina — Ask anything. Know everything. Upload documents and ask natural language questions via RAG.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(ingest.router)
app.include_router(query.router)


# Health Check
@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health_check():
    return HealthResponse(
        status="healthy",
        environment=settings.APP_ENV,
        llm_model=settings.LLM_MODEL_NAME,
        embed_model=settings.EMBEDDING_MODEL
    )
