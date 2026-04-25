# ✨ Lumina — Ask anything. Know everything.

An end-to-end Retrieval Augmented Generation (RAG) pipeline that lets you
upload any document and ask natural language questions about it.
Answers are grounded strictly in your document — no hallucination.

---

## 🧠 How It Works
```
Document uploaded
      │
loader    →  raw text (PDF / TXT / CSV)
      │
chunker   →  overlapping text chunks
      │
embedder  →  vectors stored in ChromaDB
      
User asks a question
      │
retriever →  top-k similar chunks via similarity search
      │
prompt    →  context + question injected into template
      │
LLM       →  grounded answer generated
      │
UI        →  answer + source citations displayed
```

---

## 🏗️ Project Structure
```
lumina/
├── src/
│   ├── ingestion/
│   │   ├── loader.py          # load PDF, TXT, CSV
│   │   ├── chunker.py         # split into overlapping chunks
│   │   └── embedder.py        # embed + store in ChromaDB
│   ├── retrieval/
│   │   └── retriever.py       # similarity search + MMR retrieval
│   ├── generation/
│   │   ├── prompt.py          # prompt templates (chat + instruct)
│   │   └── chain.py           # LangChain RAG + conversational chains
│   └── utils/
│       └── config.py          # central config via pydantic-settings
├── api/
│   ├── main.py                # FastAPI app init — wiring only
│   ├── schemas.py             # Pydantic request/response models
│   ├── dependencies.py        # shared embedding model
│   └── routers/
│       ├── ingest.py          # POST /ingest
│       └── query.py           # POST /query
├── ui/
│   └── app.py                 # Streamlit chat interface
├── tests/
│   ├── test_loader.py
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_chain.py
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI pipeline
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## ⚙️ Setup — Local
```bash
# 1. Clone
git clone https://github.com/AmitNaikDev/InsightRAG-Intelligent-Document-Analysis-Bot.git
cd lumina

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Create required directories
mkdir -p chroma_store data/uploads outputs
```

---

## 🚀 Running Locally

### Terminal 1 — Start the API
```bash
uvicorn main:app --reload --port 8000
```

### Terminal 2 — Start the UI
```bash
streamlit run ui/app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🐳 Running with Docker
```bash
# 1. Add your API key to .env
cp .env.example .env

# 2. Build and start all services
docker-compose up --build

# 3. Open in browser
# UI  → http://localhost:8501
# API → http://localhost:8000/docs
```
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clears ChromaDB data)
docker-compose down -v
```

---

## 🔁 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | API status + active models |
| `/ingest` | POST | Upload document → embed → store |
| `/query` | POST | Ask a question → answer + citations |
| `/query/status/{collection}` | GET | Check if collection is ready |
| `/docs` | GET | Interactive Swagger UI |

### Example — Ingest
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/sample_docs/sample.pdf"
```

Response:
```json
{
  "message":         "Document ingested successfully.",
  "collection_name": "sample",
  "total_chunks":    42,
  "file_name":       "sample.pdf"
}
```

### Example — Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question":        "What is the main topic of this document?",
    "collection_name": "sample",
    "top_k":           4
  }'
```

Response:
```json
{
  "answer":          "The document covers...",
  "collection_name": "sample",
  "question":        "What is the main topic of this document?",
  "sources": [
    {
      "source":      "sample.pdf",
      "chunk_index": 3,
      "content":     "..."
    }
  ],
  "total_sources": 4
}
```

---

## 🧪 Running Tests
```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 🔑 Key ML / Engineering Concepts Applied

| Concept | Where |
|---|---|
| RAG pipeline | Full ingestion + retrieval + generation loop |
| Vector embeddings | `sentence-transformers` — all-MiniLM-L6-v2 |
| Vector database | ChromaDB — persisted to disk |
| MMR retrieval | Diversity-aware chunk selection |
| Prompt engineering | Separate system + human message templates |
| LCEL chain | Modern LangChain `prompt \| llm \| parser` pattern |
| Conversational RAG | Memory-aware multi-turn Q&A |
| Source citations | Every answer includes chunk-level sources |
| Dependency injection | FastAPI `Depends()` for shared embedding model |
| Docker | Multi-service containerisation |
| CI/CD | GitHub Actions — test on every push |

---

## 🛠️ Tech Stack

`LangChain` · `ChromaDB` · `sentence-transformers` · `OpenAI API` ·
`FastAPI` · `Streamlit` · `Docker` · `GitHub Actions` · `Pydantic`

---

## ⚠️ Disclaimer

This project is for educational purposes.
Answers are generated by an LLM and may not always be accurate.
Do not use for legal, medical, or financial decisions.