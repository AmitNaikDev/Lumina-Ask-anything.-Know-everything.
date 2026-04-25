
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import requests
import streamlit as st
from pathlib import Path

# Config

API_BASE_URL     = os.getenv("API_BASE_URL", "http://localhost:8000")
ALLOWED_TYPES    = ["pdf", "txt", "csv"]
MAX_FILE_SIZE_MB = 10


# Page Setup

st.set_page_config(
    page_title="Lumina — Ask anything. Know everything.",
    page_icon="✨",
    layout="wide"
)


# Session State Init

def init_session_state() -> None:
    """Initialize session state keys to persist data across Streamlit reruns."""
    defaults = {
        "collection_name": None,      # set after successful ingest
        "file_name":       None,      # display name of uploaded doc
        "chat_history":    [],        # list of {role, content, sources}
        "ingested":        False,     # controls which view to show
        "total_chunks":    0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# API Helpers

def ingest_document(file) -> dict | None:
    """Send uploaded file to API and return response JSON or None on failure."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ingest",
            files={"file": (file.name, file.getvalue(), file.type)},
            timeout=120     # large PDFs take time to embed
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Make sure the FastAPI server is running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Ingest failed: {e.response.json().get('detail', str(e))}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None


def query_document(question: str, collection_name: str, top_k: int = 4) -> dict | None:
    """Send a question to API and return response JSON or None on failure."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "question":        question,
                "collection_name": collection_name,
                "top_k":           top_k
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Query failed: {e.response.json().get('detail', str(e))}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None


def check_api_health() -> bool:
    """Checks if the FastAPI backend is reachable."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


# Sidebar

def render_sidebar() -> int:
    """Render sidebar with health indicator, uploader, settings, and reset button."""
    with st.sidebar:
        st.title("✨ Lumina")
        st.caption("Ask anything. Know everything.")
        st.divider()

        # API Health
        if check_api_health():
            st.success("API connected", icon="🟢")
        else:
            st.error("API not reachable", icon="🔴")
            st.caption("Run: `uvicorn api.main:app --reload`")

        st.divider()

        # File Uploader
        st.subheader("📂 Upload Document")
        uploaded_file = st.file_uploader(
            label="Choose a file",
            type=ALLOWED_TYPES,
            help=f"Supported: PDF, TXT, CSV — Max {MAX_FILE_SIZE_MB}MB"
        )

        if uploaded_file:
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"File too large ({file_size_mb:.1f}MB). Max is {MAX_FILE_SIZE_MB}MB.")
            else:
                st.caption(f"📎 {uploaded_file.name} ({file_size_mb:.2f} MB)")

                if st.button("🚀 Ingest Document", use_container_width=True):
                    with st.spinner("Ingesting document... this may take a moment."):
                        result = ingest_document(uploaded_file)

                    if result:
                        st.session_state.collection_name = result["collection_name"]
                        st.session_state.file_name       = result["file_name"]
                        st.session_state.total_chunks    = result["total_chunks"]
                        st.session_state.ingested        = True
                        st.session_state.chat_history    = []
                        st.success(f"✅ Ingested successfully!")
                        st.caption(
                            f"Collection: `{result['collection_name']}` "
                            f"| {result['total_chunks']} chunks"
                        )

        st.divider()

        # Settings
        st.subheader("⚙️ Settings")
        top_k = st.slider(
            label="Chunks to retrieve (top_k)",
            min_value=1,
            max_value=10,
            value=4,
            help="Higher = more context for the LLM, but slower."
        )

        st.divider()

        # Document Info
        if st.session_state.ingested:
            st.subheader("📋 Active Document")
            st.info(
                f"**{st.session_state.file_name}**\n\n"
                f"Collection: `{st.session_state.collection_name}`\n\n"
                f"Chunks: {st.session_state.total_chunks}"
            )

        # Reset Session
        if st.session_state.ingested:
            if st.button("🗑️ Reset Session", use_container_width=True):
                for key in st.session_state:
                    del st.session_state[key]
                st.rerun()

    return top_k


# Chat History Renderer

def render_chat_history() -> None:
    """Render all previous chat messages with expandable sources for assistant replies."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources only for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander(
                    f"📚 Sources used ({len(message['sources'])} chunks)"
                ):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Chunk {i+1}** — `{source['source']}` "
                                    f"(index: {source['chunk_index']})")
                        st.caption(source["content"])
                        if i < len(message["sources"]) - 1:
                            st.divider()


# Main Chat View

def render_chat(top_k: int) -> None:
    """Render main chat interface and handle user input/API calls."""
    st.title("💬 Ask Lumina")
    st.caption(
        f"Chatting with: **{st.session_state.file_name}** "
        f"| Collection: `{st.session_state.collection_name}`"
    )

    # Render existing messages
    render_chat_history()

    # Chat input
    if question := st.chat_input("Ask a question about your document..."):

        # Add user message to history and render it
        st.session_state.chat_history.append({
            "role":    "user",
            "content": question,
            "sources": []
        })
        with st.chat_message("user"):
            st.markdown(question)

        # Query the API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = query_document(
                    question=question,
                    collection_name=st.session_state.collection_name,
                    top_k=top_k
                )

            if result:
                answer  = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:
                    with st.expander(f"📚 Sources used ({len(sources)} chunks)"):
                        for i, source in enumerate(sources):
                            st.markdown(
                                f"**Chunk {i+1}** — `{source['source']}` "
                                f"(index: {source['chunk_index']})"
                            )
                            st.caption(source["content"])
                            if i < len(sources) - 1:
                                st.divider()

                # Persist to history
                st.session_state.chat_history.append({
                    "role":    "assistant",
                    "content": answer,
                    "sources": sources
                })
            else:
                st.error("Failed to get an answer. Please try again.")


# Landing View

def render_landing() -> None:
    """Show landing page with upload guidance before document ingestion."""
    st.title("✨ Lumina")
    st.subheader("Ask anything. Know everything.")
    st.markdown(
        "Upload any document and ask natural language questions about it. "
        "Lumina retrieves the most relevant passages and generates "
        "accurate, grounded answers — no hallucination."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Upload")
        st.caption("Upload a PDF, TXT, or CSV file from the sidebar.")
    with col2:
        st.markdown("### 2️⃣ Ingest")
        st.caption("Click Ingest — your document is chunked, embedded, and stored.")
    with col3:
        st.markdown("### 3️⃣ Ask")
        st.caption("Ask any question — get answers grounded in your document.")

    st.divider()
    st.info("👈 Upload a document in the sidebar to get started.")


# App Entry Point

def main() -> None:
    top_k = render_sidebar()

    if st.session_state.ingested:
        render_chat(top_k)
    else:
        render_landing()


if __name__ == "__main__":
    main()