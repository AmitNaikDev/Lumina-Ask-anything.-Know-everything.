# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.11 slim — smaller image, faster builds
FROM python:3.11-slim

# ── Environment flags ─────────────────────────────────────────────────────────
# Prevents .pyc files being written to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python buffering stdout/stderr — logs appear immediately
ENV PYTHONUNBUFFERED=1

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# libgl1 and libglib2.0 needed by some PDF processing libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first — Docker caches this layer
# So rebuilds skip pip install if requirements.txt hasn't changed
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── App source ────────────────────────────────────────────────────────────────
COPY . .

# ── Persistent volume mount points ────────────────────────────────────────────
# These directories are mounted from the host via docker-compose
# so data survives container restarts
RUN mkdir -p chroma_store data/uploads outputs

# ── Ports ─────────────────────────────────────────────────────────────────────
# FastAPI runs on 8000, Streamlit on 8501
EXPOSE 8000 8501