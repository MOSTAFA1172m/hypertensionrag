# Hypertension RAG — production image
# Retriever: BM25 over the deduplicated corpus (data/processed/validated_chunks_dedup.json)
# Generator: Google Gemini API (GEMINI_API_KEY required at runtime)

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

# libgomp1 for numpy/pymupdf wheels on slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY app.py run_wsgi.py entrypoint.sh ./
COPY src/ src/
COPY templates/ templates/

# Seed artifacts (copied into the writable volume by entrypoint.sh on first run)
COPY data/processed/validated_chunks_dedup.json /app/seed/processed/validated_chunks_dedup.json
COPY data/source_registry.json /app/seed/source_registry.json

RUN chmod +x entrypoint.sh

EXPOSE 7860

CMD ["sh", "entrypoint.sh"]