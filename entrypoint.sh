#!/bin/sh
# Seed runtime data on first start, then launch the app.
# The /app/data volume may be empty on a fresh host; copy seed artifacts from
# the image layer so BM25 retrieval works out of the box.
set -e

mkdir -p /app/data/processed /app/data/raw /app/data/qdrant

[ -f /app/data/processed/validated_chunks_dedup.json ] || \
    cp /app/seed/processed/validated_chunks_dedup.json /app/data/processed/validated_chunks_dedup.json

[ -f /app/data/source_registry.json ] || \
    cp /app/seed/source_registry.json /app/data/source_registry.json

exec python run_wsgi.py