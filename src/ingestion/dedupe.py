"""
dedupe.py — Corpus-level exact-text deduplication for the ingestion pipeline.

Motivation: the original chunker assigned the same page text block to many
consecutive sections (e.g. ESC sec_079..sec_087 all carrying an identical
8.6.2 block), inflating the index with redundant chunks (813 -> 435, 46%
redundant). BM25 over the deduplicated corpus is the winning retriever
(see evaluation/EXPERIMENTS.md).

Design: chunks are grouped by normalized text (whitespace/soft-hyphen/
control-char insensitive). For each group ONE canonical chunk is kept = the
member with the lowest section number. Vectors are reused from the original
Qdrant collection (text is identical, so embeddings are identical too), so
rebuilding the dedup index costs no additional embedding API calls.

Call `rebuild_dedup_corpus()` after every ingest sync to keep the dedup
artifacts and the `hypertension_guidelines_dedup` collection in sync.
"""

import json
import re
import uuid
from collections import defaultdict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"[\x00-\x1f\x7f\ufffd]", " ", text)
    return re.sub(r"\s+", "", text).lower()


def section_number(chunk: dict) -> int:
    m = re.search(r"(\d+)", chunk.get("section_id", ""))
    return int(m.group(1)) if m else 10**9


def chunk_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def dedupe_chunks(chunks: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Group chunks by normalized text; keep canonical = lowest section number.

    Returns (canonical_chunks, remap) where remap maps dropped chunk_id ->
    canonical chunk_id.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        groups[normalize_text(c["text"])].append(c)

    canonicals: list[dict] = []
    remap: dict[str, str] = {}
    for members in groups.values():
        members_sorted = sorted(members, key=section_number)
        canonical = members_sorted[0]
        canonicals.append(canonical)
        for m in members:
            if m["chunk_id"] != canonical["chunk_id"]:
                remap[m["chunk_id"]] = canonical["chunk_id"]
    return canonicals, remap


def rebuild_dedup_corpus(
    chunks_path: Path,
    dedup_chunks_path: Path,
    remap_path: Path,
    embeddings_dedup_path: Path,
    qdrant_path: Path,
    original_collection: str,
    dedup_collection: str,
    client: QdrantClient | None = None,
    vector_size: int = 3072,
) -> tuple[list[dict], dict[str, str]]:
    """Recompute dedup corpus artifacts from validated_chunks.json.

    Reads vectors from the original Qdrant collection (no re-embedding),
    picks canonical chunks, writes validated_chunks_dedup.json,
    chunk_remap_dedup.json and embeddings_dedup.json, and rebuilds the
    `dedup_collection` collection from scratch.
    """
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    owns_client = client is None
    if client is None:
        client = QdrantClient(path=str(qdrant_path))

    try:
        # 1. pull vectors from the original collection
        vec_by_id: dict[str, list[float]] = {}
        offset = None
        while True:
            pts, offset = client.scroll(
                collection_name=original_collection, limit=1000,
                with_vectors=True, with_payload=True, offset=offset,
            )
            for p in pts:
                cid = p.payload.get("chunk_id")
                if cid:
                    vec_by_id[cid] = p.vector
            if offset is None:
                break

        missing = [c["chunk_id"] for c in chunks if c["chunk_id"] not in vec_by_id]
        if missing:
            raise RuntimeError(
                f"{len(missing)} chunks missing vectors in '{original_collection}' "
                f"(e.g. {missing[0]}). Re-index them first."
            )

        # 2. dedupe
        canonicals, remap = dedupe_chunks(chunks)

        # 3. write artifacts
        for path, data in (
            (dedup_chunks_path, canonicals),
            (remap_path, remap),
            (embeddings_dedup_path, [{**dict(c), "embedding": vec_by_id[c["chunk_id"]]} for c in canonicals]),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        # 4. rebuild dedup collection
        client.delete_collection(dedup_collection)
        client.create_collection(
            collection_name=dedup_collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=chunk_uuid(c["chunk_id"]),
                vector=vec_by_id[c["chunk_id"]],
                payload={
                    "chunk_id": c["chunk_id"],
                    "document_id": c["document_id"],
                    "document_name": c.get("document_name"),
                    "pdf_file": c.get("pdf_file", ""),
                    "section_id": c["section_id"],
                    "section_title": c.get("section_title"),
                    "page_start": c.get("page_start"),
                    "page_end": c.get("page_end"),
                    "token_count": c.get("token_count"),
                    "text": c.get("text"),
                },
            )
            for c in canonicals
        ]
        client.upsert(collection_name=dedup_collection, points=points)
        info = client.get_collection(dedup_collection)

        print(
            f"  [dedupe] {len(chunks)} -> {len(canonicals)} chunks "
            f"({len(remap)} dropped); collection '{dedup_collection}' = {info.points_count} pts"
        )
        return canonicals, remap
    finally:
        if owns_client:
            client.close()


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parents[2] / "data"
    rebuild_dedup_corpus(
        chunks_path=BASE / "processed" / "validated_chunks.json",
        dedup_chunks_path=BASE / "processed" / "validated_chunks_dedup.json",
        remap_path=BASE / "processed" / "chunk_remap_dedup.json",
        embeddings_dedup_path=BASE / "processed" / "embeddings_dedup.json",
        qdrant_path=BASE / "qdrant",
        original_collection="hypertension_guidelines",
        dedup_collection="hypertension_guidelines_dedup",
    )