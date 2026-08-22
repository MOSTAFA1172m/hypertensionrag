"""
benchmark_50.py

Retrieval evaluation on evaluation/test_set_50.json (50 questions) using the
project's existing retrieval modules:
  - rag/bm25_retriever.py        -> Sparse (BM25Okapi)
  - rag/hybrid_retriever.py      -> RRF and relative-score fusion
  - data/qdrant collection       -> Dense (gemini-embedding-001)

Metrics: Hit Rate@1/3/5, Precision@1/3/5, Recall@1/3/5, MRR, Avg Latency (ms).
Results are saved to evaluation/retrieval_50_results.json.

Dense/Hybrid modes require GEMINI_API_KEY (reads .env or environment).
Sparse mode runs fully offline.

Usage:
    python evaluation/benchmark_50.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from rag.bm25_retriever import BM25Retriever
from rag.hybrid_retriever import reciprocal_rank_fusion, relative_score_fusion

load_dotenv()

TEST_SET_PATH   = "evaluation/test_set_50.json"
CHUNKS_PATH     = "data/processed/validated_chunks.json"
OUTPUT_JSON     = "evaluation/retrieval_50_results.json"
QDRANT_PATH     = "data/qdrant"
COLLECTION_NAME = "hypertension_guidelines"

K_VALUES = [1, 3, 5]
HYBRID_TOP = 10  # per-retriever candidate pool before fusion


def load_test_set():
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def has_api_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


class DenseSearcher:
    """Wraps Gemini query embedding + Qdrant dense search. Created only when a key exists."""

    def __init__(self, qdrant: QdrantClient):
        from google import genai
        from google.genai import types

        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.types = types
        self.qdrant = qdrant

    def embed_query(self, q: str) -> list[float]:
        for attempt in range(5):
            try:
                resp = self.client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=q,
                    config=self.types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
                )
                return resp.embeddings[0].values
            except Exception as e:
                time.sleep(2.0)
        raise RuntimeError("Gemini embed query failed after retries.")

    def search(self, q_vec: list[float], top_n: int) -> list[dict]:
        hits = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            limit=top_n,
            with_payload=True,
        ).points
        res = []
        for h in hits:
            item = dict(h.payload)
            item["dense_score"] = float(h.score)
            item["score"] = float(h.score)
            res.append(item)
        return res


def evaluate_mode(name: str, run_fn, test_set) -> dict:
    latencies = []
    hit_counts = {k: 0 for k in K_VALUES}
    precision_sums = {k: 0.0 for k in K_VALUES}
    recall_sums = {k: 0.0 for k in K_VALUES}
    mrr_sum = 0.0
    per_query = []

    for tc in test_set:
        qid = tc["query_id"]
        query = tc["query"]
        expected_chunks = set(tc["expected_chunk_ids"])
        expected_section = tc.get("expected_section_id")

        t0 = time.perf_counter()
        retrieved = run_fn(query)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)

        retrieved_ids = [r["chunk_id"] for r in retrieved]
        retrieved_sids = [r.get("section_id") for r in retrieved]

        rank = None
        for idx, (cid, sid) in enumerate(zip(retrieved_ids, retrieved_sids), 1):
            if cid in expected_chunks or sid == expected_section:
                rank = idx
                break
        mrr_sum += (1.0 / rank) if rank is not None else 0.0

        q_record = {"query_id": qid, "top_retrieved": retrieved_ids[:3]}

        for k in K_VALUES:
            k_ids = retrieved_ids[:k]
            k_sids = retrieved_sids[:k]
            matches = sum(
                1 for cid, sid in zip(k_ids, k_sids)
                if cid in expected_chunks or sid == expected_section
            )
            hit_counts[k] += 1 if matches > 0 else 0
            precision_sums[k] += matches / k
            recall_sums[k] += min(1.0, matches / max(1, len(expected_chunks)))
            q_record[f"hit@{k}"] = 1 if matches > 0 else 0
            q_record[f"p@{k}"] = round(matches / k, 4)
            q_record[f"recall@{k}"] = round(min(1.0, matches / max(1, len(expected_chunks))), 4)

        per_query.append(q_record)

    n = len(test_set)
    metrics = {
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "mrr": round(mrr_sum / n, 4),
    }
    for k in K_VALUES:
        metrics[f"hit_rate@{k}"] = round(hit_counts[k] / n, 4)
        metrics[f"precision@{k}"] = round(precision_sums[k] / n, 4)
        metrics[f"recall@{k}"] = round(recall_sums[k] / n, 4)

    return {"mode": name, "metrics": metrics, "per_query": per_query}


def main():
    test_set = load_test_set()
    chunks = load_chunks()
    print(f"Loaded {len(test_set)} test queries and {len(chunks)} chunks.")
    print(f"GEMINI_API_KEY present: {has_api_key()}\n")

    bm25 = BM25Retriever(chunks=chunks)
    qdrant = QdrantClient(path=QDRANT_PATH)

    modes = []  # (title, run_fn)
    modes.append(("Sparse (BM25Okapi)", lambda q: bm25.search(q, top_k=max(K_VALUES))))

    if has_api_key():
        dense = DenseSearcher(qdrant)
        modes.append(("Dense (gemini-embedding-001)", lambda q: dense.search(dense.embed_query(q), top_n=max(K_VALUES))))
        modes.append(("Hybrid (RRF, c=60)", lambda q: reciprocal_rank_fusion(
            dense.search(dense.embed_query(q), top_n=HYBRID_TOP),
            bm25.search(q, top_k=HYBRID_TOP),
            top_k=max(K_VALUES), c=60, dense_weight=0.5,
        )))
        modes.append(("Hybrid (Score Fusion, a=0.6)", lambda q: relative_score_fusion(
            dense.search(dense.embed_query(q), top_n=HYBRID_TOP),
            bm25.search(q, top_k=HYBRID_TOP),
            top_k=max(K_VALUES), alpha=0.6,
        )))
    else:
        print("SKIPPING dense/hybrid modes — set GEMINI_API_KEY to enable them.\n")

    results = {"benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
               "test_set": TEST_SET_PATH, "test_set_size": len(test_set),
               "results": {}}
    for title, run_fn in modes:
        print(f">>> {title} ...")
        results["results"][title] = evaluate_mode(title, run_fn, test_set)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 95)
    print("RETRIEVAL BENCHMARK — test_set_50.json")
    print("=" * 95)
    rows = [
        ("Hit Rate @ 1", "hit_rate@1", "{:.1%}"),
        ("Hit Rate @ 3", "hit_rate@3", "{:.1%}"),
        ("Hit Rate @ 5", "hit_rate@5", "{:.1%}"),
        ("Precision @ 3", "precision@3", "{:.1%}"),
        ("Precision @ 5", "precision@5", "{:.1%}"),
        ("Recall @ 5", "recall@5", "{:.1%}"),
        ("MRR", "mrr", "{:.4f}"),
        ("Avg Latency (ms)", "avg_latency_ms", "{:.1f}"),
    ]
    titles = list(results["results"].keys())
    header = f"{'Metric':<20} | " + " | ".join(t[:20].ljust(22) for t in titles)
    print(header)
    print("-" * 95)
    for label, key, fmt in rows:
        vals = " | ".join(fmt.format(results["results"][t]["metrics"][key]).ljust(22) for t in titles)
        print(f"{label:<20} | {vals}")
    print("=" * 95)
    print(f"Detailed results saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()