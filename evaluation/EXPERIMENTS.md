# Experiment Log — Hypertension Guideline RAG

Consolidated record of every retrieval/rerank/generation experiment run in this project.
All per-query detail lives in the referenced JSON result files under `evaluation/`.

Corpus: WHO 2021 guideline, 2024 ESC guideline, USPSTF screening recommendation, CDC Health E-Stats.
Test sets: `test_set_50.json` (50 Q), `test_set_200.json` (200 Q: WHO 60 / ESC 90 / USPSTF 30 / CDC 20).
Scoring: hit = retrieved chunk_id is an expected chunk **or** its section_id matches `expected_section_id`.

---

## Phase 1 — Early exploration (small sets, original corpus)

| ID | Date | Setup | Result (hit@1 / hit@3 / hit@5 / MRR) | Artifact |
|----|------|-------|--------------------------------------|----------|
| E01 | 2026-08-19 | BM25 on 50-Q set (original corpus) | 0.78 / 0.96 / 0.96 / 0.85 | `retrieval_50_results.json` |
| E02 | 2026-08-17 | 10-Q set: Dense vs Sparse vs Hybrid RRF vs Hybrid Score | Dense 0.90/1.0/1.0/0.95; Sparse 0.50/0.80/0.90/0.675; RRF 0.90/0.90/1.0/0.925; Score 0.90/1.0/1.0/0.933 | `hybrid_benchmark_results.json` |
| E03 | 2026-08-17 | Embedding-model comparison, 10-Q: gemini-embedding-001 vs all-MiniLM-L6-v2 | Gemini 0.90/1.0/1.0/0.95; MiniLM 0.70/0.70/0.80/0.725 | `model_benchmark_results.json` |
| E04 | 2026-08-17 | Top-k tuning (k=1..10) | Best tradeoff k=2 (hit 1.0, P 0.85, 1229 ctx tokens); k≥2 hit saturates | `top_k_tuning_results.json` |
| E05 | 2026-08-17 | Chunk-size ablation (300/40, 600/80, 900/150 tokens) | 3 configs compared; 600/80 kept as default | `chunking_ablation_results.json` |

---

## Phase 2 — 200-question benchmark, ORIGINAL corpus (813 chunks)

`test_set_200.json` built (60/90/30/20 per doc; 158 unique expected chunks; all section/page checks pass).
Benchmark: `benchmark_200.py --corpus original` → `retrieval_200_results.json`.

| ID | Mode | hit@1 | hit@3 | hit@5 | MRR | Latency |
|----|------|------:|------:|------:|----:|--------:|
| E06 | Sparse (BM25Okapi) | 0.580 | 0.760 | 0.870 | 0.686 | 4.7 ms |
| E06 | BM25 + CrossEncoder (ms-marco-MiniLM-L-6-v2) | 0.575 | 0.770 | 0.880 | 0.688 | 1353 ms |
| E06 | Dense (gemini-embedding-001) | 0.345 | 0.560 | 0.650 | 0.456 | 32 ms |
| E06 | Hybrid RRF (c=60) | 0.450 | 0.770 | 0.870 | 0.616 | 34 ms |
| E06 | Hybrid Score Fusion (a=0.6) | 0.450 | 0.685 | 0.800 | 0.582 | 32 ms |
| E06 | Hybrid RRF + CrossEncoder | 0.525 | 0.755 | 0.890 | 0.658 | 1154 ms |

**Finding:** BM25 beat every dense/hybrid/rerank variant on the original corpus. Dense was notably weak (0.345).

---

## Phase 3 — Reranker analysis (original corpus)

| ID | Analysis | Result |
|----|----------|--------|
| E07 | Reranker ceiling: is correct chunk in BM25 top-k? | top-5: 87%, top-10: 94%, top-20: 97%, never: 3% |
| E07 | Rerank lift/regress on original corpus | MiniLM-CE lifted 12 queries to @1, regressed 13 (net ~0) |

---

## Phase 4 — Corpus deduplication

| ID | Step | Detail |
|----|------|--------|
| E08 | Duplication audit (exact normalized-text groups) | 813 chunks → 435 unique texts; **126 duplicate groups; 504 chunks involved; 378 redundant (46%)**; 0 cross-document groups; pattern = same text block assigned to many consecutive ESC sections (e.g., sec_079–087) |
| E09 | Rebuild dedup corpus | `validated_chunks_dedup.json` (435), `embeddings_dedup.json`, `chunk_remap_dedup.json` (378 mappings), Qdrant collection `hypertension_guidelines_dedup` (435 pts, 3072-dim cosine, vectors reused from original index) |
| E10 | Remap test set to canonical chunks | `test_set_200_dedup.json`; **73 entries** remapped (expected chunk was a duplicate → lowest-section canonical; section/page/citation updated) |

---

## Phase 5 — 200-question benchmark, DEDUP corpus (435 chunks)

`benchmark_200.py --corpus dedup` → `retrieval_200_dedup_results.json`.

| ID | Mode | hit@1 | hit@3 | hit@5 | MRR | Latency |
|----|------|------:|------:|------:|----:|--------:|
| E11 | Sparse (BM25Okapi) | **0.755** | 0.890 | 0.945 | **0.826** | 2.0 ms |
| E11 | BM25 + CrossEncoder | 0.760 | 0.895 | 0.945 | 0.830 | 1028 ms |
| E11 | Dense (gemini-embedding-001) | 0.495 | 0.730 | 0.805 | 0.616 | 12 ms |
| E11 | Hybrid RRF (c=60) | 0.650 | 0.890 | 0.955 | 0.772 | 14 ms |
| E11 | Hybrid Score Fusion (a=0.6) | 0.630 | 0.870 | 0.955 | 0.753 | 13 ms |
| E11 | Hybrid RRF + CrossEncoder | 0.725 | 0.895 | 0.950 | 0.813 | 926 ms |

**Dedup impact (Δ vs original):** BM25 hit@1 +17.5 pts, MRR +0.14; dense +15 pts; hybrid RRF +20 pts. Deduplication was the single biggest win.

---

## Phase 6 — Ceiling analysis (dedup corpus)

| ID | Analysis | Result |
|----|----------|--------|
| E12 | Chunk-level: correct chunk in BM25 top-k | top-1: 68%, top-3: 86%, top-5: 92%, top-10: 97.5%, top-20: 98%; never: Q016, Q018, Q095, Q107 |
| E13 | Benchmark-scoring (chunk-OR-section) | 49 queries fail @1; **45 recoverable by a perfect reranker**; 4 never in top-10 (Q016, Q018, Q095, Q130) |

---

## Phase 7 — Reranker / dense variants on dedup corpus

| ID | Mode | hit@1 | hit@3 | hit@5 | MRR | Latency | Artifact |
|----|------|------:|------:|------:|----:|--------:|----------|
| E14 | BM25 (baseline) | 0.755 | 0.890 | 0.945 | 0.826 | 2 ms | `medcpt_results.json` |
| E14 | BM25 + MedCPT-Cross-Encoder (NCBI clinical CE) | 0.705 | 0.900 | 0.940 | 0.805 | 5625 ms | `medcpt_results.json` |
| E15 | MedCPT dense (bi-encoder) | 0.385 | 0.605 | 0.700 | 0.505 | 97 ms | `medcpt_stack_results.json` |
| E15 | Hybrid RRF (BM25 + MedCPT dense) | 0.610 | 0.830 | 0.905 | 0.727 | 107 ms | `medcpt_stack_results.json` |
| E15 | Hybrid RRF + MedCPT-CE | 0.700 | 0.905 | 0.955 | 0.804 | 5929 ms | `medcpt_stack_results.json` |
| — | BM25 + bge-reranker-base | **INCOMPLETE** (eval timed out on CPU; bge fully downloaded & weights verified) | — | — | — | — |

**Finding:** even the clinical cross-encoder (MedCPT, PubMedBERT-based, "the only reranker specialized in medical data") loses to plain BM25. Root cause: 512-token truncation vs long guideline sections + fact-specific queries → full-text BM25 wins.
Note: `pritamdeka/S-PubMedBert-MS-MARCO` was rejected as a reranker — it is a **bi-encoder** (classifier weights randomly initialized when loaded as CrossEncoder).

---

## Phase 8 — Answer generation (BM25 dedup, top-5 → Gemini)

`evaluation/eval_generation.py` (resumable `generate`/`judge`/`report`), model `gemini-flash-lite-latest`
(rate-limited ~10 RPM; `gemini-2.5-flash` is deprecated/404, `gemini-flash-latest` hits free-tier 429 quota).
→ `generation_results.json` (200 records: query, answer, expected, sources, judge scores, citation check).

| Metric | Score |
|--------|------:|
| Correctness (0–5) | **4.84** |
| Faithfulness (0–5) | **5.00** |
| Citation validity (0–1, LLM judge) | **0.99** |
| Citations found / auto-valid | 326 / 279 |

By document (correctness): WHO 4.83, ESC 4.84, USPSTF 5.00, CDC 4.65.

**Findings:**
- Perfect grounding — zero hallucination across all 200 answers (faithfulness 5.0/5).
- All low-correctness cases (Q019, Q084, Q095, Q152, Q191, Q015…) are **retrieval** gaps: correct chunk absent from BM25 top-5 → the model **refused cleanly** instead of inventing content.
- The 47 auto-checked "invalid" citations are mostly checker false-positives (child subsection cited vs parent section title retrieved; page offsets); LLM judge scored 0.99.

---

## Key conclusions

1. **BM25 on the dedup corpus is the production retriever**: 75.5% hit@1, 0.826 MRR, ~2 ms/query.
2. **Deduplication (813→435) was the single biggest lever**: +17.5 pts hit@1, +15–20 across all modes.
3. **No dense, hybrid, or reranker variant beats plain BM25** on this corpus (2 embedding models, 3 cross-encoders, 6 fused/rerank combos tested). Guideline content is lexical (BP ranges, drug names, tables) → BM25 is the right tool.
4. **End-to-end quality is strong**: perfect faithfulness, 4.84/5 correctness; remaining errors trace to retrieval misses (≈11 queries), where the model correctly refuses.

## Reproducibility notes

- All benchmark modes share one scorer (`evaluation/benchmark_200.py` `evaluate_mode`), configurable via `--corpus original|dedup`.
- Dense Gemini: batch-embedded queries (100/req) to stay under the 100 req/min free-tier embedding quota.
- Dedup index rebuilt offline by reusing vectors from the original Qdrant collection (no re-embedding cost).