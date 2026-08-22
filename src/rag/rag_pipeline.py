"""
rag_pipeline.py

Core RAG logic:
  1. Retrieve via BM25 over the deduplicated corpus (sparse, default)
  2. Build a grounded prompt and call Gemini to generate an answer
  3. Return the answer + source metadata for citation

Retrieval config follows the benchmark conclusions (see evaluation/EXPERIMENTS.md):
BM25 on the dedup corpus (435 chunks) beat every dense/hybrid/rerank variant
(75.5% hit@1, 0.826 MRR, ~2 ms). Modes "dense" and "hybrid" remain available.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

from rag.bm25_retriever import BM25Retriever
from rag.hybrid_retriever import reciprocal_rank_fusion, relative_score_fusion

# ============================================================
# Configuration
# ============================================================

QDRANT_PATH     = "data/qdrant"
COLLECTION_NAME = "hypertension_guidelines_dedup"
EMBED_MODEL     = "gemini-embedding-001"
LLM_MODEL       = "gemini-flash-latest"
TOP_K           = 5   # chunks to retrieve per query
RETRIEVAL_MODE  = "sparse"  # "sparse" (default), "hybrid", "dense"

DEDUP_CHUNKS_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "validated_chunks_dedup.json"

# document_id -> (display name used in citations, pdf filename)
DOC_META = {
    "who_guideline_01": (
        "WHO Guideline for the Pharmacological Treatment of Hypertension in Adults",
        "WHO_guideline_01.pdf",
    ),
    "doc_2024-esc-hypertension": (
        "2024 ESC Guidelines for the management of elevated blood pressure and hypertension",
        "2024-ESC-hypertension.pdf",
    ),
    "doc_hypertension-screening-adults-final-rec-statement": (
        "Screening for Hypertension in Adults: US Preventive Services Task Force Reaffirmation Recommendation Statement",
        "hypertension-screening-adults-final-rec-statement.pdf",
    ),
    "doc_cdc_164016_ds1": (
        "Hypertension Prevalence, Awareness, Treatment, and Control Among Adults — United States, 2017–March 2020",
        "cdc_164016_DS1.pdf",
    ),
}

SYSTEM_PROMPT = """You are the Hypertension Guidelines Assistant, a citation-bound clinical evidence tool for the WHO Guideline for the Pharmacological Treatment of Hypertension in Adults, the 2024 ESC Guidelines for the management of elevated blood pressure and hypertension, the USPSTF hypertension screening recommendation, and CDC hypertension statistics. You are not a general medical advisor.

CONTEXT BOUNDARY
Answer ONLY using the provided context passages below. Do not use outside medical knowledge, training data, or general clinical judgment to fill gaps — even if you believe you know the answer. If it isn't in the retrieved passages, it does not exist for the purposes of this answer.

ALLOWED:
- Paraphrasing retrieved text for clarity
- Combining multiple retrieved passages that address the same question
- Stating your confidence based on how directly the evidence supports the claim
- Saying "I don't have enough information" when appropriate

PROHIBITED:
- Adding facts, thresholds, drug names, or statistics not present in the retrieved text
- Using general medical training knowledge to fill gaps
- Softening or skipping a refusal to seem more helpful
- Guessing dosages, thresholds, or intervals not explicitly stated in context

RESPONSE FORMAT
For every clinical/guideline question, structure your answer as:

**Recommendation:** A short, direct answer in plain language.
**Excerpt:** The exact retrieved text (verbatim, quoted) that supports it.
**Citation:** [Document Name, Section X.Y, Page N] — always include all three when available in the passage metadata. If section is not available in the metadata, use [Document Name, Page N]. Never cite a bare number like [1] alone.

Use one citation per excerpt. If multiple passages support one recommendation, list each with its own excerpt + citation, or combine only when they're clearly the same point — never merge two different sources into a single citation.

Use bullet points and bold text for readability. For general greetings ("hi", "who are you"), skip this structure — respond briefly, introduce yourself, and state what you can help with.

REFUSAL LOGIC — refuse when:
1. No relevant passages were retrieved for the question
2. The retrieved passages only partially address the specific question asked (e.g. they discuss the drug class but not the exact dosage asked about)
3. The question falls outside the guidelines' scope entirely

When refusing, your message must:
1. State clearly that the available evidence is insufficient
2. Briefly note what the retrieved context does cover, so the gap feels transparent
3. Suggest a next step — rephrasing the question, or consulting a clinician directly

Example refusal: "I couldn't find enough information in the indexed guidelines to answer this confidently. The retrieved context covers [topic X], but not [specific ask]. Try rephrasing, or consult a clinician directly."

Never invent a citation that isn't grounded in the provided context list."""


# ============================================================
# Setup
# ============================================================

load_dotenv()

_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise EnvironmentError("GEMINI_API_KEY not found in .env")

_client = genai.Client(api_key=_api_key)
_qdrant_client: QdrantClient | None = None

def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
    return _qdrant_client

_bm25_retriever: BM25Retriever | None = None

def get_bm25_retriever(force_reload: bool = False) -> BM25Retriever:
    global _bm25_retriever
    if _bm25_retriever is None:
        _bm25_retriever = BM25Retriever(chunks_path=DEDUP_CHUNKS_PATH)
    elif force_reload:
        _bm25_retriever.reload(chunks_path=DEDUP_CHUNKS_PATH)
    return _bm25_retriever


# ============================================================
# Data classes
# ============================================================

@dataclass
class Source:
    chunk_id:      str
    document_id:   str
    document_name: str
    pdf_file:      str
    section_title: str
    page_start:    int
    page_end:      int
    score:         float
    text:          str


@dataclass
class RAGResult:
    answer:  str
    sources: list[Source]


# ============================================================
# Pipeline steps
# ============================================================

def _embed_query(query: str) -> list[float]:
    """Embed the user query for retrieval (asymmetric: RETRIEVAL_QUERY)."""
    response = _client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values


def _format_source(item: dict, score: float) -> Source:
    doc_id = item.get("document_id", "who_guideline_01")
    meta = DOC_META.get(doc_id)
    if meta:
        doc_name, pdf_file = meta
    else:
        doc_name = item.get("document_name") or doc_id.replace("_", " ").title()
        pdf_file = item.get("pdf_file") or f"{doc_id}.pdf"
    return Source(
        chunk_id      = item.get("chunk_id", ""),
        document_id   = doc_id,
        document_name = doc_name,
        pdf_file      = pdf_file,
        section_title = item.get("section_title", ""),
        page_start    = item.get("page_start", 0),
        page_end      = item.get("page_end", 0),
        score         = round(score, 4),
        text          = item.get("text", ""),
    )



def _retrieve_dense(query_vector: list[float], top_k: int = TOP_K) -> list[dict]:
    """Search Qdrant for dense vector similarity."""
    qclient = get_qdrant_client()
    hits = qclient.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points


    results = []
    for hit in hits:
        item = dict(hit.payload)
        item["dense_score"] = float(hit.score)
        item["score"] = float(hit.score)
        results.append(item)
    return results


def _retrieve_sparse(query: str, top_k: int = TOP_K) -> list[dict]:
    """Search BM25 index for lexical keyword similarity."""
    retriever = get_bm25_retriever()
    hits = retriever.search(query, top_k=top_k)
    for h in hits:
        h["score"] = h.get("bm25_score", 0.0)
    return hits


def _retrieve(
    query_vector: list[float] | None = None,
    query_text: str = "",
    top_k: int = TOP_K,
    mode: str = RETRIEVAL_MODE,
) -> list[Source]:
    """Hybrid, Dense, or Sparse retrieval returning unified top-k Source objects."""
    if mode == "dense":
        if query_vector is None:
            query_vector = _embed_query(query_text)
        items = _retrieve_dense(query_vector, top_k=top_k)
        return [_format_source(item, item["dense_score"]) for item in items]

    elif mode == "sparse":
        items = _retrieve_sparse(query_text, top_k=top_k)
        return [_format_source(item, item["bm25_score"]) for item in items]

    else:
        # Default: Hybrid retrieval with Reciprocal Rank Fusion
        if query_vector is None:
            query_vector = _embed_query(query_text)
        dense_hits = _retrieve_dense(query_vector, top_k=max(top_k * 2, 10))
        sparse_hits = _retrieve_sparse(query_text, top_k=max(top_k * 2, 10))
        fused_hits = reciprocal_rank_fusion(dense_hits, sparse_hits, top_k=top_k, c=60, dense_weight=0.5)
        return [_format_source(item, item["score"]) for item in fused_hits]


def _build_context(sources: list[Source]) -> str:
    """Format retrieved chunks into a numbered context block with document, section, and page metadata."""
    blocks = []
    for i, s in enumerate(sources, 1):
        page_str = f"Page {s.page_start}" if s.page_start == s.page_end else f"Pages {s.page_start}-{s.page_end}"
        section_str = f", Section: {s.section_title}" if s.section_title else ""
        header = f"[{i}] Document: {s.document_name}{section_str} | {page_str}"
        blocks.append(f"{header}\n{s.text}")
    return "\n\n---\n\n".join(blocks)


FALLBACK_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
]


def _generate(query: str, context: str, history: list[dict] | None = None) -> str:
    """Call Gemini with conversation history and grounded context."""
    # Build conversation turns for multi-turn memory
    history_block = ""
    if history:
        turns = []
        for turn in history[-6:]:  # Keep last 6 turns to avoid token overflow
            turns.append(f"User: {turn['query']}\nAssistant: {turn['answer']}")
        history_block = "\n\n".join(turns) + "\n\n"

    prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"{history_block}"
        f"QUESTION: {query}\n\n"
        f"ANSWER:"
    )
    last_err = None
    for model_name in FALLBACK_MODELS:
        for attempt in range(2):
            try:
                response = _client.models.generate_content(
                    model=model_name,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,
                        max_output_tokens=1024,
                    ),
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                last_err = e
                import time
                time.sleep(1.0)
    raise RuntimeError(f"All LLM models failed. Last error: {last_err}")


# ============================================================
# Public entry point
# ============================================================

def answer(
    query: str,
    mode: str = RETRIEVAL_MODE,
    top_k: int = TOP_K,
    history: list[dict] | None = None,
) -> RAGResult:
    """Run the full RAG pipeline and return an answer with sources."""
    sources     = _retrieve(query_text=query, top_k=top_k, mode=mode)
    context     = _build_context(sources)
    answer_text = _generate(query, context, history=history)
    return RAGResult(answer=answer_text, sources=sources)
