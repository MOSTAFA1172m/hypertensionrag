"""
eval_generation.py

End-to-end answer-generation evaluation on the dedup corpus.

Pipeline per query (Q001-Q200, test_set_200_dedup.json):
  BM25 top-k (dedup corpus)  ->  grounded Gemini generation  ->  LLM-judge
  scoring (correctness / faithfulness / citation validity) + automatic
  citation validation against the retrieved context.

Stages (resumable):
  python evaluation/eval_generation.py generate   # produce answers
  python evaluation/eval_generation.py judge      # score answers
  python evaluation/eval_generation.py report     # print summary table

Output: evaluation/generation_results.json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag.bm25_retriever import BM25Retriever

load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------- config
TEST_SET_PATH = ROOT / "evaluation" / "test_set_200_dedup.json"
CHUNKS_PATH   = ROOT / "data" / "processed" / "validated_chunks_dedup.json"
RESULTS_PATH  = ROOT / "evaluation" / "generation_results.json"

TOP_K = 5
LLM_MODEL = "gemini-flash-lite-latest"
FALLBACKS = ["gemini-flash-lite-latest", "gemini-flash-latest"]
RPM_LIMIT = 10  # free-tier flash quota is ~15 req/min; stay safely under it

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
**Citation:** [Document Name, Section X.Y, Page N] — always include all three when available in the passage metadata. If section is not available, use [Document Name, Page N]. Never cite a bare number like [1] alone.
Use one citation per excerpt. If multiple passages support one recommendation, list each with its own excerpt + citation.
Use bullet points and bold text for readability.

REFUSAL LOGIC — refuse when:
1. No relevant passages were retrieved for the question
2. The retrieved passages only partially address the specific question asked
3. The question falls outside the guidelines' scope entirely
When refusing, state clearly that the available evidence is insufficient, briefly note what the retrieved context does cover, and suggest a rephrasing or consulting a clinician.

Never invent a citation that isn't grounded in the provided context list."""

# ---------------------------------------------------------------- client / rate limit
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
_last_call = [0.0]


def _throttle():
    elapsed = time.time() - _last_call[0]
    if elapsed < 60.0 / RPM_LIMIT:
        time.sleep(60.0 / RPM_LIMIT - elapsed)
    _last_call[0] = time.time()


def _generate(query: str, context: str) -> str:
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
    last_err = None
    for model in FALLBACKS:
        for attempt in range(3):
            try:
                _throttle()
                resp = _client.models.generate_content(
                    model=model,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,
                        max_output_tokens=1536,
                    ),
                    contents=prompt,
                )
                return resp.text
            except Exception as e:
                last_err = e
                wait = 15 * (attempt + 1) if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) else 2 * (attempt + 1)
                time.sleep(wait)
    raise RuntimeError(f"generate failed: {last_err}")


JUDGE_PROMPT = """You are a strict evaluator of a grounded RAG system answering clinical hypertension guideline questions.

You are given:
- QUESTION: the user's question
- CONTEXT: the retrieved guideline passages the assistant was allowed to use
- EXPECTED: the reference answer (from the test set)
- GENERATED: the assistant's answer

Score three dimensions. Be strict and literal:

1. correctness (0-5): Does GENERATED give the same factual answer as EXPECTED? Penalize wrong numbers, thresholds, drug names, statistics, or omissions of the key fact. 5 = fully matches, 0 = contradicts.
2. faithfulness (0-5): Is EVERY claim in GENERATED directly supported by CONTEXT? Penalize invented facts, unsupported numbers, or content from the model's own knowledge. 5 = fully grounded, 0 = hallucinated.
3. citation_ok (0-1): Are proper citations present (format like [Document Name, Section, Page]) AND do they reference documents/sections/pages that exist in CONTEXT? 1 = yes, 0 = no.

Return ONLY a JSON object:
{{"correctness": int, "faithfulness": int, "citation_ok": int, "reason": "one sentence"}}

QUESTION:
{question}

CONTEXT:
{context}

EXPECTED:
{expected}

GENERATED:
{generated}"""


def _judge(rec) -> dict:
    context = rec.get("_context", "")
    prompt = JUDGE_PROMPT.format(
        question=rec["query"], context=context,
        expected=rec["expected_answer"], generated=rec["answer"])
    last_err = None
    for model in FALLBACKS:
        for attempt in range(3):
            try:
                _throttle()
                resp = _client.models.generate_content(
                    model=model,
                    config=types.GenerateContentConfig(
                        temperature=0.0, max_output_tokens=512,
                        response_mime_type="application/json"),
                    contents=prompt,
                )
                return json.loads(resp.text)
            except Exception as e:
                last_err = e
                wait = 15 * (attempt + 1) if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) else 2 * (attempt + 1)
                time.sleep(wait)
    return {"correctness": -1, "faithfulness": -1, "citation_ok": 0, "reason": f"judge failed: {last_err}"}


# ---------------------------------------------------------------- citation auto-check
def _check_citations(answer: str, sources: list) -> dict:
    """Verify each [..] citation's Section/Page appears in the retrieved sources."""
    sects = set()
    pages = set()
    for s in sources:
        m = re.match(r"\s*(\d+(?:\.\d+)*)", s.get("section_title", "").strip())
        if m:
            sects.add(m.group(1))
        pages.add(s.get("page_start"))
        pages.add(s.get("page_end"))
    cites = re.findall(r"\[([^\]]+)\]", answer)
    valid = []
    for c in cites:
        sec_m = re.search(r"Section\s+([\d.]+)", c)
        page_m = re.search(r"Page[s]?\s+(\d+)", c)
        ok_sec = (not sec_m) or (sec_m.group(1).rstrip(".") in sects)
        ok_page = (not page_m) or (int(page_m.group(1)) in pages)
        valid.append(bool(ok_sec and ok_page))
    return {"citations_found": len(cites), "citations_valid": sum(valid),
            "citations_invalid": len(valid) - sum(valid)}


# ---------------------------------------------------------------- helpers
def _load() -> tuple:
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test = json.load(f)
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    return test, chunks


def _build_context(sources: list) -> str:
    blocks = []
    for i, s in enumerate(sources, 1):
        page_str = f"Page {s['page_start']}" if s["page_start"] == s["page_end"] else f"Pages {s['page_start']}-{s['page_end']}"
        section_str = f", Section: {s['section_title']}" if s.get("section_title") else ""
        blocks.append(f"[{i}] Document: {s['document_name']}{section_str} | {page_str}\n{s['text']}")
    return "\n\n---\n\n".join(blocks)


def _load_results() -> dict:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_results(results: dict):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------- stages
def stage_generate():
    test, chunks = _load()
    bm25 = BM25Retriever(chunks=chunks)
    results = _load_results()
    todo = [e for e in test if e["query_id"] not in results or not results[e["query_id"]].get("answer")]
    print(f"total={len(test)} already_done={len(test) - len(todo)} to_generate={len(todo)}")

    for i, e in enumerate(todo, 1):
        hits = bm25.search(e["query"], top_k=TOP_K)
        sources = [{
            "chunk_id": h["chunk_id"], "document_name": h["document_name"],
            "document_id": h["document_id"], "section_title": h["section_title"],
            "page_start": h["page_start"], "page_end": h["page_end"], "text": h["text"],
        } for h in hits]
        context = _build_context(sources)
        answer = _generate(e["query"], context)
        results[e["query_id"]] = {
            "query": e["query"], "expected_answer": e["answer"],
            "expected_chunk_ids": e["expected_chunk_ids"],
            "answer": answer, "_context": context, "sources": sources,
        }
        if i % 10 == 0 or i == len(todo):
            _save_results(results)
        print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(todo)} {e['query_id']} done", flush=True)
    _save_results(results)
    print("generate stage complete.", flush=True)


def stage_judge():
    results = _load_results()
    todo = [qid for qid, r in results.items() if r.get("answer") and "judge" not in r]
    print(f"to_judge={len(todo)}")
    for n, qid in enumerate(todo, 1):
        r = results[qid]
        r["judge"] = _judge(r)
        r["citation_check"] = _check_citations(r["answer"], r["sources"])
        if n % 10 == 0:
            _save_results(results)
        print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(todo)} judged", flush=True)
    _save_results(results)
    print("judge stage complete.", flush=True)


def stage_report():
    results = _load_results()
    judged = {qid: r for qid, r in results.items() if "judge" in r}
    if not judged:
        print("no judged results. run 'judge' stage first.")
        return
    n = len(judged)
    corr = sum(r["judge"]["correctness"] for r in judged.values()) / n
    faith = sum(r["judge"]["faithfulness"] for r in judged.values()) / n
    cit_ok = sum(r["judge"]["citation_ok"] for r in judged.values()) / n
    cit_found = sum(r["citation_check"]["citations_found"] for r in judged.values())
    cit_valid = sum(r["citation_check"]["citations_valid"] for r in judged.values())
    cit_inv = sum(r["citation_check"]["citations_invalid"] for r in judged.values())

    print("=" * 64)
    print("ANSWER-GENERATION EVALUATION (BM25 dedup corpus, top-5)")
    print("=" * 64)
    print(f"queries:               {n}")
    print(f"correctness (0-5):     {corr:.2f}")
    print(f"faithfulness (0-5):    {faith:.2f}")
    print(f"citation valid (0-1):  {cit_ok:.2f}")
    print(f"citations found:       {cit_found} | valid: {cit_valid} | invalid: {cit_inv}")
    print("-" * 64)

    by_doc = {}
    for qid, r in judged.items():
        doc = "CDC" if "cdc" in r["expected_chunk_ids"][0] else (
            "ESC" if "esc" in r["expected_chunk_ids"][0] else (
                "USPSTF" if "screening" in r["expected_chunk_ids"][0] else "WHO"))
        by_doc.setdefault(doc, []).append(r)
    print(f"{'document':<10} {'n':>3} {'correct':>9} {'faithful':>10} {'cit_ok':>7}")
    for doc in ["WHO", "ESC", "USPSTF", "CDC"]:
        rs = by_doc.get(doc, [])
        if not rs:
            continue
        c = sum(r["judge"]["correctness"] for r in rs) / len(rs)
        f = sum(r["judge"]["faithfulness"] for r in rs) / len(rs)
        v = sum(r["judge"]["citation_ok"] for r in rs) / len(rs)
        print(f"{doc:<10} {len(rs):>3} {c:>9.2f} {f:>10.2f} {v:>7.2f}")

    # low scorers
    print("\nlowest correctness:")
    worst = sorted(judged.items(), key=lambda kv: (kv[1]["judge"]["correctness"], kv[1]["judge"]["faithfulness"]))[:8]
    for qid, r in worst:
        print(f"  {qid} c={r['judge']['correctness']} f={r['judge']['faithfulness']} cit={r['judge']['citation_ok']} | {r['query'][:80]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["generate", "judge", "report"])
    args = parser.parse_args()
    {"generate": stage_generate, "judge": stage_judge, "report": stage_report}[args.stage]()