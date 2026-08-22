"""
medcpt_reranker.py

Clinical cross-encoder reranker using NCBI's MedCPT-Cross-Encoder
(PubMedBERT-initialized, trained on 18M PubMed query-article pairs).

Unlike sentence-transformers CrossEncoder models, MedCPT is loaded via
transformers AutoModelForSequenceClassification directly, so this module
implements the same interface as rag.cross_encoder_reranker.CrossEncoderReranker.

Usage:
    reranker = MedCPTReranker()
    reranked = reranker.rerank(query, bm25_top10, top_k=5)
"""

from typing import Any

import torch

MODEL_NAME = "ncbi/MedCPT-Cross-Encoder"
MAX_LENGTH = 512
BATCH_SIZE = 16


class MedCPTReranker:
    """Clinical cross-encoder reranker wrapping NCBI MedCPT-Cross-Encoder."""

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None,
                 max_length: int = MAX_LENGTH, batch_size: int = BATCH_SIZE):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        """Return a relevance score for each (query, text) pair."""
        if not texts:
            return []
        self._ensure_model()
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                enc = self._tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self._model(**enc).logits
                scores.extend(logits.flatten().tolist())
        return scores

    def rerank(self, query: str, chunks: list[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
        """Re-rank a list of chunk dicts by (query, chunk) relevance.

        Mutates each chunk with a 'rerank_score' and updates 'score'.
        Returns the chunks sorted by descending relevance, truncated to top_k.
        """
        if not chunks:
            return []
        texts = [c.get("text", "") for c in chunks]
        scores = self.score_pairs(query, texts)
        for c, s in zip(chunks, scores):
            c["rerank_score"] = s
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        reranked = [chunks[i] for i in order]
        for c in reranked:
            c["score"] = c.get("rerank_score", 0.0)
        if top_k is not None:
            reranked = reranked[:top_k]
        return reranked