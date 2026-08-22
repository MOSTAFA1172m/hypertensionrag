"""
cross_encoder_reranker.py

Re-ranks a candidate pool of retrieved chunks using a cross-encoder
(sentence-transformers CrossEncoder). Unlike BM25 (lexical) or bi-encoder
cosine similarity, a cross-encoder scores each (query, chunk) pair jointly,
giving much more accurate relevance estimates at the cost of more compute
per candidate.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (CPU-friendly, 512-token
context). The model is loaded lazily on first use and cached.

Usage:
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, bm25_top10, top_k=5)
"""

from typing import Any

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_LENGTH = 512


class CrossEncoderReranker:
    """Thin wrapper around a sentence-transformers CrossEncoder for reranking."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 max_length: int = MAX_LENGTH):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        """Return a relevance score for each (query, text) pair."""
        if not texts:
            return []
        self._ensure_model()
        pairs = [(query, t) for t in texts]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]

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