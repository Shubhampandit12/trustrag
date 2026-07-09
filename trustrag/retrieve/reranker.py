"""Cross-encoder reranking with BAAI/bge-reranker-base.

The reranker logits are the abstention gate's strongest cheap signal, so this
stage does double duty: better context AND uncertainty features.
"""
from __future__ import annotations

from trustrag.schemas import Chunk, ScoredChunk


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[Chunk], top_n: int = 5) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [[query, c.text] for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [ScoredChunk(chunk=c, score=float(s)) for c, s in ranked[:top_n]]
