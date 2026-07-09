"""Per-question in-memory index.

Task 1 gives ~5 pages/question -> a few hundred chunks. Exact inner-product
search (IndexFlatIP over L2-normalized vectors = cosine) is correct and fast;
no save/load, no persistent index infra (deliberately — see plan §10).

Uses FAISS when installed; otherwise falls back to a pure-numpy brute-force
that computes the identical exact inner-product top-k. The fallback keeps the
eval core runnable without the heavy stack and is mathematically equivalent
for this problem size.
"""
from __future__ import annotations

import numpy as np


class InMemoryIndex:
    def __init__(self, embeddings, chunks):
        self.chunks = chunks
        self.embeddings = np.asarray(embeddings, dtype="float32")
        self._faiss_index = None
        try:
            import faiss  # lazy; optional
            dim = self.embeddings.shape[1] if len(self.embeddings) else 768
            self._faiss_index = faiss.IndexFlatIP(dim)
            if len(self.embeddings):
                self._faiss_index.add(self.embeddings)
        except Exception:
            self._faiss_index = None  # numpy fallback

    def search(self, query_emb, top_k: int):
        if len(self.chunks) == 0:
            return []
        query_emb = np.asarray(query_emb, dtype="float32")
        k = min(top_k, len(self.chunks))
        if self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(query_emb, k)
            pairs = zip(scores[0], idxs[0])
        else:
            # exact inner-product top-k, identical to IndexFlatIP
            sims = (self.embeddings @ query_emb[0])
            top = np.argsort(-sims)[:k]
            pairs = ((float(sims[i]), int(i)) for i in top)
        out = []
        for score, idx in pairs:
            if idx == -1:
                continue
            out.append((self.chunks[idx], float(score)))
        return out
