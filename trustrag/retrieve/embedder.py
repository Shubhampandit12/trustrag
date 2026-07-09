"""Dense embedding + chunking with BAAI/bge-base-en-v1.5.

sentence-transformers / torch are imported lazily inside the class so importing
this module does not pull in the GPU stack.
"""
from __future__ import annotations

from trustrag.schemas import Chunk


def chunk_text(text: str, page_url: str, page_name: str = "",
               chunk_tokens: int = 512, overlap: int = 64) -> list[Chunk]:
    """Word-approximate chunking (good enough; real token chunking optional).

    ~chunk_tokens words per chunk with `overlap` words of overlap. Kept simple
    and dependency-free so the data path is testable without a tokenizer.
    """
    words = (text or "").split()
    if not words:
        return []
    step = max(1, chunk_tokens - overlap)
    chunks = []
    for start in range(0, len(words), step):
        piece = " ".join(words[start:start + chunk_tokens])
        if piece.strip():
            chunks.append(Chunk(text=piece, page_url=page_url, page_name=page_name))
        if start + chunk_tokens >= len(words):
            break
    return chunks


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5",
                 query_instruction: str = "Represent this sentence for searching relevant passages: "):
        self.model_name = model_name
        self.query_instruction = query_instruction
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_passages(self, texts: list[str]):
        import numpy as np
        if not texts:
            return np.zeros((0, 768), dtype="float32")
        return self.model.encode(texts, normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")

    def embed_query(self, query: str):
        return self.model.encode([self.query_instruction + query],
                                 normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")
