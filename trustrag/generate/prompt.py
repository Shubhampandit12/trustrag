"""Grounding prompt: forces citation, an explicit "I don't know", false-premise
handling, and passes query_time for time-sensitive questions.
"""
from __future__ import annotations

from trustrag.schemas import ScoredChunk

SYSTEM_PROMPT = """You are a careful question-answering assistant. Answer ONLY \
from the provided context. Follow these rules strictly:
1. If the context does not contain enough information to answer, reply EXACTLY: "I don't know."
2. If the question rests on a false premise, say so instead of playing along.
3. Be concise — give the direct factual answer, no preamble.
4. Do not use outside knowledge; ground every claim in the context."""


def build_prompt(query: str, reranked: list[ScoredChunk], query_time: str | None = None,
                 idk_string: str = "I don't know.") -> list[dict]:
    ctx_blocks = []
    for i, sc in enumerate(reranked, 1):
        name = sc.chunk.page_name or sc.chunk.page_url
        ctx_blocks.append(f"[{i}] (source: {name})\n{sc.chunk.text}")
    context = "\n\n".join(ctx_blocks) if ctx_blocks else "(no context retrieved)"
    time_note = f"\nThe question is asked at: {query_time}." if query_time else ""

    user = (
        f"Context:\n{context}\n\n"
        f"Question: {query}{time_note}\n\n"
        f"Answer from the context only. If unsure, reply exactly \"{idk_string}\"."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
