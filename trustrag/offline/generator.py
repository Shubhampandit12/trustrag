"""Deterministic extractive "LLM" for offline execution.

Parses the user message produced by build_prompt, extracts context sentences,
and returns a grounded or ungrounded answer depending on evidence strength.
This allows the gate to learn a real signal: strong lexical overlap with
the query -> likely correct; weak overlap + ungrounded guess -> incorrect.
"""
from __future__ import annotations

import hashlib
import re

_STOP = frozenset(
    "the a an is are was were be been being in on at of for to from by with and or "
    "but not that this it its what which who where when how".split()
)


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]


def _hash_int(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def extract_context_and_question(user_msg: str) -> tuple[list[str], str]:
    """Parse the user message from build_prompt into (context_sentences, question)."""
    question = ""
    q_match = re.search(r"Question:\s*(.+?)(?:\n|$)", user_msg)
    if q_match:
        question = q_match.group(1).strip()

    # Context is between "Context:" and "Question:"
    ctx_match = re.search(r"Context:\s*\n(.*?)(?:\nQuestion:)", user_msg, re.DOTALL)
    ctx_text = ctx_match.group(1) if ctx_match else ""

    # Split context blocks by [N] markers, then split into sentences
    blocks = re.split(r"\[\d+\]\s*\(source:[^)]*\)\s*", ctx_text)
    sentences = []
    for block in blocks:
        # Split on sentence boundaries
        for sent in re.split(r"[.!?]+\s*", block):
            sent = sent.strip()
            if len(sent) > 10:
                sentences.append(sent)
    return sentences, question


class OfflineGenerator:
    """Deterministic extractive generator matching Generator.generate(messages)->str."""

    def __init__(self, strong_threshold: float = 0.35, idk_prob: float = 0.15):
        self.strong_threshold = strong_threshold
        self.idk_prob = idk_prob

    def generate(self, messages: list[dict]) -> str:
        """Extract or fabricate an answer from the prompt context."""
        user_msg = ""
        for m in messages:
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        sentences, question = extract_context_and_question(user_msg)
        if not sentences or not question:
            return "I don't know."

        q_tokens = set(_tokenize(question))
        if not q_tokens:
            return "I don't know."

        # Score each sentence by token-overlap with the question
        scored = []
        for sent in sentences:
            s_tokens = set(_tokenize(sent))
            if not s_tokens:
                continue
            overlap = len(q_tokens & s_tokens) / len(q_tokens)
            scored.append((overlap, sent))
        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            return "I don't know."

        best_score, best_sent = scored[0]

        # Deterministic randomness seeded on the question
        h = _hash_int(question)

        if best_score >= self.strong_threshold:
            # Strong evidence: return the best sentence (extractive, grounded)
            # Trim to a concise answer: take the part after removing question tokens
            return self._trim(best_sent, q_tokens)
        else:
            # Weak evidence: decide between IDK and hallucination
            if (h % 100) < int(self.idk_prob * 100):
                return "I don't know."
            else:
                # Generate an ungrounded guess (NOT from context)
                return self._fabricate(question, h)

    def _trim(self, sent: str, q_tokens: set) -> str:
        """Return the sentence trimmed — keeping the factual payload."""
        words = sent.split()
        # Remove leading words that are just query tokens
        start = 0
        for i, w in enumerate(words):
            if w.lower().rstrip(".,;:") in q_tokens:
                start = i + 1
            else:
                break
        trimmed = " ".join(words[start:]) if start < len(words) else sent
        return trimmed[:200].strip() if trimmed else sent[:200]

    def _fabricate(self, question: str, h: int) -> str:
        """Generate a plausible-looking but WRONG answer (the hallucination case)."""
        # Use hash to pick a fake value — recognizably not from context
        fakes = [
            "approximately $3.7 billion",
            "January 15, 2019",
            "the award was given to someone else",
            "it was approximately 42%",
            "the answer is 7",
            "founded in 2005",
            "the record was set in 1998",
            "yes, according to the report",
            "about 2.5 million units",
            "the CEO resigned in March",
        ]
        return fakes[h % len(fakes)]
