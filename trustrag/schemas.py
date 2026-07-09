"""Shared dataclasses. No heavy imports here so the eval core stays light."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A text chunk from one retrieved page."""
    text: str
    page_url: str
    page_name: str = ""


@dataclass
class ScoredChunk:
    """A chunk plus a relevance score (dense sim or cross-encoder logit)."""
    chunk: Chunk
    score: float


@dataclass
class Answer:
    """The one output type returned by TrustRAGPipeline.answer()."""
    text: str
    citations: list[str] = field(default_factory=list)   # page_urls placed in the prompt
    confidence: float = 0.0                               # calibrated P(correct) in [0, 1]
    abstained: bool = False
    reason: str = "answered"                              # answered | low_confidence | false_premise
    features: dict[str, float] = field(default_factory=dict)
    raw_answer: str = ""                                  # pre-abstention generation (for gate labels)

    def to_dict(self) -> dict:
        return {
            "answer": self.text,
            "citations": self.citations,
            "confidence": round(self.confidence, 4),
            "abstained": self.abstained,
            "reason": self.reason,
        }
