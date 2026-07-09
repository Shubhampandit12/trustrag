"""NLI entailment/contradiction of the answer against retrieved evidence.

One small CPU forward pass per evidence chunk with a DeBERTa-v3 MNLI model.
Produces the nli_entail_max / nli_contradict_max abstention signals; a high
contradiction (evidence refutes the answer) or uniformly low entailment is a
strong "don't answer" cue and is how false-premise questions get caught.
"""
from __future__ import annotations

from trustrag.schemas import ScoredChunk


class NLIScorer:
    # DeBERTa-v3-mnli label order: [entailment, neutral, contradiction]
    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
        self.model_name = model_name
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            from transformers import pipeline  # lazy
            self._pipe = pipeline("text-classification", model=self.model_name,
                                  top_k=None)
        return self._pipe

    def score(self, answer: str, reranked: list[ScoredChunk]) -> tuple[list[float], list[float]]:
        """Return (entail_probs, contradict_probs), one per evidence chunk."""
        if not reranked or not answer.strip():
            return [], []
        inputs = [{"text": sc.chunk.text, "text_pair": answer} for sc in reranked]
        results = self.pipe(inputs)
        entail, contradict = [], []
        for scores in results:
            d = {r["label"].lower(): r["score"] for r in scores}
            entail.append(float(d.get("entailment", 0.0)))
            contradict.append(float(d.get("contradiction", 0.0)))
        return entail, contradict
