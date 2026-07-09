"""Backend-independent uncertainty signals for the abstention gate.

FEATURE-PARITY CONTRACT: these features are computed identically at
gate-training time and at serve time. None of them requires token logprobs,
so the gate never silently degrades when the demo swaps the generation
backend. (Logprob-based features are a *bonus* only under an identical vLLM
backend — deliberately excluded from this MVP feature set.)

Every extractor is a pure function of already-computed retrieval/rerank/NLI
outputs, so this module imports nothing heavy and is fully unit-testable.
"""
from __future__ import annotations

from collections.abc import Sequence

# The ordered feature vector. Order is the correctness contract between
# train and serve — do not reorder without retraining the gate.
FEATURE_NAMES = [
    "rerank_top1",        # best cross-encoder logit (evidence strength)
    "rerank_margin",      # top1 - top2 (how decisive the best evidence is)
    "rerank_mean_top5",   # mean of kept reranked logits (overall support)
    "n_chunks_above_tau",  # coverage: # reranked chunks above rerank_tau
    "nli_entail_max",     # max entailment of answer by any evidence chunk
    "nli_contradict_max",  # max contradiction (evidence refutes the answer)
    "answer_len_tokens",  # crude answer length (very long/short correlate w/ error)
]

# Sentinel used when retrieval is empty (teaches the gate "no evidence -> abstain").
_NO_EVIDENCE = -10.0


def extract_features(
    *,
    rerank_scores: Sequence[float],
    nli_entail_scores: Sequence[float] = (),
    nli_contradict_scores: Sequence[float] = (),
    answer_text: str = "",
    rerank_tau: float = 0.0,
) -> dict[str, float]:
    """Compute the backend-independent feature dict.

    rerank_scores: cross-encoder logits for the kept (reranked) chunks, desc.
    nli_*_scores:  per-chunk entailment/contradiction probs of the answer.
    """
    scores = [float(s) for s in rerank_scores]
    if scores:
        top1 = scores[0]
        top2 = scores[1] if len(scores) > 1 else scores[0]
        margin = top1 - top2
        mean_top5 = sum(scores[:5]) / len(scores[:5])
        n_above = float(sum(1 for s in scores if s >= rerank_tau))
    else:
        top1 = margin = mean_top5 = _NO_EVIDENCE
        n_above = 0.0

    entail_max = max((float(x) for x in nli_entail_scores), default=0.0)
    contradict_max = max((float(x) for x in nli_contradict_scores), default=0.0)
    answer_len = float(len(answer_text.split()))

    return {
        "rerank_top1": top1,
        "rerank_margin": margin,
        "rerank_mean_top5": mean_top5,
        "n_chunks_above_tau": n_above,
        "nli_entail_max": entail_max,
        "nli_contradict_max": contradict_max,
        "answer_len_tokens": answer_len,
    }


def features_to_vector(feats: dict[str, float]) -> list[float]:
    """Dict -> ordered vector following FEATURE_NAMES (fills missing with sentinel)."""
    return [float(feats.get(name, _NO_EVIDENCE)) for name in FEATURE_NAMES]
