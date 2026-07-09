"""End-to-end pipeline smoke test with FAKE heavy components (no torch/GPU).

Proves the one inference path wires together: chunk -> retrieve -> rerank ->
generate -> signals -> gate -> Answer, in both abstention-OFF and gated modes.
"""
from __future__ import annotations

import numpy as np

from trustrag.abstain.gate import train_gate
from trustrag.abstain.signals import extract_features, features_to_vector
from trustrag.config import load_config
from trustrag.pipeline import PipelineComponents, TrustRAGPipeline
from trustrag.schemas import ScoredChunk


class FakeEmbedder:
    def embed_passages(self, texts):
        # deterministic non-zero vectors so FAISS has something to index
        return np.array([[float(len(t) % 7) + 1.0, 1.0] for t in texts], dtype="float32")

    def embed_query(self, query):
        return np.array([[1.0, 1.0]], dtype="float32")


class FakeReranker:
    def rerank(self, query, candidates, top_n=5):
        # score by lexical overlap with the query (crude but deterministic)
        qset = set(query.lower().split())
        scored = []
        for c in candidates:
            overlap = len(qset & set(c.text.lower().split()))
            scored.append(ScoredChunk(chunk=c, score=float(overlap)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_n]


class FakeGenerator:
    def __init__(self, answer):
        self.answer = answer

    def generate(self, messages):
        return self.answer


class FakeNLI:
    def __init__(self, entail, contradict):
        self.entail, self.contradict = entail, contradict

    def score(self, answer, reranked):
        n = len(reranked)
        return [self.entail] * n, [self.contradict] * n


def _clip01(x):
    return float(min(1.0, max(0.0, x)))


def _fitted_gate():
    # realistic OVERLAPPING data so calibrated confidence spreads across (0,1)
    # and tau* lands near 0.5 (a separable toy set saturates confidence to 1.0).
    rng = np.random.default_rng(3)
    X, y = [], []
    for _ in range(500):
        latent = rng.normal(0.0, 1.0)
        correct = rng.random() < 1.0 / (1.0 + np.exp(-1.8 * latent))
        top1 = 2.0 + 2.0 * latent + rng.normal(0, 0.5)
        f = extract_features(
            rerank_scores=[top1, top1 - abs(rng.normal(1.0, 0.5))],
            nli_entail_scores=[_clip01(0.55 + 0.2 * latent + rng.normal(0, 0.08))],
            nli_contradict_scores=[_clip01(0.4 - 0.2 * latent + rng.normal(0, 0.08))],
            answer_text=" ".join(["w"] * max(1, int(round(5 - latent)))),
        )
        X.append(features_to_vector(f))
        y.append(1.0 if correct else 0.0)
    gate, _ = train_gate(np.array(X), np.array(y), np.array(X), np.array(y),
                         calibration="temperature")
    return gate


PAGES = [
    {"page_url": "u1", "page_name": "p1", "text": "Paris is the capital of France and a large city."},
    {"page_url": "u2", "page_name": "p2", "text": "The Eiffel Tower is a landmark in Paris France."},
]


def test_pipeline_abstention_off_always_answers():
    cfg = load_config()
    comps = PipelineComponents(
        embedder=FakeEmbedder(), reranker=FakeReranker(),
        generator=FakeGenerator("Paris"), nli=FakeNLI(0.9, 0.05), gate=None,
    )
    pipe = TrustRAGPipeline(comps, cfg)
    ans = pipe.answer("What is the capital of France?", PAGES)
    assert ans.abstained is False
    assert ans.text == "Paris"
    assert ans.citations                      # cited the retrieved pages
    assert set(ans.features) == set(extract_features(rerank_scores=[1.0]).keys())


def test_pipeline_gated_answers_confident():
    cfg = load_config()
    comps = PipelineComponents(
        embedder=FakeEmbedder(), reranker=FakeReranker(),
        generator=FakeGenerator("Paris"), nli=FakeNLI(0.95, 0.02),
        gate=_fitted_gate(),
    )
    pipe = TrustRAGPipeline(comps, cfg)
    ans = pipe.answer("What is the capital of France?", PAGES)
    assert 0.0 <= ans.confidence <= 1.0
    # strong evidence -> should answer, not abstain
    assert ans.abstained is False


def test_pipeline_gated_abstains_on_no_evidence():
    # No retrievable pages -> sentinel features -> the gate must abstain.
    cfg = load_config()
    comps = PipelineComponents(
        embedder=FakeEmbedder(), reranker=FakeReranker(),
        generator=FakeGenerator("Definitely 42, no doubt."),
        nli=FakeNLI(0.05, 0.9),
        gate=_fitted_gate(),
    )
    pipe = TrustRAGPipeline(comps, cfg)
    ans = pipe.answer("What is the meaning of life?", [])   # empty evidence
    assert ans.abstained is True
    assert ans.reason == "low_confidence"
    assert ans.text == cfg.generate.idk_string


def test_threshold_override_flips_decision():
    cfg = load_config()
    comps = PipelineComponents(
        embedder=FakeEmbedder(), reranker=FakeReranker(),
        generator=FakeGenerator("Paris"), nli=FakeNLI(0.95, 0.02),
        gate=_fitted_gate(),
    )
    pipe = TrustRAGPipeline(comps, cfg)
    # forcing tau=1.0 must abstain regardless of confidence (the demo slider)
    ans = pipe.answer("What is the capital of France?", PAGES, threshold_override=1.0)
    assert ans.abstained is True
