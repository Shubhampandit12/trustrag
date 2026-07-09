"""TrustRAGPipeline.answer() — the ONE inference path.

Batch eval, FastAPI, and Gradio all call this method, so there is zero
train/serve drift. Heavy components (embedder, reranker, generator, NLI) are
injected; a lightweight construction helper builds the real ones lazily.

The pipeline can run with abstention OFF (`gate=None`) to produce gate-training
labels, and with a fitted gate to serve. The feature set is identical in both
modes — the feature-parity contract.
"""
from __future__ import annotations

from dataclasses import dataclass

from trustrag.abstain.signals import extract_features
from trustrag.generate.prompt import build_prompt
from trustrag.retrieve.embedder import chunk_text
from trustrag.schemas import Answer, Chunk


@dataclass
class PipelineComponents:
    embedder: object
    reranker: object
    generator: object
    nli: object | None = None
    gate: object | None = None       # AbstentionGate | None (None => answer always)


class TrustRAGPipeline:
    def __init__(self, components: PipelineComponents, cfg):
        self.c = components
        self.cfg = cfg

    def answer(self, query: str, pages: list[dict], query_time: str | None = None,
               threshold_override: float | None = None) -> Answer:
        """pages: [{page_url, page_name, text}]. Returns one Answer."""
        rcfg = self.cfg.retrieve
        gcfg = self.cfg.generate

        # 1. chunk every page
        all_chunks: list[Chunk] = []
        for p in pages:
            all_chunks.extend(chunk_text(
                p.get("text", ""), p.get("page_url", ""), p.get("page_name", ""),
                chunk_tokens=rcfg.chunk_tokens, overlap=rcfg.chunk_overlap,
            ))

        # 2. embed + FAISS retrieve top_k
        reranked = []
        if all_chunks:
            from trustrag.retrieve.faiss_store import InMemoryIndex
            embs = self.c.embedder.embed_passages([c.text for c in all_chunks])
            index = InMemoryIndex(embs, all_chunks)
            q_emb = self.c.embedder.embed_query(query)
            hits = index.search(q_emb, rcfg.top_k)
            cand = [c for c, _ in hits]
            # 3. rerank -> top_n
            reranked = self.c.reranker.rerank(query, cand, top_n=rcfg.top_n)

        # 4. grounded generation
        messages = build_prompt(query, reranked, query_time, idk_string=gcfg.idk_string)
        raw_answer = self.c.generator.generate(messages)

        # 5. backend-independent signals
        rerank_scores = [sc.score for sc in reranked]
        entail, contradict = ([], [])
        if self.c.nli is not None:
            entail, contradict = self.c.nli.score(raw_answer, reranked)
        feats = extract_features(
            rerank_scores=rerank_scores,
            nli_entail_scores=entail,
            nli_contradict_scores=contradict,
            answer_text=raw_answer,
            rerank_tau=rcfg.rerank_tau,
        )
        citations = [sc.chunk.page_url for sc in reranked]

        # 6. gate decision
        if self.c.gate is None:
            # abstention OFF (label-generation mode): always surface the answer.
            return Answer(text=raw_answer, citations=citations, confidence=1.0,
                          abstained=False, reason="answered", features=feats,
                          raw_answer=raw_answer)

        confidence = self.c.gate.predict_confidence(feats)
        tau = threshold_override if threshold_override is not None else self.c.gate.a.threshold
        abstain = confidence < tau
        if abstain:
            return Answer(text=gcfg.idk_string, citations=citations, confidence=confidence,
                          abstained=True, reason="low_confidence", features=feats,
                          raw_answer=raw_answer)
        return Answer(text=raw_answer, citations=citations, confidence=confidence,
                      abstained=False, reason="answered", features=feats,
                      raw_answer=raw_answer)


def build_pipeline(cfg, *, with_gate: bool = True, with_nli: bool = True) -> TrustRAGPipeline:
    """Construct the real (heavy) components from config. Imports are lazy inside
    each component, so this only pulls the GPU stack when actually called."""
    from trustrag.generate.generator import Generator
    from trustrag.retrieve.embedder import Embedder
    from trustrag.retrieve.reranker import Reranker

    embedder = Embedder(cfg.retrieve.embedder_model, cfg.retrieve.query_instruction)
    reranker = Reranker(cfg.retrieve.reranker_model)
    generator = Generator(cache_dir=cfg.service.gen_cache_dir,
                          max_tokens=cfg.generate.max_tokens,
                          temperature=cfg.generate.temperature)
    nli = None
    if with_nli:
        from trustrag.nli import NLIScorer
        nli = NLIScorer(cfg.nli.model)
    gate = None
    if with_gate:
        from trustrag.abstain.gate import AbstentionGate
        from trustrag.config import resolve_path
        gate = AbstentionGate.load(resolve_path(cfg.gate.artifact_path))
    return TrustRAGPipeline(
        PipelineComponents(embedder=embedder, reranker=reranker, generator=generator,
                           nli=nli, gate=gate),
        cfg,
    )
