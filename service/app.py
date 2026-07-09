"""FastAPI service. Two endpoints: POST /answer, GET /health.

CPU app (embed/rerank/NLI/gate) reaches the GPU generator over LLM_BASE_URL.
The pipeline is built once at startup and shared. One structured JSON log line
per request to stdout (no Prometheus scrape stack — see plan §10).
"""
from __future__ import annotations

import json
import os
import time

from fastapi import FastAPI

from service.models import AnswerRequest, AnswerResponse, Citation
from trustrag.config import load_config

app = FastAPI(title="TrustRAG", version="0.1.0")
_STATE: dict = {}


def _pipeline():
    if "pipeline" not in _STATE:
        cfg = load_config()
        from trustrag.pipeline import build_pipeline
        _STATE["cfg"] = cfg
        _STATE["pipeline"] = build_pipeline(cfg, with_gate=True, with_nli=True)
    return _STATE["pipeline"]


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    t0 = time.perf_counter()
    pipe = _pipeline()
    pages = [{"page_url": "", "page_name": f"doc{i}", "text": d}
             for i, d in enumerate(req.documents or [])]
    ans = pipe.answer(req.query, pages, query_time=req.query_time,
                      threshold_override=req.threshold_override)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    resp = AnswerResponse(
        answer=ans.text,
        abstained=ans.abstained,
        confidence=ans.confidence,
        reason=ans.reason,
        citations=[Citation(page_url=u, page_name="") for u in ans.citations],
        meta={"latency_ms": latency_ms, "n_docs": len(pages),
              "model": os.environ.get(_STATE["cfg"].generate.model_env, "unknown")},
    )
    print(json.dumps({"event": "answer", "query": req.query[:120],
                      "abstained": ans.abstained, "confidence": round(ans.confidence, 3),
                      "latency_ms": latency_ms}))
    return resp
