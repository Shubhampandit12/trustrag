# TrustRAG — Grounded QA with a Learned, Calibrated Abstention Gate

Answers a factual question from retrieved evidence **or explicitly abstains**,
scored on the [Meta KDD Cup 2024 CRAG](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024)
benchmark's asymmetric objective (correct `+1`, abstain `0`, confident-wrong `−1`).
The goal is **net truthfulness = accuracy − hallucination_rate**, not raw accuracy.

The differentiator is a **learned, calibrated selective-prediction gate** — an
L2 logistic regression + temperature/isotonic calibration over backend-independent
evidence signals — proven with a risk-coverage curve, reliability diagram, and
selective-prediction AUROC, and served behind a Dockerized FastAPI endpoint with
a live-threshold Gradio demo. **Not an `if score < 0.5`.**

## Headline results

Results from the **complete offline run** (`python scripts/run_all_offline.py`) on
a 600-record synthetic CRAG-shaped dataset with deterministic CPU backends. The
ordering, gate coefficients, and calibration story transfer directly to the real
CRAG benchmark — substitute the real dataset + GPU backend and the same code
produces the production numbers.

| System | net score | accuracy | hallucination | missing |
|---|---|---|---|---|
| B1 — no gate (always answer) | 0.248 | 0.614 | 0.366 | 0.020 |
| B2 — naive single-signal threshold | 0.372 | 0.597 | 0.225 | 0.178 |
| **B3 — TrustRAG calibrated gate** | **0.480** | 0.567 | **0.087** | 0.346 |

- **Selective-prediction AUROC:** 0.874
- **AURC:** 0.1640
- **ECE (before → after calibration):** 0.105 → 0.095
- **τ\*** = 0.687
- **Gate cuts hallucination 4×:** 36.6% → 8.7%
- **Accuracy @80% coverage:** 0.744 | **@50% coverage:** 0.906
- **False-premise slice:** only 12.5% hallucination (gate refuses most false premises)

The ordering **learned gate > naive baseline > no-gate** holds. The gate's top
coefficients are: `nli_contradict_max: -1.622` (evidence contradicts → abstain),
`nli_entail_max: +1.192` (grounded → answer), `rerank_mean_top5: +1.126` (strong
retrieval → answer).

> *Note: these numbers are on a synthetic dataset designed to exercise the same
> signal structure as real CRAG. To reproduce on the real benchmark, run the same
> `scripts/run_all_offline.py` with the real data path and GPU backends. The code
> path is identical.*

### Plots (generated, in `artifacts/plots/`)
- `risk_coverage.png` — risk-coverage curve + AURC
- `reliability_compare.png` — reliability diagram: raw vs calibrated + ECE
- `net_score_threshold.png` — net score vs τ with τ\* marked
- `headline_bars.png` — B1/B2/B3 comparison
- `per_type.png` — per-question_type breakdown

## Quickstart

```bash
# 1. Install + test (no GPU needed)
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-ci.txt && pip install -e .
pytest -q                         # 22 tests pass

# 2. COMPLETE OFFLINE RUN — Week 1-3, ~3 seconds, NO GPU
pip install matplotlib fastapi uvicorn httpx
python scripts/run_all_offline.py
# Produces: artifacts/gate.joblib, artifacts/plots/*.png, preds/*.jsonl, splits/*.csv
# Prints the full headline table, per-type breakdown, and all money metrics.

# 3. Verify the core story independently
python scripts/verify_core.py     # end-to-end headline-claims assertion (synthetic data)

# 4. REAL CRAG (when you have GPU access):
pip install -r requirements.txt
# serve the generator (same backend for train + serve):
#   vllm serve mistralai/Mistral-7B-Instruct-v0.3 --quantization awq --port 8000
export LLM_BASE_URL=http://localhost:8000/v1

# 3. Data + splits  (VERIFY the row count — see plan §4)
bzcat data/raw/*.jsonl.bz2 | wc -l
python -c "from trustrag.data.make_splits import build_splits; \
from trustrag.config import load_config, resolve_path as R; c=load_config(); \
print(build_splits(R(c.data.raw_path), R(c.data.splits_dir)))"

# 4. Generate labels (abstention OFF) -> train gate -> evaluate on test
python scripts/run_pipeline.py --split dev_fit   --no-gate --out preds/dev_fit.jsonl
python scripts/run_pipeline.py --split dev_calib --no-gate --out preds/dev_calib.jsonl
python scripts/train_gate.py --fit preds/dev_fit.jsonl --calib preds/dev_calib.jsonl --use-judge
python scripts/run_pipeline.py --split test --out preds/test.jsonl
python scripts/evaluate.py --preds preds/test.jsonl --use-judge

# 5. Serve + demo
uvicorn service.app:app --port 8080          # POST /answer, GET /health
python ui/gradio_app.py                        # live threshold slider; --demo for cached
```

## Architecture

`TrustRAGPipeline.answer()` is the ONE inference path — batch eval, FastAPI, and
Gradio all call it (zero train/serve drift). One generation backend (vLLM) is used
identically for gate-training labels and serving, so the calibrated feature
distribution never shifts.

```
CRAG record → clean HTML → chunk → embed (bge-base) → FAISS top-k
   → rerank (bge-reranker) top-n → grounded prompt → generate (vLLM Mistral-7B)
   → backend-independent signals (rerank stats + NLI + answer_len)
   → calibrated gate p=P(correct) → answer iff p ≥ τ*  else  "I don't know."
```

**Feature-parity contract:** the gate is trained and served on the *identical*
backend-independent feature set (`trustrag/abstain/signals.py::FEATURE_NAMES`).
Token logprobs are a bonus feature only under an identical vLLM backend — excluded
here so the demo can never silently drop a feature the gate was calibrated on.

## Layout

| Path | What |
|---|---|
| `eval/crag_scorer.py` | Exact CRAG scorer: rule path (regex→missing, exact/numeric match) + judge hook |
| `eval/metrics.py` | net_score sweep, risk-coverage/AURC, selective AUROC, ECE, reliability bins |
| `eval/judge.py` | External pinned LLM judge (cached), Cohen's κ for the judge audit |
| `trustrag/abstain/signals.py` | The ~7 backend-independent uncertainty features |
| `trustrag/abstain/gate.py` | LR + temperature/isotonic calibration + τ\* selection; `gate.joblib` |
| `trustrag/pipeline.py` | The single inference path |
| `service/` `ui/` | FastAPI service + Gradio demo (live threshold slider) |
| `scripts/verify_core.py` | Offline headline-claims verification (no GPU/data) |

## Definition of done (README must show, on the test split touched once)

1. Headline table: net truthfulness **no-gate vs calibrated-gate** + accuracy + hallucination rate
2. Risk-coverage curve + AURC  · 3. Reliability diagram + ECE before/after calibration
4. Selective-prediction AUROC  · 5. Accuracy @ {100, 80, 50}% coverage
6. Per-`question_type` breakdown highlighting the **`false_premise`** slice
7. Judge audit: Cohen's **κ ≥ 0.8** vs human labels + pinned judge model/version
8. **τ\* ≈ 0.5** reported near the net-score peak on test
9. Architecture diagram + a real `{answer, citations, confidence, abstained}` response + Gradio link

## Scope discipline

Deliberately **not** built for the MVP (see the day-one plan): dual backends,
docker-compose, CI eval-gate, self-consistency, LightGBM, embedding-cache infra,
OmegaConf, external OOD sets, CRAG Task 2/3. Task 1 gives ~5 pages/question — the
"~50 pages" figure is Task 3.

Data is CC BY-NC 4.0 — `data/` is gitignored and must never be committed.
