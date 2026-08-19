# TrustRAG

A retrieval-augmented QA pipeline for the [Meta KDD Cup 2024 CRAG benchmark](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024)
that answers a question from retrieved evidence or abstains, using a small
learned classifier — not a hand-picked confidence threshold — to decide which.

## 1. The problem

CRAG scores answers asymmetrically: correct is +1, a confident wrong answer
is -1, and saying "I don't know" is 0. Under that scoring, a RAG pipeline
that always answers gets punished hard for every hallucination, and the
metric that matters is `net_score = accuracy - hallucination_rate`, not raw
accuracy. So the real engineering problem isn't "generate an answer" — it's
"decide, per question, whether the retrieved evidence actually supports an
answer confident enough to be worth giving."

The obvious first cut is a threshold on some retrieval score ("abstain if
top rerank score < X"). This project instead trains a small logistic
regression over several retrieval/grounding signals, calibrates its output
probability, and picks the abstention threshold by sweeping CRAG's own net
score on a held-out split.

## 2. How it works

```
CRAG record -> clean HTML -> chunk -> embed (bge-base) -> FAISS top-k
   -> rerank (bge-reranker) top-n -> grounded prompt -> generate (vLLM Mistral-7B)
   -> NLI-score the generation against its evidence
   -> 7 backend-independent features -> calibrated gate p = P(correct)
   -> answer if p >= tau*, else "I don't know."
```

`TrustRAGPipeline.answer()` (`trustrag/pipeline.py`) is the single inference
path — the batch evaluator, the FastAPI service, and the Gradio demo all call
the same method. That's a deliberate choice, not an incidental one: the
easiest way to get a silent train/serve mismatch in a system like this is to
have the evaluation script build features one way and the serving code build
them a slightly different way. Routing everything through one function makes
that class of bug structurally impossible instead of something you have to
remember to keep in sync.

The seven gate features live in `trustrag/abstain/signals.py`
(`FEATURE_NAMES`): the top rerank score, the margin between the top two
reranked chunks, mean of the top-5 rerank scores, a coverage count of chunks
above a rerank threshold, max NLI entailment and contradiction between the
generated answer and its evidence, and the answer's token length. None of
them require the generator's token logprobs, which would be a stronger
signal but only under the exact same backend the gate was calibrated on —
deliberately left out so the Gradio demo (or any future backend swap) can't
silently degrade the gate by dropping a feature it was trained on. The
project calls this the "feature-parity contract" and enforces it structurally
by having `extract_features()` be the only thing that produces gate inputs,
at both train and serve time.

The gate itself (`trustrag/abstain/gate.py`) is: `StandardScaler` -> L2
`LogisticRegression` -> temperature (or isotonic) calibration on a held-out
calibration fold -> `tau*` chosen as the threshold that maximizes CRAG net
score on that fold. There's a built-in sanity check here worth knowing about:
under CRAG's scoring, the expected value of answering is `2p - 1`, so a
well-calibrated gate should pick `tau* ≈ 0.5` on its own — if a run comes back
with `tau*` far from 0.5, that's a signal the calibration step did something
wrong, not that 0.5 is a magic number to hardcode.

## 3. The offline evaluation harness — and why it exists

CRAG's real data is CC BY-NC 4.0 licensed (never committed here — see
`.gitignore`), and the real pipeline needs a GPU serving Mistral-7B through
vLLM plus `sentence-transformers`, `faiss`, and `transformers` for the
retrieval/reranking/NLI stack. None of that is available in CI or on a
laptop, which makes it easy to end up unable to test your own evaluation and
calibration logic without spinning up expensive infrastructure first.

`trustrag/offline/backends.py` sidesteps that by implementing torch-free,
deterministic stand-ins for every heavy component: `OfflineEmbedder` is a
hashing bag-of-words encoder with sublinear TF weighting, `OfflineReranker`
is an actual BM25 implementation (not a stub — real IDF, real length
normalization) run over the candidate set, and `OfflineNLI` approximates
entailment as answer/chunk token overlap and contradiction as an on-topic
chunk containing a conflicting number or a negation cue. They're not meant to
be as good as the real bge/DeBERTa models — they're meant to produce
real-valued, non-constant scores driven by the actual text, so that the gate
training code, the calibration code, and the metrics code all run against
inputs with genuine signal and can be verified before anyone pays for a GPU.
`trustrag/data/synth_crag.py` generates a matching synthetic dataset in the
real CRAG schema (bz2 JSONL, same record shape) with four question
difficulties mixed in fixed proportions, so the shape of the eval loop can be
exercised end to end.

## 4. A gotcha this repo currently has

While verifying this README I ran `scripts/run_all_offline.py`, which is
supposed to be the single command that runs the whole offline demo end to
end and prints a headline results table. It fails immediately:

```
ModuleNotFoundError: No module named 'trustrag.data.crag_loader'
```

`trustrag/data/crag_loader.py` and `trustrag/data/make_splits.py` are
imported by both `scripts/run_all_offline.py` and `scripts/run_pipeline.py`
(for `stream_records`, `record_to_pages`, `build_splits`, `load_split_ids`)
but were never actually committed to the repo — `git log` and the working
tree confirm they don't exist. `trustrag/data/synth_crag.py` (data
generation) works fine standalone; it's the loading/splitting step
immediately after it that's missing. Concretely, that means the "complete
offline run" headline table this README previously advertised (specific net
scores, AUROC, ECE numbers) was never actually produced by running that
script — those numbers described what the pipeline was expected to produce,
not what it did produce. That's worth saying plainly rather than repeating
the numbers as if they were measured.

What follows below is only what I could actually run.

## 5. What's actually verified

```
$ pytest -q
......................                                                   [100%]
22 passed
```

The 22 tests cover the CRAG scorer (`tests/test_scorer.py`: abstention
detection, exact/containment/numeric matching, the missing-beats-wrong rule,
Cohen's kappa), the gate and selective-prediction metrics
(`tests/test_metrics_gate.py`: AUROC, risk-coverage/AURC, ECE, the
threshold sweep, gate training + joblib round-trip), and a pipeline smoke
test with fake heavy components (`tests/test_pipeline_smoke.py`) that proves
the chunk -> retrieve -> rerank -> generate -> signal -> gate wiring holds
together in both abstention-off and gated modes.

`scripts/verify_core.py` doesn't touch the broken data-loading path at all —
it builds synthetic feature data directly from two independent latent
factors (retrieval quality and grounding quality; correctness requires both
to be good, which no single feature can capture alone) and runs it through
the real `train_gate`/`crag_metrics`/`eval.metrics` functions. Actual output
from running it:

```
=== CRAG headline (same test split, same scorer) ===
system           net     acc   halluc  missing
B1 nogate      0.000   0.500    0.500    0.000
B2 naive       0.149   0.331    0.182    0.486
B3 gate        0.199   0.311    0.113    0.576

=== selective-prediction ===
selective AUROC (gate) : 0.781   vs single-signal top1: 0.705
AURC                   : 0.2852
ECE raw->calib         : 0.0489 -> 0.0287
acc@100% cov          : 0.500
acc@ 80% cov          : 0.581
acc@ 50% cov          : 0.700
tau*                   : 0.533
```

Three things this actually shows: the ordering `learned gate (0.199) > naive
single-signal threshold (0.149) > always-answer (0.000)` holds on net score;
the gate's combined signal beats the single best feature at ranking
correctness (AUROC 0.781 vs. 0.705), which is the whole point of learning a
combination instead of thresholding one number; and calibration measurably
tightens ECE (0.049 -> 0.029) while `tau*` lands at 0.533, close to the 0.5
the expected-score-rule predicts. None of this says anything about accuracy
on real CRAG questions — it's a synthetic sanity check on the gate-training
and calibration *code*, deliberately constructed so a single feature
provably can't solve it, to prove the learned combination is doing something
a naive threshold can't. Getting real numbers on real CRAG data needs the
loader modules from section 4 plus GPU access for the actual retrieval/
generation/NLI models.

## 6. Running it

```bash
# Install the light stack (no torch/vllm/faiss) and run the tests
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-ci.txt && pip install -e .
pytest -q

# Run the synthetic verification above (no GPU, no data, ~1s)
python scripts/verify_core.py

# For the real CRAG pipeline you need a GPU and the licensed dataset:
pip install -r requirements.txt
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --quantization awq --port 8000
export LLM_BASE_URL=http://localhost:8000/v1
# scripts/run_pipeline.py and scripts/run_all_offline.py both need
# trustrag/data/crag_loader.py + trustrag/data/make_splits.py, which aren't
# in the repo yet (see section 4) — write those first.

# Once a gate is trained and saved to artifacts/gate.joblib:
uvicorn service.app:app --port 8080   # POST /answer, GET /health
python ui/gradio_app.py               # live threshold slider demo
```

## 7. Layout

```
trustrag/
  pipeline.py            single TrustRAGPipeline.answer() inference path
  config.py               loads config.yaml into plain dataclasses
  schemas.py               Chunk / ScoredChunk / Answer dataclasses
  nli.py                    DeBERTa-v3 MNLI entailment/contradiction scoring
  abstain/
    signals.py                the 7 backend-independent gate features
    gate.py                    LR + temperature/isotonic calibration + tau*
  retrieve/
    embedder.py                bge-base embedding + chunking
    faiss_store.py              FAISS index with a pure-numpy fallback
    reranker.py                  bge-reranker cross-encoder
  generate/
    generator.py                vLLM OpenAI-compatible client, disk-cached
    prompt.py                    grounding + false-premise + IDK prompt
  offline/
    backends.py                torch-free BM25/hashing-BoW/lexical-NLI stand-ins
    generator.py                 deterministic offline generator
  data/
    synth_crag.py               synthetic CRAG-schema data generator
eval/
  crag_scorer.py            rule-based scorer + LLM-judge residue path
  metrics.py                  risk-coverage, AURC, selective AUROC, ECE
  judge.py                     external pinned LLM judge, cached, Cohen's kappa
service/app.py            FastAPI: POST /answer, GET /health
ui/gradio_app.py           live abstention-threshold demo
scripts/
  verify_core.py           offline headline-claims check (no GPU/data)
  run_all_offline.py        full offline demo (currently broken — see section 4)
  run_pipeline.py            real-CRAG inference over a split
  train_gate.py               fit + calibrate the gate from prediction logs
  evaluate.py                   score a predictions file with the CRAG scorer
tests/                     22 tests, see section 5
```

Data is CC BY-NC 4.0 — `data/` is gitignored and must never be committed.
