"""External LLM judge for the CRAG scorer residue.

MUST be a different model from the generator (avoid self-preference bias).
Every verdict is cached on sha1(query|gold|pred) so re-runs are free and
deterministic. temperature=0, constrained one-token output.

The `openai` client is imported lazily so importing this module (and the whole
eval core) never requires the heavy stack — tests inject a fake judge instead.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

_ALLOWED = {"correct", "incorrect", "missing"}

JUDGE_SYSTEM = """You grade whether a predicted answer to a question is correct, \
given the gold answer(s). Reply with EXACTLY one word: correct, incorrect, or missing.
- "correct": the prediction conveys the same fact as any gold answer (paraphrase/rounding ok).
- "incorrect": the prediction states a different or wrong fact (a confident hallucination).
- "missing": the prediction refuses, says it does not know, or gives no answer.
For a FALSE-PREMISE question, the gold answer rejects the premise; a prediction that \
correctly rejects the premise is "correct", and one that plays along is "incorrect"."""

_FEWSHOTS = [
    ("Q: What is the capital of France?\nGold: Paris\nPrediction: The capital is Paris.", "correct"),
    ("Q: Who painted the Mona Lisa?\nGold: Leonardo da Vinci\nPrediction: Michelangelo.", "incorrect"),
    ("Q: What is the population of Atlantis in 2020?\nGold: Atlantis is a mythical place; the premise is false.\nPrediction: I don't know.", "missing"),
    ("Q: When did the sun orbit the Earth last?\nGold: The premise is false; the Earth orbits the Sun.\nPrediction: The Sun does not orbit the Earth; that premise is incorrect.", "correct"),
]


def cache_key(query: str, gold: str, pred: str, alts: list[str] | None = None) -> str:
    alt_str = "|".join(sorted(alts)) if alts else ""
    raw = f"{query}␟{gold}␟{alt_str}␟{pred}".encode()
    return hashlib.sha1(raw).hexdigest()


class JudgeCache:
    """Append-only JSONL cache of {key: label}."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mem: dict[str, str] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._mem[rec["key"]] = rec["label"]

    def get(self, key: str) -> str | None:
        return self._mem.get(key)

    def put(self, key: str, label: str) -> None:
        if key in self._mem:
            return
        self._mem[key] = label
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({"key": key, "label": label}) + "\n")


def _build_prompt(query: str, gold: str, alts: list[str], pred: str) -> str:
    golds = "; ".join([gold, *alts]) if alts else gold
    shots = "\n\n".join(f"{q}\nVerdict: {a}" for q, a in _FEWSHOTS)
    return (
        f"{shots}\n\n"
        f"Q: {query}\nGold: {golds}\nPrediction: {pred}\nVerdict:"
    )


def make_llm_judge(
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    cache_path: str | Path = "artifacts/judge_cache.jsonl",
    query_getter: Callable[[], str] | None = None,
) -> Callable[[str, str, list[str]], str]:
    """Return a judge fn (pred, gold, alts) -> label. Requires `openai` at call time.

    `query_getter` lets the caller thread the current question in (the scorer's
    judge signature is (pred, gold, alts); the query is supplied via closure).
    """
    base_url = base_url or os.environ.get("JUDGE_BASE_URL")
    model = model or os.environ.get("JUDGE_MODEL")
    api_key = api_key or os.environ.get("JUDGE_API_KEY", "EMPTY")
    cache = JudgeCache(cache_path)
    _client_box: dict = {}

    def _client():
        if "c" not in _client_box:
            from openai import OpenAI  # lazy
            _client_box["c"] = OpenAI(base_url=base_url, api_key=api_key)
        return _client_box["c"]

    def judge(pred: str, gold: str, alts: list[str]) -> str:
        query = query_getter() if query_getter else ""
        key = cache_key(query, gold, pred, alts)
        cached = cache.get(key)
        if cached is not None:
            return cached
        prompt = _build_prompt(query, gold, alts, pred)
        resp = _client().chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1,
        )
        label = resp.choices[0].message.content.strip().lower()
        if label not in _ALLOWED:
            label = "incorrect"
        cache.put(key, label)
        return label

    return judge


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Cohen's kappa between two labelings (for the judge audit, target >= 0.8)."""
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("label lists must be equal, non-empty length")
    cats = sorted(set(labels_a) | set(labels_b))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    n = len(labels_a)
    conf = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        conf[idx[a]][idx[b]] += 1
    po = sum(conf[i][i] for i in range(k)) / n
    row = [sum(conf[i]) for i in range(k)]
    col = [sum(conf[i][j] for i in range(k)) for j in range(k)]
    pe = sum((row[i] / n) * (col[i] / n) for i in range(k))
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)
