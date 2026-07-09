"""Exact CRAG scorer.

Two-stage judge:
  Stage A (rule path, free, deterministic):
    1. abstention regex  -> "missing"  (checked FIRST so IDK is never scored wrong)
    2. normalized exact / numeric match against {answer} ∪ alt_ans -> "correct"
    3. otherwise -> defer to Stage B (or "incorrect" if no judge available)
  Stage B (LLM judge, residue only): see eval/judge.py — cached, pinned, temperature=0.

Scoring (CRAG): correct +1, incorrect -1, missing 0.
    net_score = (n_correct - n_incorrect) / n = accuracy - hallucination_rate  ∈ [-1, 1]
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

# Labels and their CRAG point values.
POINTS = {"correct": 1, "incorrect": -1, "missing": 0}

# Abstention / "I don't know" surface forms. Matched on the normalized answer.
_ABSTAIN_PATTERNS = [
    r"\bi\s*do\s*n['’]?t\s+know\b",
    r"\bi\s+cannot\s+(find|answer|determine)\b",
    r"\bi\s+can['’]?t\s+(find|answer|determine)\b",
    r"\bunable\s+to\s+(find|answer|determine)\b",
    r"\bnot\s+enough\s+(information|evidence|context)\b",
    r"\bno\s+(relevant\s+)?(information|answer|evidence)\b",
    r"\bcannot\s+be\s+(found|determined|answered)\b",
]
_ABSTAIN_RE = re.compile("|".join(_ABSTAIN_PATTERNS), re.IGNORECASE)

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
# Words the normalizer should drop when comparing short factual answers.
_ARTICLES_RE = re.compile(r"\b(the|a|an)\b", re.IGNORECASE)
# Strip all punctuation for text comparison; numeric answers are matched
# separately on the raw string via _extract_number, so this is safe.
_PUNCT_RE = re.compile(r"[^\w\s]")


@dataclass
class Judged:
    label: str          # correct | incorrect | missing
    stage: str          # "rule" | "judge"
    reason: str = ""


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation/articles, collapse whitespace."""
    if s is None:
        return ""
    s = s.strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _ARTICLES_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_abstention(pred: str) -> bool:
    """True if the prediction is an explicit refusal / IDK (empty counts too)."""
    if pred is None or not pred.strip():
        return True
    return bool(_ABSTAIN_RE.search(pred))


def _extract_number(s: str) -> float | None:
    m = _NUM_RE.search(s or "")
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _numeric_match(pred: str, gold: str, tol: float) -> bool:
    """Relative-tolerance numeric match (used for finance-style answers)."""
    p, g = _extract_number(pred), _extract_number(gold)
    if p is None or g is None:
        return False
    if g == 0:
        return abs(p) <= tol
    return abs(p - g) / abs(g) <= tol


def _exact_or_containment(pred: str, gold: str) -> bool:
    np_, ng = normalize_text(pred), normalize_text(gold)
    if not ng:
        return False
    if np_ == ng:
        return True
    # Short gold answers (entities/dates) count if contained as a whole phrase.
    if len(ng) <= 40 and re.search(rf"\b{re.escape(ng)}\b", np_):
        return True
    return False


def rule_label(
    pred: str,
    gold: str,
    alt_answers: Iterable[str] = (),
    *,
    numeric_tolerance: float = 0.01,
    is_finance: bool = False,
) -> str | None:
    """Stage A. Returns 'missing'|'correct' if confident, else None (defer to judge).

    We only *positively* assign 'correct' here; a non-match returns None so the
    LLM judge can catch paraphrases. This keeps the free path high-precision.
    """
    if is_abstention(pred):
        return "missing"
    golds = [gold, *[a for a in alt_answers if a]]
    for g in golds:
        if _exact_or_containment(pred, g):
            return "correct"
        if is_finance and _numeric_match(pred, g, numeric_tolerance):
            return "correct"
    return None


def judge_one(
    pred: str,
    gold: str,
    alt_answers: Iterable[str] = (),
    *,
    numeric_tolerance: float = 0.01,
    is_finance: bool = False,
    llm_judge: Callable[[str, str, list[str]], str] | None = None,
) -> Judged:
    """Full two-stage judgment for one (pred, gold) pair."""
    alt = list(alt_answers)
    label = rule_label(
        pred, gold, alt,
        numeric_tolerance=numeric_tolerance, is_finance=is_finance,
    )
    if label is not None:
        return Judged(label=label, stage="rule")
    # Residue: defer to the LLM judge if wired, else be conservative.
    if llm_judge is None:
        return Judged(label="incorrect", stage="rule", reason="no-judge-fallback")
    verdict = llm_judge(pred, gold, alt)
    if verdict not in POINTS:
        verdict = "incorrect"
    return Judged(label=verdict, stage="judge")


def crag_metrics(labels: list[str]) -> dict:
    """Aggregate CRAG metrics from a list of {correct,incorrect,missing} labels."""
    n = len(labels)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "hallucination_rate": 0.0,
                "missing_rate": 0.0, "net_score": 0.0}
    c = labels.count("correct")
    w = labels.count("incorrect")
    m = labels.count("missing")
    return {
        "n": n,
        "accuracy": c / n,
        "hallucination_rate": w / n,   # confident-wrong rate
        "missing_rate": m / n,         # abstain rate
        "net_score": (c - w) / n,      # == accuracy - hallucination_rate
    }
