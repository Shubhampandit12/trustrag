"""Scorer correctness on hand-built cases — the load-bearing wall."""
from __future__ import annotations

from eval.crag_scorer import (
    crag_metrics,
    is_abstention,
    judge_one,
    normalize_text,
    rule_label,
)
from eval.judge import cohen_kappa


def test_abstention_detection():
    assert is_abstention("I don't know.")
    assert is_abstention("I don’t know")           # curly apostrophe
    assert is_abstention("I cannot find that information.")
    assert is_abstention("")                        # empty = abstain
    assert is_abstention("   ")
    assert not is_abstention("The capital is Paris.")


def test_normalize():
    assert normalize_text("The  Quick, Brown!!") == "quick brown"
    assert normalize_text("Paris.") == "paris"


def test_rule_label_missing_beats_correctness():
    # An IDK is scored 'missing' even if the gold happens to appear in boilerplate.
    assert rule_label("I don't know.", "Paris") == "missing"


def test_rule_label_exact_and_containment():
    assert rule_label("Paris", "Paris") == "correct"
    assert rule_label("The capital is Paris.", "Paris") == "correct"
    assert rule_label("London", "Paris") is None      # defer to judge, not auto-wrong


def test_rule_label_numeric_finance():
    # ±1% tolerance for finance-style numeric answers.
    assert rule_label("$100.5", "100", is_finance=True, numeric_tolerance=0.01) == "correct"
    assert rule_label("$110", "100", is_finance=True, numeric_tolerance=0.01) is None


def test_judge_one_fallback_is_incorrect_without_judge():
    j = judge_one("London", "Paris")
    assert j.label == "incorrect" and j.stage == "rule"


def test_judge_one_uses_llm_for_residue():
    def fake_judge(pred, gold, alts):
        return "correct"
    j = judge_one("The City of Light", "Paris", llm_judge=fake_judge)
    assert j.label == "correct" and j.stage == "judge"


def test_crag_metrics_net_score_identity():
    labels = ["correct"] * 6 + ["incorrect"] * 2 + ["missing"] * 2
    m = crag_metrics(labels)
    assert m["n"] == 10
    assert abs(m["accuracy"] - 0.6) < 1e-9
    assert abs(m["hallucination_rate"] - 0.2) < 1e-9
    assert abs(m["missing_rate"] - 0.2) < 1e-9
    # net_score == accuracy - hallucination_rate
    assert abs(m["net_score"] - (m["accuracy"] - m["hallucination_rate"])) < 1e-9
    assert abs(m["net_score"] - 0.4) < 1e-9


def test_crag_metrics_empty():
    m = crag_metrics([])
    assert m["n"] == 0 and m["net_score"] == 0.0


def test_cohen_kappa_perfect_and_random():
    a = ["correct", "incorrect", "missing", "correct"]
    assert abs(cohen_kappa(a, a) - 1.0) < 1e-9
    # disagreement lowers kappa below 1
    b = ["incorrect", "incorrect", "missing", "correct"]
    assert cohen_kappa(a, b) < 1.0
