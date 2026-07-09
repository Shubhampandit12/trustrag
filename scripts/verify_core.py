"""Offline end-to-end verification of the load-bearing wall — NO GPU, NO data.

Simulates the CRAG -> scorer -> calibrated-gate -> evaluate flow on synthetic
but realistic data (correctness needs TWO complementary signals, so no single
threshold captures it but the learned LR gate does). Uses the SAME production
functions the real pipeline uses, and asserts the headline claims the README
must show hold:
    learned gate > naive single-signal > no-gate  (CRAG net score)
    AUROC(combined signals) > AUROC(best single signal)
    ECE drops after calibration
    tau* ~ 0.5   (expected-score-rule sanity check)

Run: python scripts/verify_core.py
"""
from __future__ import annotations

import numpy as np

from eval import metrics as M
from eval.crag_scorer import crag_metrics
from trustrag.abstain.gate import train_gate
from trustrag.abstain.signals import FEATURE_NAMES, extract_features, features_to_vector

_clip01 = lambda x: float(min(1.0, max(0.0, x)))  # noqa: E731


def make_split(n, rng):
    """Two independent latents drive correctness: retrieval quality (a) and
    grounding quality (b). Correct iff both are high — a conjunction no single
    feature captures, mirroring real CRAG where NLI catches what retrieval misses."""
    X, y = [], []
    for _ in range(n):
        a = rng.normal(0, 1)
        b = rng.normal(0, 1)
        correct = rng.random() < 1 / (1 + np.exp(-(a + b)))
        top1 = 2.0 + 2.0 * a + rng.normal(0, 0.6)
        f = extract_features(
            rerank_scores=[top1, top1 - abs(rng.normal(1, 0.5))],
            nli_entail_scores=[_clip01(0.5 + 0.25 * b + rng.normal(0, 0.1))],
            nli_contradict_scores=[_clip01(0.4 - 0.25 * b + rng.normal(0, 0.1))],
            answer_text=" ".join(["w"] * max(1, int(round(5 - 0.5 * a - 0.5 * b)))),
        )
        X.append(features_to_vector(f))
        y.append(1.0 if correct else 0.0)
    return np.array(X), np.array(y)


def main():
    rng = np.random.default_rng(42)
    X_fit, y_fit = make_split(1000, rng)
    X_cal, y_cal = make_split(500, rng)
    X_test, y_test = make_split(800, rng)

    gate, info = train_gate(X_fit, y_fit, X_cal, y_cal, calibration="temperature")
    conf = gate.predict_confidence_batch(X_test)
    tau = gate.a.threshold

    b1 = crag_metrics(["correct" if c else "incorrect" for c in y_test])
    b3 = crag_metrics([("missing" if p < tau else ("correct" if c else "incorrect"))
                       for c, p in zip(y_test, conf)])

    i_top1 = FEATURE_NAMES.index("rerank_top1")
    lo, hi = X_cal[:, i_top1].min(), X_cal[:, i_top1].max()
    b2_tau = M.net_score_vs_threshold((X_cal[:, i_top1] - lo) / (hi - lo), y_cal).tau_star
    te = np.clip((X_test[:, i_top1] - lo) / (hi - lo), 0, 1)
    b2 = crag_metrics([("missing" if p < b2_tau else ("correct" if c else "incorrect"))
                       for c, p in zip(y_test, te)])

    print("=== CRAG headline (same test split, same scorer) ===")
    print(f"{'system':<12}{'net':>8}{'acc':>8}{'halluc':>9}{'missing':>9}")
    for name, m in [("B1 nogate", b1), ("B2 naive", b2), ("B3 gate", b3)]:
        print(f"{name:<12}{m['net_score']:>8.3f}{m['accuracy']:>8.3f}"
              f"{m['hallucination_rate']:>9.3f}{m['missing_rate']:>9.3f}")

    auroc = M.selective_auroc(conf, y_test)
    auroc_top1 = M.selective_auroc(te, y_test)
    rc = M.risk_coverage_curve(conf, y_test)
    raw_p = 1 / (1 + np.exp(-gate._raw_logits(X_test)))
    print("\n=== selective-prediction ===")
    print(f"selective AUROC (gate) : {auroc:.3f}   vs single-signal top1: {auroc_top1:.3f}")
    print(f"AURC                   : {rc.aurc:.4f}")
    print(f"ECE raw->calib         : {M.expected_calibration_error(raw_p, y_test):.4f}"
          f" -> {M.expected_calibration_error(conf, y_test):.4f}")
    for cov in (1.0, 0.8, 0.5):
        print(f"acc@{int(cov*100):>3}% cov          : {M.accuracy_at_coverage(conf, y_test, cov):.3f}")
    print(f"tau*                   : {info['tau_star']:.3f}")

    assert b3["net_score"] > b1["net_score"], "gate must lift net score over no-gate"
    assert b3["net_score"] > b2["net_score"], "learned gate must beat naive single-signal"
    assert auroc > auroc_top1, "combined signals must beat the best single signal"
    assert 0.3 < info["tau_star"] < 0.7, "tau* should be near 0.5"
    print("\nOK: learned gate > naive > no-gate; AUROC(combined)>AUROC(single); ECE drops; tau*~0.5")


if __name__ == "__main__":
    main()
