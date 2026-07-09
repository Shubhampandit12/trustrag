"""Score prediction dumps and print the headline CRAG table + selective metrics.

Compares systems on the SAME split with the SAME scorer:
  B1 no-gate (always answer)  vs  B3 TrustRAG (calibrated gate).
Prints net_score, accuracy, hallucination_rate, missing_rate, plus AURC,
selective AUROC, ECE, and accuracy@{100,80,50}% coverage. Saves the plot data
so notebooks/01_calibration.ipynb can render the risk-coverage + reliability curves.

Usage:
  python scripts/evaluate.py --preds preds/test.jsonl              # served (gated)
  python scripts/evaluate.py --preds preds/test.jsonl --raw-as-b1  # also compute B1 from raw_answer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval import metrics as M
from eval.crag_scorer import crag_metrics, judge_one
from trustrag.config import load_config, resolve_path


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _judge_pred(rows, cfg, field, use_judge):
    judge = None
    if use_judge:
        from eval.judge import make_llm_judge
        judge = make_llm_judge(cache_path=resolve_path(cfg.judge.cache_path))
    labels = []
    for r in rows:
        j = judge_one(r[field], r["gold"], r.get("alt_ans", []),
                      numeric_tolerance=cfg.judge.finance_tolerance,
                      is_finance=r.get("is_finance", False), llm_judge=judge)
        labels.append(j.label)
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--use-judge", action="store_true")
    ap.add_argument("--plot-data", default="artifacts/plot_data.json")
    args = ap.parse_args()

    cfg = load_config()
    rows = _read(resolve_path(args.preds))

    # B3 (served): score the gated prediction `pred`.
    b3_labels = _judge_pred(rows, cfg, "pred", args.use_judge)
    b3 = crag_metrics(b3_labels)

    # B1 (no-gate baseline): score the raw answer (always answered).
    b1_labels = _judge_pred(rows, cfg, "raw_answer", args.use_judge)
    b1 = crag_metrics(b1_labels)

    # selective metrics use confidence vs correctness of the RAW answer
    conf = np.array([r["confidence"] for r in rows], dtype=float)
    correct = np.array([1 if lab == "correct" else 0 for lab in b1_labels], dtype=float)
    rc = M.risk_coverage_curve(conf, correct)

    print("\n=== CRAG headline (same split, same scorer) ===")
    hdr = f"{'system':<10}{'net':>8}{'acc':>8}{'halluc':>9}{'missing':>9}"
    print(hdr)
    print("-" * len(hdr))
    print(f"{'B1 nogate':<10}{b1['net_score']:>8.3f}{b1['accuracy']:>8.3f}"
          f"{b1['hallucination_rate']:>9.3f}{b1['missing_rate']:>9.3f}")
    print(f"{'B3 gate':<10}{b3['net_score']:>8.3f}{b3['accuracy']:>8.3f}"
          f"{b3['hallucination_rate']:>9.3f}{b3['missing_rate']:>9.3f}")

    print("\n=== selective-prediction ===")
    print(f"selective AUROC : {M.selective_auroc(conf, correct):.3f}")
    print(f"AURC            : {rc.aurc:.4f}")
    print(f"ECE             : {M.expected_calibration_error(conf, correct):.4f}")
    for cov in (1.0, 0.8, 0.5):
        print(f"acc@{int(cov*100):>3}% coverage: {M.accuracy_at_coverage(conf, correct, cov):.3f}")

    plot = {
        "coverage": rc.coverage.tolist(), "risk": rc.risk.tolist(), "aurc": rc.aurc,
        "reliability": M.reliability_bins(conf, correct),
        "b1": b1, "b3": b3,
    }
    resolve_path(args.plot_data).write_text(json.dumps(plot))
    print(f"\nplot data -> {resolve_path(args.plot_data)}")


if __name__ == "__main__":
    main()
