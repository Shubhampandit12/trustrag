"""Train the calibrated abstention gate from labeled prediction dumps.

Inputs: dev_fit + dev_calib prediction JSONL (produced with --no-gate). Each row
carries `features` and enough to judge correctness. We label with the CRAG
scorer (rule path + optional external judge), assemble feature matrices, fit the
gate, and persist artifacts/gate.joblib.

Usage:
  python scripts/train_gate.py --fit preds/dev_fit.jsonl --calib preds/dev_calib.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.crag_scorer import judge_one
from trustrag.abstain.gate import train_gate
from trustrag.abstain.signals import features_to_vector
from trustrag.config import load_config, resolve_path


def _label_rows(rows, cfg, use_judge):
    """Return (X, y) where y=1 iff the raw answer is CRAG-correct."""
    judge = None
    if use_judge:
        from eval.judge import make_llm_judge
        judge = make_llm_judge(cache_path=resolve_path(cfg.judge.cache_path))
    X, y = [], []
    for r in rows:
        j = judge_one(
            r["raw_answer"], r["gold"], r.get("alt_ans", []),
            numeric_tolerance=cfg.judge.finance_tolerance,
            is_finance=r.get("is_finance", False),
            llm_judge=judge,
        )
        # 'missing' shouldn't occur here (abstention was OFF) but guard anyway.
        if j.label == "missing":
            continue
        X.append(features_to_vector(r["features"]))
        y.append(1 if j.label == "correct" else 0)
    return np.array(X, dtype=float), np.array(y, dtype=float)


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--use-judge", action="store_true", help="wire the external LLM judge")
    args = ap.parse_args()

    cfg = load_config()
    X_fit, y_fit = _label_rows(_read(resolve_path(args.fit)), cfg, args.use_judge)
    X_cal, y_cal = _label_rows(_read(resolve_path(args.calib)), cfg, args.use_judge)
    print(f"fit: {len(y_fit)} rows ({y_fit.mean():.1%} correct) | "
          f"calib: {len(y_cal)} rows ({y_cal.mean():.1%} correct)")

    gate, info = train_gate(
        X_fit, y_fit, X_cal, y_cal,
        calibration=cfg.gate.calibration, c=cfg.gate.c,
    )
    gate.save(resolve_path(cfg.gate.artifact_path))
    print(json.dumps(info, indent=2))
    print(f"tau* = {info['tau_star']:.3f}  (sanity: healthy calibration => ~0.5)")
    print(f"saved gate -> {resolve_path(cfg.gate.artifact_path)}")


if __name__ == "__main__":
    main()
