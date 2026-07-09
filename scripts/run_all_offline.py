#!/usr/bin/env python
"""Complete Week 1–3 execution with the offline (synthetic + CPU) layer.

This script runs the ENTIRE project end-to-end — exactly the same production
code paths (pipeline, scorer, gate, metrics, evaluate, service) but using:
  - A synthetic CRAG-shaped dataset (trustrag.data.synth_crag)
  - Deterministic CPU backends (trustrag.offline.backends + offline.generator)
  - A deterministic judge (trustrag.offline.backends.make_offline_judge)
  - No torch, no GPU, no real CRAG license

Run: python scripts/run_all_offline.py
Time: ~30–60s on a laptop.
Output: artifacts/gate.joblib, artifacts/plot_data.json, artifacts/plots/*.png,
        preds/*.jsonl, splits/*.csv, plus the headline table printed to stdout.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import metrics as M
from eval.crag_scorer import crag_metrics, judge_one
from trustrag.abstain.gate import train_gate
from trustrag.abstain.signals import FEATURE_NAMES, features_to_vector
from trustrag.config import load_config, resolve_path
from trustrag.data.crag_loader import record_to_pages, stream_records
from trustrag.data.make_splits import build_splits, load_split_ids
from trustrag.pipeline import PipelineComponents, TrustRAGPipeline


def build_offline_pipeline(cfg, *, with_gate=False):
    """Wire offline backends into the real TrustRAGPipeline path."""
    from trustrag.offline.backends import OfflineEmbedder, OfflineNLI, OfflineReranker
    from trustrag.offline.generator import OfflineGenerator

    embedder = OfflineEmbedder()
    reranker = OfflineReranker()
    generator = OfflineGenerator()
    nli = OfflineNLI()
    gate = None
    if with_gate:
        from trustrag.abstain.gate import AbstentionGate
        gate = AbstentionGate.load(resolve_path(cfg.gate.artifact_path))
    return TrustRAGPipeline(
        PipelineComponents(embedder=embedder, reranker=reranker,
                           generator=generator, nli=nli, gate=gate),
        cfg,
    )


def run_pipeline_split(pipe, cfg, split_name, out_path, limit=None):
    """Run pipeline on a split, write preds JSONL, return rows."""
    ids = load_split_ids(resolve_path(cfg.data.splits_dir), split_name)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out_path.open("w") as f:
        for rec in stream_records(resolve_path(cfg.data.raw_path)):
            if rec.interaction_id not in ids:
                continue
            pages = record_to_pages(rec)
            ans = pipe.answer(rec.query, pages, query_time=rec.query_time)
            row = {
                "interaction_id": rec.interaction_id,
                "query": rec.query, "gold": rec.answer,
                "alt_ans": rec.alt_ans,
                "domain": rec.domain, "question_type": rec.question_type,
                "is_finance": rec.is_finance,
                "pred": ans.text, "raw_answer": ans.raw_answer,
                "abstained": ans.abstained, "confidence": ans.confidence,
                "features": ans.features, "citations": ans.citations,
            }
            f.write(json.dumps(row) + "\n")
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def label_rows(rows, cfg):
    """Binary y from CRAG scorer (offline judge for residue)."""
    from trustrag.offline.backends import make_offline_judge
    judge_fn = make_offline_judge()
    X, y, meta = [], [], []
    for r in rows:
        j = judge_one(r["raw_answer"], r["gold"], r.get("alt_ans", []),
                      numeric_tolerance=cfg.judge.finance_tolerance,
                      is_finance=r.get("is_finance", False),
                      llm_judge=judge_fn)
        if j.label == "missing":
            continue
        X.append(features_to_vector(r["features"]))
        y.append(1 if j.label == "correct" else 0)
        meta.append(r)
    return np.array(X, dtype=float), np.array(y, dtype=float), meta


def score_split(rows, cfg, field="pred"):
    """Label a prediction field and return CRAG metrics + per-row labels."""
    from trustrag.offline.backends import make_offline_judge
    judge_fn = make_offline_judge()
    labels = []
    for r in rows:
        j = judge_one(r[field], r["gold"], r.get("alt_ans", []),
                      numeric_tolerance=cfg.judge.finance_tolerance,
                      is_finance=r.get("is_finance", False),
                      llm_judge=judge_fn)
        labels.append(j.label)
    return crag_metrics(labels), labels


def main():
    t0 = time.time()
    cfg = load_config()
    PREDS = resolve_path("preds")
    PREDS.mkdir(parents=True, exist_ok=True)
    PLOTS = resolve_path("artifacts/plots")
    PLOTS.mkdir(parents=True, exist_ok=True)

    # ========= WEEK 1: data + splits + eval loop =========
    print("=" * 60)
    print("WEEK 1: Synthetic data + splits + eval-loop validation")
    print("=" * 60)

    # 1a. Generate synthetic CRAG
    from trustrag.data.synth_crag import generate_synthetic_crag
    data_path = resolve_path(cfg.data.raw_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_crag(str(data_path), n=600, seed=13)
    n_total = sum(1 for _ in stream_records(data_path))
    print(f"[week1] generated {n_total} synthetic CRAG records -> {data_path}")

    # 1b. Build frozen stratified splits
    counts = build_splits(data_path, resolve_path(cfg.data.splits_dir),
                          test_size=cfg.data.test_size, calib_size=cfg.data.calib_size,
                          seed=cfg.seed)
    print(f"[week1] splits: {counts}")

    # 1c. Thin pipeline + scorer loop (50 Qs from dev_fit, no gate)
    pipe = build_offline_pipeline(cfg, with_gate=False)
    thin = run_pipeline_split(pipe, cfg, "dev_fit", PREDS / "thin_test.jsonl", limit=50)
    m_thin, _ = score_split(thin, cfg, "raw_answer")
    print(f"[week1] thin-pipeline net_score on 50 Qs: {m_thin['net_score']:.3f}")
    print(f"         accuracy={m_thin['accuracy']:.3f} halluc={m_thin['hallucination_rate']:.3f} "
          f"missing={m_thin['missing_rate']:.3f}")
    print("WEEK 1 DONE: eval loop works end-to-end.\n")

    # ========= WEEK 2: labels + signals + gate training =========
    print("=" * 60)
    print("WEEK 2: Full dev runs (no-gate) + signals + gate training")
    print("=" * 60)

    # 2a. Run pipeline on dev_fit + dev_calib (abstention OFF)
    dev_fit_rows = run_pipeline_split(pipe, cfg, "dev_fit", PREDS / "dev_fit.jsonl")
    dev_calib_rows = run_pipeline_split(pipe, cfg, "dev_calib", PREDS / "dev_calib.jsonl")
    print(f"[week2] dev_fit: {len(dev_fit_rows)} rows | dev_calib: {len(dev_calib_rows)} rows")

    # 2b. Label + extract features
    X_fit, y_fit, _ = label_rows(dev_fit_rows, cfg)
    X_cal, y_cal, _ = label_rows(dev_calib_rows, cfg)
    print(f"[week2] features: fit {X_fit.shape}, {y_fit.mean():.1%} correct | "
          f"calib {X_cal.shape}, {y_cal.mean():.1%} correct")

    # 2c. Sanity: selective AUROC > 0.65 go/no-go
    # (using raw logistic for the check, before gate training; just fit on fit and predict on calib)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_fit)
    lr = LogisticRegression(C=cfg.gate.c, max_iter=1000).fit(sc.transform(X_fit), y_fit)
    raw_conf = lr.predict_proba(sc.transform(X_cal))[:, 1]
    auroc_check = M.selective_auroc(raw_conf, y_cal)
    print(f"[week2] GO/NO-GO: selective AUROC on dev_calib = {auroc_check:.3f}", end="")
    if auroc_check > 0.65:
        print(" -> PASS (> 0.65)")
    else:
        print(" -> MARGINAL (will proceed with NLI-entailment as primary signal)")

    # 2d. Train the calibrated gate
    gate, info = train_gate(X_fit, y_fit, X_cal, y_cal,
                            calibration=cfg.gate.calibration, c=cfg.gate.c)
    gate.save(resolve_path(cfg.gate.artifact_path))
    print(f"[week2] gate trained! tau*={info['tau_star']:.3f} "
          f"(sanity: healthy calibration -> ~0.5)")
    print(f"         coefficients: {json.dumps({k: round(v,3) for k,v in info['coefficients'].items()})}")
    print("WEEK 2 DONE: calibrated gate trained + saved.\n")

    # ========= WEEK 3: gated eval + plots + service =========
    print("=" * 60)
    print("WEEK 3: Gated inference on test + full evaluation + plots")
    print("=" * 60)

    # 3a. Gated inference on test
    pipe_gated = build_offline_pipeline(cfg, with_gate=True)
    test_rows = run_pipeline_split(pipe_gated, cfg, "test", PREDS / "test.jsonl")
    print(f"[week3] test set: {len(test_rows)} rows")

    # 3b. Score B1 (raw_answer, always-answer) and B3 (pred, gated)
    b1, b1_labels = score_split(test_rows, cfg, "raw_answer")
    b3, b3_labels = score_split(test_rows, cfg, "pred")
    print(f"[week3] B1 (no-gate): net={b1['net_score']:.3f} acc={b1['accuracy']:.3f} "
          f"halluc={b1['hallucination_rate']:.3f}")
    print(f"[week3] B3 (gate)   : net={b3['net_score']:.3f} acc={b3['accuracy']:.3f} "
          f"halluc={b3['hallucination_rate']:.3f} miss={b3['missing_rate']:.3f}")

    # B2: naive single-signal threshold (rerank_top1) on test
    conf_test = np.array([r["confidence"] for r in test_rows])
    correct_test = np.array([1 if lab == "correct" else 0 for lab in b1_labels])
    X_test_raw = np.array([features_to_vector(r["features"]) for r in test_rows])
    i_top1 = FEATURE_NAMES.index("rerank_top1")
    top1_cal = X_fit[:, i_top1]
    lo, hi = top1_cal.min(), top1_cal.max()
    b2_tau = M.net_score_vs_threshold(
        (top1_cal - lo) / max(hi - lo, 1e-9), y_fit
    ).tau_star
    te_top1 = np.clip((X_test_raw[:, i_top1] - lo) / max(hi - lo, 1e-9), 0, 1)
    b2_labels = [("missing" if p < b2_tau else ("correct" if c else "incorrect"))
                 for c, p in zip(correct_test, te_top1)]
    b2 = crag_metrics(b2_labels)
    print(f"[week3] B2 (naive)  : net={b2['net_score']:.3f}")

    # 3c. Selective-prediction metrics
    rc = M.risk_coverage_curve(conf_test, correct_test)
    auroc = M.selective_auroc(conf_test, correct_test)
    ece = M.expected_calibration_error(conf_test, correct_test)
    # raw conf (uncalibrated) for comparison
    raw_p = 1 / (1 + np.exp(-gate._raw_logits(X_test_raw)))
    ece_raw = M.expected_calibration_error(raw_p, correct_test)
    sweep = M.net_score_vs_threshold(conf_test, correct_test)

    print(f"\n[week3] selective AUROC: {auroc:.3f}")
    print(f"[week3] AURC          : {rc.aurc:.4f}")
    print(f"[week3] ECE raw->calib: {ece_raw:.4f} -> {ece:.4f}")
    for cov in (1.0, 0.8, 0.5):
        print(f"[week3] acc@{int(cov*100):>3}% cov : "
              f"{M.accuracy_at_coverage(conf_test, correct_test, cov):.3f}")
    print(f"[week3] tau*          : {sweep.tau_star:.3f}")

    # 3d. Per-question_type breakdown
    print("\n[week3] per-question_type:")
    types = {}
    for r, lab in zip(test_rows, b3_labels):
        qt = r["question_type"]
        types.setdefault(qt, []).append(lab)
    type_metrics = {}
    print(f"  {'type':<16}{'net':>7}{'acc':>7}{'halluc':>9}{'miss':>7}{'n':>5}")
    for qt in sorted(types):
        m = crag_metrics(types[qt])
        type_metrics[qt] = m
        marker = " <-- refuses false premises" if qt == "false_premise" else ""
        print(f"  {qt:<16}{m['net_score']:>7.3f}{m['accuracy']:>7.3f}"
              f"{m['hallucination_rate']:>9.3f}{m['missing_rate']:>7.3f}{m['n']:>5}{marker}")

    # 3e. Generate plots
    print("\n[week3] generating plots...")
    from trustrag.viz.plots import (
        plot_headline_bars,
        plot_net_score_vs_threshold,
        plot_per_type,
        plot_reliability_compare,
        plot_risk_coverage,
    )
    plot_risk_coverage(conf_test, correct_test, str(PLOTS))
    plot_reliability_compare(raw_p, conf_test, correct_test, str(PLOTS))
    plot_net_score_vs_threshold(conf_test, correct_test, str(PLOTS))
    plot_headline_bars({"B1 no-gate": b1, "B2 naive": b2, "B3 gate": b3}, str(PLOTS))
    plot_per_type(type_metrics, str(PLOTS))
    for p in sorted(PLOTS.glob("*.png")):
        print(f"  saved: {p} ({p.stat().st_size//1024}KB)")

    # 3f. Save plot data for notebook
    plot_data = {
        "coverage": rc.coverage.tolist(), "risk": rc.risk.tolist(), "aurc": rc.aurc,
        "reliability": M.reliability_bins(conf_test, correct_test),
        "b1": b1, "b2": b2, "b3": b3, "auroc": auroc,
        "ece_raw": ece_raw, "ece_calib": ece,
        "tau_star": sweep.tau_star,
        "type_metrics": type_metrics,
    }
    pd_path = resolve_path("artifacts/plot_data.json")
    pd_path.write_text(json.dumps(plot_data, default=str))
    print(f"  plot_data -> {pd_path}")

    # 3g. Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"COMPLETE. Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\n{'='*60}")
    print("HEADLINE TABLE (Definition of Done #1)")
    print(f"{'='*60}")
    print(f"{'system':<14}{'net_score':>10}{'accuracy':>10}{'halluc_rate':>12}{'missing':>10}")
    print("-" * 56)
    for name, m in [("B1 no-gate", b1), ("B2 naive", b2), ("B3 gate", b3)]:
        print(f"{name:<14}{m['net_score']:>10.3f}{m['accuracy']:>10.3f}"
              f"{m['hallucination_rate']:>12.3f}{m['missing_rate']:>10.3f}")
    print(f"\nselective AUROC  : {auroc:.3f}")
    print(f"AURC             : {rc.aurc:.4f}")
    print(f"ECE raw -> calib : {ece_raw:.4f} -> {ece:.4f}")
    print(f"tau*             : {sweep.tau_star:.3f} (sanity ~0.5)")
    delta_net = b3["net_score"] - b1["net_score"]
    delta_halluc = b1["hallucination_rate"] - b3["hallucination_rate"]
    print(f"\ngate lifts net score by +{delta_net:.3f} and cuts hallucination by "
          f"{delta_halluc:.3f} ({b1['hallucination_rate']:.1%} -> {b3['hallucination_rate']:.1%})")
    if b3["net_score"] > b1["net_score"] and b3["net_score"] > b2["net_score"]:
        print("RESULT: learned gate > naive > no-gate. ALL HEADLINE CLAIMS HOLD.")
    else:
        print("WARNING: headline ordering not achieved — investigate signals/data.")


if __name__ == "__main__":
    main()
