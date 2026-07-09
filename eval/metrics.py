"""Selective-prediction metrics: risk-coverage, AURC, selective AUROC, ECE,
and the CRAG net-score-by-threshold sweep used to pick tau*.

All functions take plain numpy arrays so this module has no torch dependency.
Convention: `confidence` is the calibrated P(correct); `correct` is a 0/1 mask
of whether the *answer* would have been correct (from the CRAG scorer).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_arrays(confidence, correct):
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=float)
    if conf.shape != corr.shape:
        raise ValueError(f"shape mismatch: {conf.shape} vs {corr.shape}")
    return conf, corr


def selective_auroc(confidence, correct) -> float:
    """AUROC of confidence as a predictor of answer correctness.

    0.5 = confidence is useless (the gate does nothing); higher is better.
    Implemented directly (Mann-Whitney U) to avoid a sklearn import here.
    """
    conf, corr = _as_arrays(confidence, correct)
    pos = conf[corr == 1]
    neg = conf[corr == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(conf, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(conf) + 1)
    # average ranks for ties
    _assign_tie_ranks(conf, ranks)
    sum_pos = ranks[corr == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    auc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _assign_tie_ranks(values: np.ndarray, ranks: np.ndarray) -> None:
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1


@dataclass
class RiskCoverage:
    coverage: np.ndarray   # fraction answered (1.0 -> 0.0)
    risk: np.ndarray       # error rate among answered
    aurc: float            # area under risk-coverage curve (lower better)


def risk_coverage_curve(confidence, correct) -> RiskCoverage:
    """Sort by confidence desc; sweep coverage from all-answered to none.

    risk@coverage = error rate among the top-`coverage` most-confident answers.
    AURC = mean risk across coverage points (lower is better).
    """
    conf, corr = _as_arrays(confidence, correct)
    n = len(conf)
    if n == 0:
        return RiskCoverage(np.array([]), np.array([]), float("nan"))
    order = np.argsort(-conf, kind="mergesort")   # most confident first
    corr_sorted = corr[order]
    cum_correct = np.cumsum(corr_sorted)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = 1.0 - (cum_correct / k)                # error among answered
    aurc = float(np.mean(risk))
    return RiskCoverage(coverage=coverage, risk=risk, aurc=aurc)


def coverage_at_risk(confidence, correct, max_risk: float) -> float:
    """Max coverage achievable while keeping error rate <= max_risk."""
    rc = risk_coverage_curve(confidence, correct)
    ok = rc.coverage[rc.risk <= max_risk]
    return float(ok.max()) if len(ok) else 0.0


def accuracy_at_coverage(confidence, correct, coverage: float) -> float:
    """Accuracy among the top-`coverage` fraction of most-confident answers."""
    conf, corr = _as_arrays(confidence, correct)
    n = len(conf)
    if n == 0:
        return float("nan")
    k = max(1, int(round(coverage * n)))
    order = np.argsort(-conf, kind="mergesort")[:k]
    return float(corr[order].mean())


def expected_calibration_error(confidence, correct, n_bins: int = 10) -> float:
    """ECE: weighted gap between confidence and empirical accuracy per bin."""
    conf, corr = _as_arrays(confidence, correct)
    n = len(conf)
    if n == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        acc = corr[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc - avg_conf)
    return float(ece)


def reliability_bins(confidence, correct, n_bins: int = 10):
    """Per-bin (mean_confidence, empirical_accuracy, count) for reliability diagrams."""
    conf, corr = _as_arrays(confidence, correct)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            out.append((float(conf[mask].mean()), float(corr[mask].mean()), int(mask.sum())))
        else:
            out.append(((lo + hi) / 2, float("nan"), 0))
    return out


@dataclass
class ThresholdSweep:
    thresholds: np.ndarray
    net_scores: np.ndarray
    tau_star: float
    best_net_score: float


def net_score_vs_threshold(confidence, correct) -> ThresholdSweep:
    """Sweep tau; answer iff confidence >= tau, else abstain (missing).

    net_score(tau) = (n_correct_answered - n_wrong_answered) / n_total.
    Abstentions score 0. tau* = argmax net_score. Sanity check: tau* ~ 0.5
    when calibration is healthy (expected-score rule: answer iff p > 0.5).
    """
    conf, corr = _as_arrays(confidence, correct)
    n = len(conf)
    if n == 0:
        return ThresholdSweep(np.array([]), np.array([]), 0.5, 0.0)
    taus = np.unique(np.concatenate([conf, [0.0, 1.0 + 1e-9]]))
    nets = []
    for t in taus:
        answered = conf >= t
        c = float(((corr == 1) & answered).sum())
        w = float(((corr == 0) & answered).sum())
        nets.append((c - w) / n)
    nets = np.asarray(nets)
    best = int(np.argmax(nets))
    return ThresholdSweep(
        thresholds=taus, net_scores=nets,
        tau_star=float(taus[best]), best_net_score=float(nets[best]),
    )
