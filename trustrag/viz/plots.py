"""Report figures for TrustRAG, rendered offline (matplotlib Agg, no display).

All plotting reuses the metric primitives in :mod:`eval.metrics` — this module
does NOT recompute risk-coverage, ECE, reliability bins, or the net-score sweep.
Each ``plot_*`` function takes an ``out_dir``, writes a single PNG, and returns
the absolute path to that PNG. Rendering is deterministic given identical
inputs (no timestamps, no RNG).

Figures:
  risk_coverage.png       plot_risk_coverage
  reliability.png         plot_reliability
  reliability_compare.png plot_reliability_compare
  net_score_threshold.png plot_net_score_vs_threshold
  headline_bars.png       plot_headline_bars
  per_type.png            plot_per_type
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eval import metrics as M  # noqa: E402

_DPI = 120


def _prep(out_dir, filename: str) -> Path:
    """Make ``out_dir`` and return the full output path for ``filename``."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / filename


def _save(fig, path: Path) -> str:
    """Tight-layout, save at fixed DPI, close the figure, return the path str."""
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    return os.fspath(path)


# --------------------------------------------------------------------------- #
# 1. Risk-coverage curve
# --------------------------------------------------------------------------- #
def plot_risk_coverage(confidence, correct, out_dir) -> str:
    """Risk-coverage curve with AURC in the title (x=coverage, y=risk).

    Sweeps from answering everything (coverage=1) down to the single most
    confident answer. A good gate keeps risk low at low coverage and bends
    upward only as it is forced to answer harder questions.
    """
    rc = M.risk_coverage_curve(confidence, correct)
    path = _prep(out_dir, "risk_coverage.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(rc.coverage, rc.risk, color="#1f77b4", lw=2, label="TrustRAG gate")
    # Baseline risk if we answered everything (constant error rate).
    if len(rc.risk):
        base = float(rc.risk[-1])  # risk at full coverage
        ax.axhline(base, color="#888888", ls="--", lw=1,
                   label=f"no-gate risk = {base:.3f}")
    ax.set_xlabel("coverage (fraction answered)")
    ax.set_ylabel("risk (error rate among answered)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_title(f"Risk-Coverage curve  (AURC = {rc.aurc:.4f}, lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 2. Reliability diagram (single) + before/after compare
# --------------------------------------------------------------------------- #
def _reliability_axes(ax, confidence, correct, n_bins, *, color, label):
    """Draw one reliability polyline + count-weighted markers onto ``ax``.

    Returns the ECE for this confidence array (via eval.metrics).
    """
    bins = M.reliability_bins(confidence, correct, n_bins=n_bins)
    xs, ys, counts = [], [], []
    for mean_conf, acc, cnt in bins:
        if cnt > 0 and not np.isnan(acc):
            xs.append(mean_conf)
            ys.append(acc)
            counts.append(cnt)
    ece = M.expected_calibration_error(confidence, correct, n_bins=n_bins)
    if xs:
        counts_arr = np.asarray(counts, dtype=float)
        # Marker area scales with bin population so sparse bins read as small.
        sizes = 30.0 + 220.0 * (counts_arr / counts_arr.max())
        ax.plot(xs, ys, "-", color=color, lw=1.8, alpha=0.9,
                label=f"{label} (ECE={ece:.3f})")
        ax.scatter(xs, ys, s=sizes, color=color, edgecolors="white",
                   linewidths=0.6, zorder=3)
    return ece


def plot_reliability(confidence, correct, out_dir, n_bins: int = 10) -> str:
    """Reliability diagram: bin mean-confidence vs empirical accuracy.

    The dashed diagonal is perfect calibration; points below it mean the model
    is over-confident, above means under-confident. ECE is in the title.
    """
    path = _prep(out_dir, "reliability.png")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], ls="--", color="#444444", lw=1, label="perfect calibration")
    ece = _reliability_axes(ax, confidence, correct, n_bins,
                            color="#d62728", label="model")
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Reliability diagram  (ECE = {ece:.4f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    return _save(fig, path)


def plot_reliability_compare(conf_raw, conf_calib, correct, out_dir, n_bins: int = 10) -> str:
    """Overlay reliability for raw vs calibrated confidence on the same panel.

    Shows the effect of calibration: the calibrated curve should hug the
    diagonal more tightly (lower ECE) than the raw one.
    """
    path = _prep(out_dir, "reliability_compare.png")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], ls="--", color="#444444", lw=1, label="perfect calibration")
    ece_raw = _reliability_axes(ax, conf_raw, correct, n_bins,
                                color="#7f7f7f", label="raw")
    ece_cal = _reliability_axes(ax, conf_calib, correct, n_bins,
                                color="#2ca02c", label="calibrated")
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Calibration: ECE {ece_raw:.4f} → {ece_cal:.4f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 3. Net score vs threshold
# --------------------------------------------------------------------------- #
def plot_net_score_vs_threshold(confidence, correct, out_dir) -> str:
    """CRAG net score as tau sweeps 0->1, with tau* and a 0.5 reference line.

    Answer iff confidence >= tau else abstain. tau* = argmax net score; the
    vertical reference at 0.5 is where a perfectly-calibrated expected-score
    rule (answer iff p>0.5) would place the threshold.
    """
    sweep = M.net_score_vs_threshold(confidence, correct)
    path = _prep(out_dir, "net_score_threshold.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(sweep.thresholds, sweep.net_scores, color="#1f77b4", lw=2,
            label="net score")
    ax.axvline(sweep.tau_star, color="#d62728", ls="-", lw=1.6,
               label=f"tau* = {sweep.tau_star:.3f}  (net={sweep.best_net_score:.3f})")
    ax.axvline(0.5, color="#888888", ls="--", lw=1,
               label="tau = 0.5 (calibrated ideal)")
    ax.axhline(0.0, color="#000000", lw=0.8, alpha=0.5)
    ax.set_xlabel("threshold tau  (answer iff confidence >= tau)")
    ax.set_ylabel("net score  (correct - wrong) / n")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("CRAG net score vs abstention threshold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", fontsize=9)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 4. Headline grouped bars (B1/B2/B3)
# --------------------------------------------------------------------------- #
def plot_headline_bars(systems: dict, out_dir) -> str:
    """Grouped bars of net_score / accuracy / hallucination_rate per system.

    ``systems`` maps a display name (e.g. "B1 nogate", "B2 heuristic",
    "B3 gate") to a crag_metrics dict as produced by
    ``eval.crag_scorer.crag_metrics``.
    """
    path = _prep(out_dir, "headline_bars.png")
    names = list(systems.keys())
    fields = [
        ("net_score", "net score", "#1f77b4"),
        ("accuracy", "accuracy", "#2ca02c"),
        ("hallucination_rate", "hallucination rate", "#d62728"),
    ]

    n_sys = len(names)
    n_grp = len(fields)
    x = np.arange(n_sys)
    width = 0.8 / n_grp

    fig, ax = plt.subplots(figsize=(1.8 * max(n_sys, 3) + 2, 4.8))
    for gi, (key, label, color) in enumerate(fields):
        vals = [float(systems[nm].get(key, 0.0)) for nm in names]
        offset = (gi - (n_grp - 1) / 2.0) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}",
                        xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3 if v >= 0 else -11),
                        textcoords="offset points",
                        ha="center", fontsize=8)

    ax.axhline(0.0, color="#000000", lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("score")
    ax.set_title("Headline CRAG metrics by system (same split, same scorer)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 5. Per-question-type bars
# --------------------------------------------------------------------------- #
def plot_per_type(rows_by_type: dict, out_dir) -> str:
    """Per-question_type net_score and missing_rate bars.

    ``rows_by_type`` maps a question_type string to a crag_metrics dict. The
    ``false_premise`` type is highlighted (bold label + hatch) because it is
    the case where abstaining/refuting is the correct behaviour.
    """
    path = _prep(out_dir, "per_type.png")
    types = list(rows_by_type.keys())
    net = [float(rows_by_type[t].get("net_score", 0.0)) for t in types]
    miss = [float(rows_by_type[t].get("missing_rate", 0.0)) for t in types]

    x = np.arange(len(types))
    width = 0.4

    fig, ax = plt.subplots(figsize=(1.4 * max(len(types), 3) + 2, 4.8))
    net_colors = ["#1f77b4"] * len(types)
    miss_colors = ["#ff7f0e"] * len(types)
    hatches = [None] * len(types)
    for i, t in enumerate(types):
        if t == "false_premise":
            net_colors[i] = "#9467bd"
            miss_colors[i] = "#c5b0d5"
            hatches[i] = "//"

    net_bars = ax.bar(x - width / 2, net, width, label="net score",
                      color=net_colors)
    miss_bars = ax.bar(x + width / 2, miss, width, label="missing rate",
                       color=miss_colors)
    for bars, hatch_on in ((net_bars, True), (miss_bars, True)):
        if not hatch_on:
            continue
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)

    ax.axhline(0.0, color="#000000", lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    labels = ax.set_xticklabels(types, rotation=30, ha="right")
    for lab, t in zip(labels, types):
        if t == "false_premise":
            lab.set_fontweight("bold")
            lab.set_color("#9467bd")
    ax.set_ylabel("score")
    ax.set_title("Per-question-type net score & missing rate "
                 "(false_premise highlighted)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    return _save(fig, path)
