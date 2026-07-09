"""Offline plotting layer for TrustRAG.

Thin matplotlib wrappers that render the paper/report figures from the same
numpy metric primitives used by the eval scripts. Every function saves a PNG
into an ``out_dir`` and returns the saved path. Metrics are imported from
``eval.metrics`` — this module never reimplements them.
"""
from __future__ import annotations

from trustrag.viz.plots import (
    plot_headline_bars,
    plot_net_score_vs_threshold,
    plot_per_type,
    plot_reliability,
    plot_reliability_compare,
    plot_risk_coverage,
)

__all__ = [
    "plot_risk_coverage",
    "plot_reliability",
    "plot_reliability_compare",
    "plot_net_score_vs_threshold",
    "plot_headline_bars",
    "plot_per_type",
]
