"""The learned, calibrated abstention gate — the CV differentiator.

Pipeline:
  1. Standardize the ~7 backend-independent signals (StandardScaler).
  2. L2 LogisticRegression -> raw P(correct). Coefficients ARE the interview
     narrative ("NLI-entailment dominates, rerank margin second").
  3. Calibrate on a held-out fold: temperature scaling (default) or isotonic.
     Report reliability/ECE before vs after.
  4. Pick tau* = argmax CRAG net_score on dev_calib. Sanity: tau* ~ 0.5, because
     expected score = 2p - 1 so the optimal rule is answer iff p > 0.5.

Serve: predict_confidence(features) -> calibrated p; abstain if p < tau*.

Only sklearn/numpy/joblib — trains in seconds on CPU, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from trustrag.abstain.signals import FEATURE_NAMES, features_to_vector


# --------------------------------------------------------------------------
# Calibrators
# --------------------------------------------------------------------------
class TemperatureScaler:
    """Single-parameter temperature scaling on logits: p = sigmoid(z / T).

    Fit T>0 by minimizing NLL via a 1-D grid + refine. Robust on small
    calibration sets (a few hundred points) where isotonic would overfit.
    """

    def __init__(self):
        self.T = 1.0

    @staticmethod
    def _nll(probs, y):
        eps = 1e-9
        p = np.clip(probs, eps, 1 - eps)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    def fit(self, logits: np.ndarray, y: np.ndarray) -> TemperatureScaler:
        grid = np.concatenate([np.linspace(0.1, 5.0, 99), np.linspace(5.0, 20.0, 31)])
        best_T, best = 1.0, float("inf")
        for T in grid:
            p = 1.0 / (1.0 + np.exp(-logits / T))
            nll = self._nll(p, y)
            if nll < best:
                best, best_T = nll, T
        self.T = float(best_T)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-logits / self.T))


@dataclass
class GateArtifacts:
    scaler: object          # sklearn StandardScaler
    clf: object             # sklearn LogisticRegression
    calibrator: object      # TemperatureScaler | sklearn IsotonicRegression
    calibration: str        # "temperature" | "isotonic"
    threshold: float        # tau*
    feature_names: list[str]


class AbstentionGate:
    def __init__(self, artifacts: GateArtifacts | None = None):
        self.a = artifacts

    # ---- inference -------------------------------------------------------
    def _raw_logits(self, X: np.ndarray) -> np.ndarray:
        Xs = self.a.scaler.transform(X)
        return self.a.clf.decision_function(Xs)

    def predict_confidence(self, feats: dict[str, float]) -> float:
        """features dict -> calibrated P(correct) in [0, 1]."""
        if self.a is None:
            raise RuntimeError("gate not fitted/loaded")
        X = np.array([features_to_vector(feats)], dtype=float)
        return float(self.predict_confidence_batch(X)[0])

    def predict_confidence_batch(self, X: np.ndarray) -> np.ndarray:
        logits = self._raw_logits(X)
        if self.a.calibration == "temperature":
            return self.a.calibrator.transform(logits)
        # isotonic: map raw sigmoid probabilities
        raw_p = 1.0 / (1.0 + np.exp(-logits))
        return self.a.calibrator.predict(raw_p)

    def should_abstain(self, confidence: float) -> bool:
        return confidence < self.a.threshold

    # ---- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.a, path)

    @classmethod
    def load(cls, path: str | Path) -> AbstentionGate:
        return cls(joblib.load(Path(path)))


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_gate(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    *,
    calibration: str = "temperature",
    c: float = 1.0,
    feature_names: list[str] | None = None,
) -> tuple[AbstentionGate, dict]:
    """Fit scaler+LR on the fit fold, calibrate + pick tau* on the calib fold.

    Returns (gate, info) where info has coefficients, tau*, and best net_score.
    Imported lazily so this module is only heavy when actually training.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from eval.metrics import net_score_vs_threshold

    feature_names = feature_names or list(FEATURE_NAMES)
    scaler = StandardScaler().fit(X_fit)
    # L2 is the LogisticRegression default; passing penalty= is deprecated in
    # newer sklearn, so we rely on the default and tune only C.
    clf = LogisticRegression(C=c, max_iter=1000)
    clf.fit(scaler.transform(X_fit), y_fit)

    logits_calib = clf.decision_function(scaler.transform(X_calib))

    if calibration == "temperature":
        calibrator = TemperatureScaler().fit(logits_calib, y_calib)
        conf_calib = calibrator.transform(logits_calib)
    elif calibration == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        raw_p = 1.0 / (1.0 + np.exp(-logits_calib))
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_p, y_calib)
        conf_calib = calibrator.predict(raw_p)
    else:
        raise ValueError(f"unknown calibration: {calibration}")

    # tau* maximizes CRAG net score on the calibration fold.
    sweep = net_score_vs_threshold(conf_calib, y_calib)

    artifacts = GateArtifacts(
        scaler=scaler, clf=clf, calibrator=calibrator,
        calibration=calibration, threshold=sweep.tau_star,
        feature_names=feature_names,
    )
    info = {
        "coefficients": dict(zip(feature_names, clf.coef_[0].tolist())),
        "intercept": float(clf.intercept_[0]),
        "tau_star": sweep.tau_star,
        "best_net_score_calib": sweep.best_net_score,
        "calibration": calibration,
    }
    return AbstentionGate(artifacts), info


# --------------------------------------------------------------------------
# Baseline gates (for the ablation that proves the learned gate's value)
# --------------------------------------------------------------------------
def naive_threshold_gate(signal_values: np.ndarray, y: np.ndarray):
    """Baseline B2: single-signal threshold (e.g. on rerank_top1).

    Picks the threshold maximizing net score. Returns (tau, net_score, confidences)
    where 'confidences' is just the min-max-normalized raw signal for curves.
    """
    from eval.metrics import net_score_vs_threshold
    sig = np.asarray(signal_values, dtype=float)
    lo, hi = np.min(sig), np.max(sig)
    conf = (sig - lo) / (hi - lo) if hi > lo else np.zeros_like(sig)
    sweep = net_score_vs_threshold(conf, y)
    return sweep.tau_star, sweep.best_net_score, conf
