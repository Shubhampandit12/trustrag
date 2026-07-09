"""Selective-prediction metrics + the calibrated gate, on synthetic data (no GPU)."""
from __future__ import annotations

import numpy as np

from eval import metrics as M
from trustrag.abstain.gate import AbstentionGate, train_gate
from trustrag.abstain.signals import FEATURE_NAMES, extract_features, features_to_vector


def test_selective_auroc_perfect_and_random():
    # confidence perfectly separates correct(1) from wrong(0)
    conf = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    corr = np.array([1, 1, 1, 0, 0, 0])
    assert M.selective_auroc(conf, corr) == 1.0
    # random-ish -> around 0.5
    rng = np.random.default_rng(0)
    c = rng.random(2000)
    y = (rng.random(2000) < 0.5).astype(float)
    assert 0.4 < M.selective_auroc(c, y) < 0.6


def test_risk_coverage_monotone_and_aurc_range():
    conf = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    corr = np.array([1, 1, 1, 0, 0, 0])
    rc = M.risk_coverage_curve(conf, corr)
    # most-confident-first: risk stays 0 until we dip into the wrong answers
    assert rc.risk[0] == 0.0
    assert 0.0 <= rc.aurc <= 1.0


def test_accuracy_at_coverage():
    conf = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    corr = np.array([1, 1, 1, 0, 0, 0])
    assert M.accuracy_at_coverage(conf, corr, 0.5) == 1.0   # top 50% all correct
    assert abs(M.accuracy_at_coverage(conf, corr, 1.0) - 0.5) < 1e-9


def test_ece_calibrated_is_low():
    # perfectly calibrated: confidence == empirical accuracy
    rng = np.random.default_rng(1)
    conf = rng.random(5000)
    corr = (rng.random(5000) < conf).astype(float)
    assert M.expected_calibration_error(conf, corr, n_bins=10) < 0.05


def test_net_score_threshold_recovers_optimal():
    # answering only the correct ones should beat answering all
    conf = np.array([0.9, 0.85, 0.8, 0.4, 0.3, 0.2])
    corr = np.array([1, 1, 1, 0, 0, 0])
    sweep = M.net_score_vs_threshold(conf, corr)
    # abstaining on the 3 wrong ones -> 3 correct / 6 = 0.5 net
    assert abs(sweep.best_net_score - 0.5) < 1e-9
    # tau* sits between the wrong-cluster and right-cluster confidences
    assert 0.4 < sweep.tau_star <= 0.8


def test_signals_no_evidence_sentinel():
    feats = extract_features(rerank_scores=[], answer_text="something")
    assert feats["n_chunks_above_tau"] == 0.0
    assert feats["rerank_top1"] < 0          # sentinel signals "no evidence"
    vec = features_to_vector(feats)
    assert len(vec) == len(FEATURE_NAMES)


def test_signals_populated():
    feats = extract_features(
        rerank_scores=[5.0, 2.0, 1.0],
        nli_entail_scores=[0.9, 0.1],
        nli_contradict_scores=[0.05, 0.02],
        answer_text="the answer is four",
        rerank_tau=0.0,
    )
    assert feats["rerank_top1"] == 5.0
    assert feats["rerank_margin"] == 3.0
    assert feats["n_chunks_above_tau"] == 3.0
    assert feats["nli_entail_max"] == 0.9
    assert feats["answer_len_tokens"] == 4.0


def _clip01(x):
    return float(min(1.0, max(0.0, x)))


def _synth_dataset(n, rng):
    """Realistic (OVERLAPPING) gate data.

    A latent 'quality' drives correctness AND all features, so the classes are
    not perfectly separable — confidences spread across (0, 1) and tau* lands
    near 0.5, mirroring real CRAG behavior (unlike a separable toy set, which
    saturates confidence to exactly 1.0 and pushes tau* to the boundary)."""
    X, y = [], []
    for _ in range(n):
        latent = rng.normal(0.0, 1.0)
        p_correct = 1.0 / (1.0 + np.exp(-1.8 * latent))    # stochastic label
        correct = rng.random() < p_correct
        top1 = 2.0 + 2.0 * latent + rng.normal(0, 0.5)
        top2 = top1 - abs(rng.normal(1.0, 0.5))
        entail = _clip01(0.55 + 0.20 * latent + rng.normal(0, 0.08))
        contra = _clip01(0.40 - 0.20 * latent + rng.normal(0, 0.08))
        n_words = max(1, int(round(5 - latent + rng.normal(0, 1))))
        feats = extract_features(
            rerank_scores=[top1, top2],
            nli_entail_scores=[entail],
            nli_contradict_scores=[contra],
            answer_text=" ".join(["w"] * n_words),
        )
        X.append(features_to_vector(feats))
        y.append(1.0 if correct else 0.0)
    return np.array(X), np.array(y)


def test_train_gate_learns_and_calibrates(tmp_path):
    rng = np.random.default_rng(7)
    X_fit, y_fit = _synth_dataset(600, rng)
    X_cal, y_cal = _synth_dataset(300, rng)
    gate, info = train_gate(X_fit, y_fit, X_cal, y_cal, calibration="temperature")

    # the gate separates correct from wrong well above chance on a fresh set
    X_te, y_te = _synth_dataset(400, rng)
    conf = gate.predict_confidence_batch(X_te)
    assert M.selective_auroc(conf, y_te) > 0.7      # informative signal
    assert 0.2 < info["tau_star"] < 0.8             # sane operating point (~0.5)

    # round-trips through joblib
    path = tmp_path / "gate.joblib"
    gate.save(path)
    reloaded = AbstentionGate.load(path)
    feats = extract_features(rerank_scores=[4.0, 1.0], nli_entail_scores=[0.9],
                             nli_contradict_scores=[0.05], answer_text="x y z")
    p = reloaded.predict_confidence(feats)
    assert 0.0 <= p <= 1.0
