"""Tests for the functional primitives (the bring-your-own-model path)."""

from __future__ import annotations

import numpy as np
import pytest


def test_top_level_imports():
    from multiclass_cwola import (  # noqa: F401
        BottleneckSimplexHead,
        MultiCWoLa,
        align_latent_classes,
        calibrate_logits,
        decode_posteriors,
        fit_calibrator,
        fit_simplex,
    )


def test_decode_posteriors_recovers_classes(make_mixtures, bayes_posteriors, class_means):
    from multiclass_cwola import align_latent_classes, decode_posteriors, fit_simplex

    X, source, y, pi = make_mixtures(seed=1)
    g = bayes_posteriors(X, class_means(X, y, 3), pi)

    ntr = 1800
    fit = fit_simplex(g[:ntr], num_vertices=3, config={"method": "archetypal"}, seed=0)
    alpha_eval = decode_posteriors(g[ntr:], fit)
    assert alpha_eval.shape == (len(g) - ntr, 3)
    aligned, _ = align_latent_classes(alpha_eval, y[ntr:])
    assert (aligned.argmax(axis=1) == y[ntr:]).mean() > 0.8


def test_align_latent_classes_resolves_k2_flip():
    from multiclass_cwola import align_latent_classes

    # Build alpha where column 0 is truly class 1 and column 1 is class 0 (a flip).
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    alpha = np.zeros((400, 2))
    alpha[np.arange(400), 1 - y] = 0.9  # peak on the *wrong* column
    alpha[np.arange(400), y] = 0.1
    aligned, mapping = align_latent_classes(alpha, y)
    assert (aligned.argmax(axis=1) == y).mean() == 1.0
    assert mapping == {0: 1, 1: 0}


def test_align_latent_classes_general_permutation_from_subset():
    from multiclass_cwola import align_latent_classes

    rng = np.random.default_rng(1)
    y_full = rng.integers(0, 3, size=600)
    perm = {0: 2, 1: 0, 2: 1}  # predicted -> true
    alpha = np.full((600, 3), 0.05)
    for i, yi in enumerate(y_full):
        pred_col = [p for p, t in perm.items() if t == yi][0]
        alpha[i, pred_col] = 0.9
    idx = np.arange(0, 600, 6)  # small labeled subset
    aligned, mapping = align_latent_classes(alpha, y_full[idx], labeled_index=idx)
    assert mapping == perm
    assert (aligned.argmax(axis=1) == y_full).mean() == 1.0


def test_predict_latent_matches_alpha_no_grad():
    torch = pytest.importorskip("torch")
    from multiclass_cwola import BottleneckSimplexHead

    head = BottleneckSimplexHead(input_dim=12, num_sources=5, num_classes=3)
    feats = torch.randn(64, 12)
    latent = head.predict_latent(feats)
    assert latent.shape == (64, 3)
    assert not latent.requires_grad
    assert torch.allclose(latent, head.alpha(feats))


def test_calibration_smoke():
    from multiclass_cwola import calibrate_logits, fit_calibrator

    rng = np.random.default_rng(0)
    logits = rng.normal(size=(500, 4)) * 3.0
    source = logits.argmax(axis=1)  # easy, overconfident task
    calib = fit_calibrator(logits, source, method="temperature")
    probs = calibrate_logits(logits, calib)
    assert probs.shape == (500, 4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
