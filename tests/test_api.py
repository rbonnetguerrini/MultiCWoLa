"""Tests for the training-free MultiCWoLa orchestrator (post-hoc + bottleneck)."""

from __future__ import annotations

import numpy as np
import pytest

from multiclass_cwola import MultiCWoLa


def test_post_hoc_recovers_and_aligns(make_mixtures, bayes_posteriors, class_means):
    X, source, y, pi = make_mixtures(seed=1)
    g = bayes_posteriors(X, class_means(X, y, 3), pi)

    ntr = 1800
    model = MultiCWoLa(K=3, simplex="archetypal_corner").fit_posteriors(
        g[:ntr], source[:ntr], y=y[:ntr]
    )
    assert model.class_posteriors_.shape == (ntr, 3)
    assert model.pi_.shape == (6, 3)
    assert model.vertices_.shape == (3, 6)
    # Held-out decode + permutation-aligned accuracy.
    assert model.predict(g[ntr:]).shape == (len(g) - ntr,)
    assert model.score(g[ntr:], y[ntr:]) > 0.8


def test_post_hoc_without_labels_still_fits(make_mixtures, bayes_posteriors, class_means):
    X, source, y, pi = make_mixtures(seed=2)
    g = bayes_posteriors(X, class_means(X, y, 3), pi)
    model = MultiCWoLa(K=3, simplex="archetypal_corner").fit_posteriors(g, source)
    assert model.alignment_ is None
    assert model.class_posteriors_.shape == (len(g), 3)
    assert model.report().worst_status in {"ok", "warn", "fail"}


def _make_bottleneck_outputs(seed, K=3, M=6, n=1200, embed=16):
    import torch

    from multiclass_cwola import BottleneckSimplexHead

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    head = BottleneckSimplexHead(input_dim=embed, num_sources=M, num_classes=K)
    feats = torch.randn(n, embed)
    alpha = head.predict_latent(feats).numpy()
    pi = head.pi().detach().numpy()
    source = rng.integers(0, M, size=n)
    return alpha, pi, source


def test_bottleneck_consumes_extracted_outputs():
    pytest.importorskip("torch")
    alpha, pi, source = _make_bottleneck_outputs(seed=3)
    model = MultiCWoLa(K=3).fit_bottleneck(alpha, pi, source)
    assert model.class_posteriors_.shape == alpha.shape
    assert model.pi_.shape == (6, 3)
    assert model.vertices_.shape == (3, 6)
    # pi_ is row-stochastic.
    assert np.allclose(model.pi_.sum(axis=1), 1.0, atol=1e-5)
    # Reconstructed g is a valid posterior cloud.
    assert model.source_posteriors_.shape == (len(alpha), 6)
    assert model.report().worst_status in {"ok", "warn", "fail"}


def test_predict_proba_rejects_bottleneck():
    pytest.importorskip("torch")
    alpha, pi, source = _make_bottleneck_outputs(seed=4)
    model = MultiCWoLa(K=3).fit_bottleneck(alpha, pi, source)
    with pytest.raises(RuntimeError):
        model.predict_proba(model.source_posteriors_)


def test_plot_writes_expected_files(tmp_path, make_mixtures, bayes_posteriors, class_means):
    X, source, y, pi = make_mixtures(seed=5)
    g = bayes_posteriors(X, class_means(X, y, 3), pi)
    model = MultiCWoLa(K=3, simplex="archetypal_corner").fit_posteriors(g, source, y=y)
    paths = model.plot(tmp_path, latent_labels=y, true_pi=pi)
    for key in ("kspace", "mspace", "pi_hat", "pi_oracle", "mixture_confusion"):
        assert key in paths and paths[key].exists() and paths[key].stat().st_size > 0


def test_not_fitted_raises():
    with pytest.raises(RuntimeError):
        MultiCWoLa(K=3).report()
