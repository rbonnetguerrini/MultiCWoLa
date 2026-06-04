"""Smoke tests for the high-level MultiCWoLa API and diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from multiclass_cwola import MultiCWoLa
from multiclass_cwola.diagnostics import compute_diagnostics


def _make_mixtures(seed: int = 0, K: int = 3, M: int = 6, n_per: int = 400):
    """Synthetic well-separated Gaussian classes drawn into M mixtures."""
    rng = np.random.default_rng(seed)
    means = rng.normal(scale=6.0, size=(K, 8))
    # Row-stochastic mixing matrix with a strong diagonal so anchors exist.
    pi = rng.dirichlet(np.ones(K) * 0.5, size=M)
    pi = 0.5 * pi + 0.5 * np.eye(M, K)
    pi /= pi.sum(axis=1, keepdims=True)

    X, source, y = [], [], []
    for m in range(M):
        classes = rng.choice(K, size=n_per, p=pi[m])
        X.append(means[classes] + rng.normal(scale=1.0, size=(n_per, 8)))
        source.append(np.full(n_per, m))
        y.append(classes)
    return (
        np.concatenate(X).astype(np.float32),
        np.concatenate(source),
        np.concatenate(y),
        pi,
    )


def _bayes_source_posteriors(X, means, pi):
    """Oracle g(x) = P(m|x) for the synthetic generator (uniform mixture prior)."""
    # p(x|k) up to constant.
    log_pk = -0.5 * ((X[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)  # (N, K)
    pk = np.exp(log_pk - log_pk.max(axis=1, keepdims=True))
    qm = pk @ pi.T  # (N, M), uniform mixture prior
    return qm / qm.sum(axis=1, keepdims=True)


def test_precomputed_posteriors_recovers_classes():
    X, source, y, pi = _make_mixtures(seed=1)
    means = np.stack([X[y == k].mean(axis=0) for k in range(3)])
    g = _bayes_source_posteriors(X, means, pi)

    model = MultiCWoLa(K=3, simplex="archetypal_corner").fit_posteriors(g, source, y=y)
    assert model.class_posteriors_.shape == (len(X), 3)
    # Near-oracle posteriors -> classes recover well above chance.
    assert model.score(g, y) > 0.8


def test_internal_mlp_end_to_end():
    X, source, y, _ = _make_mixtures(seed=2)
    model = MultiCWoLa(
        K=3,
        backbone="mlp",
        simplex="archetypal_corner",
        train_kwargs={"epochs": 6},
    ).fit(X, source, y=y)
    assert model.source_posteriors_.shape[1] == 6
    assert model.class_posteriors_.shape == (len(X), 3)
    # Aligned accuracy should clear chance (1/3) by a wide margin.
    assert model.score(X, y) > 0.55


def test_bottleneck_end_to_end():
    X, source, y, _ = _make_mixtures(seed=5)
    model = MultiCWoLa(
        K=3,
        mode="bottleneck",
        backbone="mlp",
        train_kwargs={"epochs": 12, "bottleneck_warmup_epochs": 2},
    ).fit(X, source, y=y)
    # alpha(x) and Pi come straight off the head; no post-hoc simplex fit.
    assert model.fit_ is None
    assert model.class_posteriors_.shape == (len(X), 3)
    assert model.pi_.shape == (6, 3)
    assert model.vertices_.shape == (3, 6)
    assert model.predict(X).shape == (len(X),)
    assert model.score(X, y) > 0.5
    # Diagnostics must still run without a post-hoc fit object.
    assert isinstance(str(model.report()), str)


def test_bottleneck_rejects_external_backbone():
    X, source, y, _ = _make_mixtures(seed=6)
    with pytest.raises(ValueError):
        MultiCWoLa(K=3, mode="bottleneck", backbone="precomputed").fit(X, source, y=y)


def test_estimator_backbone():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    X, source, y, _ = _make_mixtures(seed=3)
    clf = LogisticRegression(max_iter=500).fit(X, source)
    model = MultiCWoLa(K=3, backbone=clf, simplex="archetypal_corner").fit(X, source, y=y)
    assert model.predict(X).shape == (len(X),)
    assert model.score(X, y) > 0.5


def test_report_runs_and_flags():
    X, source, y, _ = _make_mixtures(seed=4)
    model = MultiCWoLa(K=3, backbone="mlp", train_kwargs={"epochs": 6}).fit(X, source, y=y)
    report = model.report()
    names = {c.name for c in model.report().checks}
    assert "rank / conditioning" in names
    assert "separability / anchors" in names
    assert isinstance(str(report), str)


def test_compute_diagnostics_detects_collapse():
    # Degenerate: two vertices identical -> A2 should fail.
    g = np.random.default_rng(0).dirichlet(np.ones(4), size=200)
    vertices = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]], dtype=float)
    alpha = np.random.default_rng(0).dirichlet(np.ones(3), size=200)
    source = np.random.default_rng(0).integers(0, 4, size=200)
    report = compute_diagnostics(
        source_posteriors=g,
        class_posteriors=alpha,
        vertices=vertices,
        source_ids=source,
    )
    rank_check = next(c for c in report.checks if c.assumption == "A2")
    assert rank_check.status == "fail"
