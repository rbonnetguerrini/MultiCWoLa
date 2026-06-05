"""Shared synthetic-data helpers for the test suite."""

from __future__ import annotations

import numpy as np
import pytest


def _make_mixtures(seed: int = 0, K: int = 3, M: int = 6, n_per: int = 400):
    """Synthetic well-separated Gaussian classes drawn into M mixtures.

    Returns (X, source, y, pi) with a strong-diagonal row-stochastic mixing matrix
    so that anchors exist and the simplex is identifiable.
    """
    rng = np.random.default_rng(seed)
    means = rng.normal(scale=6.0, size=(K, 8))
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
    """Oracle g(x) = P(m | x) for the synthetic generator (uniform mixture prior)."""
    log_pk = -0.5 * ((X[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)  # (N, K)
    pk = np.exp(log_pk - log_pk.max(axis=1, keepdims=True))
    qm = pk @ pi.T  # (N, M)
    return qm / qm.sum(axis=1, keepdims=True)


def _class_means(X, y, K):
    return np.stack([X[y == k].mean(axis=0) for k in range(K)])


@pytest.fixture
def make_mixtures():
    return _make_mixtures


@pytest.fixture
def bayes_posteriors():
    return _bayes_source_posteriors


@pytest.fixture
def class_means():
    return _class_means
