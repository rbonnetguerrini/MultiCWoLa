"""Mixing-matrix utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MixingMatrixInfo:
    """A generated mixing matrix plus diagnostics."""

    pi: np.ndarray
    rank: int
    condition_number: float
    row_similarity: float


def _condition_number_surrogate(pi: np.ndarray) -> float:
    singular_values = np.linalg.svd(pi, compute_uv=False)
    if singular_values[-1] < 1e-8:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def _row_similarity(pi: np.ndarray) -> float:
    similarities: list[float] = []
    for i in range(pi.shape[0]):
        for j in range(i + 1, pi.shape[0]):
            numerator = float(np.dot(pi[i], pi[j]))
            denominator = float(np.linalg.norm(pi[i]) * np.linalg.norm(pi[j]) + 1e-8)
            similarities.append(numerator / denominator)
    return float(np.mean(similarities)) if similarities else 1.0


def generate_mixing_matrix(
    num_sources: int,
    num_classes: int,
    mode: str = "dirichlet",
    dirichlet_alpha: float = 0.8,
    similarity: float = 0.9,
    condition_strength: float = 0.25,
    purity: float = 1.0,
    seed: int = 0,
) -> MixingMatrixInfo:
    """Generate an M x K row-stochastic mixing matrix."""
    rng = np.random.default_rng(seed)
    if mode == "dirichlet":
        raw = rng.dirichlet(np.full(num_classes, dirichlet_alpha), size=num_sources)
    elif mode == "near_rank_deficient":
        base = rng.dirichlet(np.full(num_classes, dirichlet_alpha), size=1)
        noise = rng.normal(scale=max(1e-3, 1.0 - similarity), size=(num_sources, num_classes))
        raw = np.clip(base + noise, 1e-5, None)
        raw /= raw.sum(axis=1, keepdims=True)
    elif mode == "nearly_identical":
        base = rng.dirichlet(np.full(num_classes, dirichlet_alpha), size=1)
        blends = similarity * np.repeat(base, num_sources, axis=0)
        random_part = (1.0 - similarity) * rng.dirichlet(
            np.full(num_classes, dirichlet_alpha), size=num_sources
        )
        raw = blends + random_part
        raw /= raw.sum(axis=1, keepdims=True)
    elif mode == "cyclic_purity":
        if num_classes < 2:
            raise ValueError("cyclic_purity requires at least two classes")
        if not 0.0 <= purity <= 1.0:
            raise ValueError(f"purity must be in [0, 1], got {purity}")
        off_target = (1.0 - purity) / float(num_classes - 1)
        raw = np.full((num_sources, num_classes), off_target, dtype=float)
        for source_id in range(num_sources):
            raw[source_id, source_id % num_classes] = purity
    else:
        raise ValueError(f"Unsupported pi mode: {mode}")

    if mode != "cyclic_purity" and num_sources >= num_classes and condition_strength > 0:
        identity_like = np.zeros((num_sources, num_classes))
        for index in range(num_sources):
            identity_like[index, index % num_classes] = 1.0
        raw = (1.0 - condition_strength) * raw + condition_strength * identity_like
        raw /= raw.sum(axis=1, keepdims=True)

    return MixingMatrixInfo(
        pi=raw,
        rank=int(np.linalg.matrix_rank(raw)),
        condition_number=_condition_number_surrogate(raw),
        row_similarity=_row_similarity(raw),
    )
