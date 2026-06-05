"""Permutation alignment helpers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


def match_label_permutation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, dict[int, int]]:
    """Align predicted labels to ground truth with Hungarian matching."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {int(pred): int(true) for true, pred in zip(row_ind, col_ind, strict=True)}
    aligned = np.asarray([mapping.get(int(label), int(label)) for label in y_pred], dtype=np.int64)
    return aligned, mapping


def align_probabilities(
    probabilities: np.ndarray, mapping: dict[int, int], num_classes: int
) -> np.ndarray:
    """Permute probability columns using the label alignment mapping."""
    aligned = np.zeros_like(probabilities)
    for pred, true in mapping.items():
        aligned[:, true] = probabilities[:, pred]
    for class_id in range(num_classes):
        if not np.any(aligned[:, class_id]):
            aligned[:, class_id] = probabilities[:, class_id]
    aligned /= np.clip(aligned.sum(axis=1, keepdims=True), 1e-8, None)
    return aligned


def align_latent_classes(
    alpha: np.ndarray,
    y: np.ndarray,
    labeled_index: np.ndarray | None = None,
    num_classes: int | None = None,
) -> tuple[np.ndarray, dict[int, int]]:
    """Resolve the latent-class permutation from a small labeled subset.

    Decoded latent classes are only identifiable up to a permutation (a binary
    flip when K=2). Given true labels for some rows, this computes the
    Hungarian permutation on those rows and applies it to *all* of ``alpha``,
    removing the ambiguity without a per-class score-and-flip workaround.

    Parameters
    ----------
    alpha : (N, K) decoded latent posteriors, e.g. from :func:`decode_posteriors`.
    y : true class labels. Either length ``N`` (aligned to ``alpha``), or length
        ``len(labeled_index)`` when ``labeled_index`` is given.
    labeled_index : optional indices of the labeled rows of ``alpha``. When
        omitted, ``y`` is assumed to cover every row of ``alpha``.
    num_classes : K; defaults to ``alpha.shape[1]``.

    Returns
    -------
    (aligned_alpha, mapping) where ``aligned_alpha`` has columns permuted so
    class ``k`` matches the true label ``k``, and ``mapping`` maps predicted ->
    true class index.
    """
    alpha = np.asarray(alpha)
    y = np.asarray(y).astype(int)
    k = int(num_classes) if num_classes is not None else alpha.shape[1]
    if labeled_index is None:
        labeled_alpha = alpha
    else:
        labeled_alpha = alpha[np.asarray(labeled_index)]
    if len(labeled_alpha) != len(y):
        raise ValueError(
            "y must match the number of labeled rows "
            f"({len(labeled_alpha)} alpha rows vs {len(y)} labels)."
        )
    _, mapping = match_label_permutation(y, labeled_alpha.argmax(axis=1), num_classes=k)
    return align_probabilities(alpha, mapping, num_classes=k), mapping


def match_vertices(
    estimated_vertices: np.ndarray,
    true_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align estimated vertices to true ones by Euclidean distance."""
    cost = np.linalg.norm(estimated_vertices[:, None, :] - true_vertices[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    aligned = np.zeros_like(estimated_vertices)
    for row, col in zip(row_ind, col_ind, strict=True):
        aligned[col] = estimated_vertices[row]
    return aligned, col_ind
