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
