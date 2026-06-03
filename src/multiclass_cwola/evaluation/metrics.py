"""Evaluation metrics used across experiments."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, f1_score, log_loss

from multiclass_cwola.evaluation.matching import match_vertices


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, num_bins: int = 10
) -> float:
    """Compute ECE from predicted probabilities and labels."""
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    total = len(labels)
    ece = 0.0
    for start, end in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        mask = (confidences >= start) & (confidences < end if end < 1.0 else confidences <= end)
        if not np.any(mask):
            continue
        accuracy = np.mean(predictions[mask] == labels[mask])
        confidence = np.mean(confidences[mask])
        ece += (np.sum(mask) / total) * abs(accuracy - confidence)
    return float(ece)


def basic_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    num_bins: int = 10,
) -> dict[str, float]:
    """Top-level classification metrics."""
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "log_loss": float(
            log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1]))
        ),
        "ece": expected_calibration_error(probabilities, labels, num_bins=num_bins),
    }


def posterior_l1_error(estimated: np.ndarray, target: np.ndarray | None) -> float | None:
    """Average L1 error for posterior vectors."""
    if target is None:
        return None
    return float(np.mean(np.abs(estimated - target)))


def vertex_recovery_error(estimated_vertices: np.ndarray, true_vertices: np.ndarray) -> float:
    """Average Euclidean vertex error after optimal matching."""
    aligned, _ = match_vertices(estimated_vertices, true_vertices)
    return float(np.mean(np.linalg.norm(aligned - true_vertices, axis=1)))


def _mean_pairwise_vertex_distance(vertices: np.ndarray) -> float:
    distances = [
        np.linalg.norm(vertices[i] - vertices[j])
        for i in range(len(vertices))
        for j in range(i + 1, len(vertices))
    ]
    if not distances:
        return 1.0
    return float(np.mean(distances))


def simplex_geometric_distance(
    estimated_vertices: np.ndarray,
    true_vertices: np.ndarray,
    *,
    normalize: bool = False,
) -> float:
    """Permutation-invariant RMS distance from the fitted simplex to the oracle simplex."""
    aligned, _ = match_vertices(estimated_vertices, true_vertices)
    distance = float(np.sqrt(np.mean(np.sum((aligned - true_vertices) ** 2, axis=1))))
    if not normalize:
        return distance
    scale = max(_mean_pairwise_vertex_distance(true_vertices), 1e-8)
    return float(distance / scale)


def mixing_matrix_recovery_error(
    estimated_pi: np.ndarray | None, true_pi: np.ndarray
) -> float | None:
    """Average row L1 error for Pi recovery."""
    if estimated_pi is None:
        return None
    cost = np.zeros((estimated_pi.shape[1], true_pi.shape[1]))
    for i in range(estimated_pi.shape[1]):
        for j in range(true_pi.shape[1]):
            cost[i, j] = np.mean(np.abs(estimated_pi[:, i] - true_pi[:, j]))
    row_ind, col_ind = linear_sum_assignment(cost)
    aligned = np.zeros_like(estimated_pi)
    for row, col in zip(row_ind, col_ind, strict=True):
        aligned[:, col] = estimated_pi[:, row]
    return float(np.mean(np.abs(aligned - true_pi)))


def runtime_memory_summary(start_time: float) -> dict[str, Any]:
    """Collect runtime and memory summaries."""
    summary: dict[str, Any] = {"runtime_seconds": time.perf_counter() - start_time}
    try:
        import resource

        summary["max_rss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        summary["max_rss_kb"] = None
    if hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available():
        summary["max_cuda_bytes"] = int(__import__("torch").cuda.max_memory_allocated())
    else:
        summary["max_cuda_bytes"] = 0
    return summary
