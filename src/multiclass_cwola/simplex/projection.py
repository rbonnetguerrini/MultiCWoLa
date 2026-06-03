"""Projection and simplex-constrained least squares."""

from __future__ import annotations

import numpy as np


def project_to_simplex(vector: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex."""
    vector = np.asarray(vector, dtype=np.float64)
    sorted_vector = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_vector)
    rho = np.nonzero(sorted_vector - (cumulative - 1.0) / (np.arange(len(vector)) + 1) > 0)[0][-1]
    theta = (cumulative[rho] - 1.0) / float(rho + 1)
    return np.maximum(vector - theta, 0.0)


def project_rows_to_simplex(matrix: np.ndarray) -> np.ndarray:
    """Vectorised row-wise projection onto the probability simplex."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        return project_to_simplex(matrix)
    n, k = matrix.shape
    sorted_desc = -np.sort(-matrix, axis=1)
    cumulative = np.cumsum(sorted_desc, axis=1)
    indices = np.arange(1, k + 1, dtype=np.float64)
    cond = sorted_desc - (cumulative - 1.0) / indices > 0
    rho = (cond * indices).argmax(axis=1)  # last True index since indices monotone & cond monotone-decreasing
    # Use np.where to find the actual last True per row to be safe
    last_true = k - 1 - np.argmax(cond[:, ::-1], axis=1)
    rho = np.where(cond.any(axis=1), last_true, 0)
    theta = (np.take_along_axis(cumulative, rho[:, None], axis=1)[:, 0] - 1.0) / (rho + 1)
    return np.maximum(matrix - theta[:, None], 0.0)


def unconstrained_simplex_least_squares(y: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Solve the unconstrained least-squares coefficients for y ~= a V."""
    gram = vertices @ vertices.T
    rhs = vertices @ y
    return np.linalg.pinv(gram) @ rhs


def simplex_least_squares(y: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Solve min_a ||y - a V|| subject to a on the simplex."""
    solution = unconstrained_simplex_least_squares(y, vertices)
    return project_to_simplex(solution)


def batch_unconstrained_least_squares(y: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Vectorised unconstrained least squares for each row of y.

    Solves a_i = (V V^T)^+ V y_i in one matmul instead of a Python per-row
    loop, since the system matrix is identical across rows.
    """
    y = np.asarray(y, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    gram = vertices @ vertices.T
    pinv = np.linalg.pinv(gram)
    return y @ vertices.T @ pinv


def batch_simplex_least_squares(y: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Vectorised simplex-constrained least squares for each row of y."""
    raw = batch_unconstrained_least_squares(y, vertices)
    return project_rows_to_simplex(raw)
