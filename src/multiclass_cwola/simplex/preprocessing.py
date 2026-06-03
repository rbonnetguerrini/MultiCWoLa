"""Affine preprocessing for simplex fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AffineProjection:
    """Rank-constrained affine projection in the ambient posterior space."""

    mean: np.ndarray
    basis: np.ndarray

    def transform(self, points: np.ndarray) -> np.ndarray:
        """Project points onto the learned affine subspace."""
        centered = points - self.mean[None, :]
        return self.mean[None, :] + centered @ self.basis @ self.basis.T


def fit_affine_projection(points: np.ndarray, affine_dim: int) -> AffineProjection | None:
    """Fit the best affine subspace of a requested dimension."""
    if affine_dim <= 0 or points.shape[1] <= affine_dim:
        return None
    mean = points.mean(axis=0)
    centered = points - mean[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:affine_dim].T
    return AffineProjection(mean=mean, basis=basis)
