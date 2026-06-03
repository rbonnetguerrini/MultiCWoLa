"""Oracle baseline that uses the true source-given-class simplex vertices."""

from __future__ import annotations

import numpy as np

from multiclass_cwola.simplex.projection import batch_simplex_least_squares


def run_known_prior_oracle(
    calibrated_source_posteriors: np.ndarray,
    true_source_given_class: np.ndarray,
) -> np.ndarray:
    """Recover latent posteriors with the true simplex vertices exposed."""
    return batch_simplex_least_squares(calibrated_source_posteriors, true_source_given_class)
