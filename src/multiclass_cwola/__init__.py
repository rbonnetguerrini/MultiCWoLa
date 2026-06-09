"""Multiclass CWoLa: prior-free multiclass recovery from unlabeled mixtures.

Bring your own model — the library never trains one. You supply the model; it provides
the structure around the two recovery methods plus alignment, diagnostics, and plotting.

- **MultiCWoLa** orchestrates a recovery from your model's outputs and gives you
  diagnostics and mixture-vs-latent plots:
  ``fit_posteriors(g, source)`` (post-hoc) or ``fit_bottleneck(alpha, pi, source)``
  (from a trained ``BottleneckSimplexHead``).
- **Functional primitives** for users who prefer plain functions:
  ``fit_simplex`` -> ``decode_posteriors`` -> ``align_latent_classes``, plus
  calibration helpers (``fit_calibrator`` / ``calibrate_logits``) and the
  ``BottleneckSimplexHead`` drop-in head. These only need source posteriors
  ``g(x) = P(m | x)``.
"""

from multiclass_cwola.api import MultiCWoLa
from multiclass_cwola.calibration.posthoc import (
    CalibrationResult,
    calibrate_logits,
    fit_calibrator,
)
from multiclass_cwola.diagnostics import DiagnosticsReport, compute_diagnostics
from multiclass_cwola.evaluation.matching import align_latent_classes
from multiclass_cwola.models.backbones import BottleneckSimplexHead
from multiclass_cwola.simplex.fitters import (
    SimplexFitResult,
    decode_posteriors,
    fit_simplex,
    transform_simplex_points,
)
from multiclass_cwola.simplex.projection import batch_simplex_least_squares

__all__ = [
    # batteries-included
    "MultiCWoLa",
    # functional primitives
    "fit_simplex",
    "SimplexFitResult",
    "decode_posteriors",
    "transform_simplex_points",
    "batch_simplex_least_squares",
    "align_latent_classes",
    "BottleneckSimplexHead",
    # calibration
    "fit_calibrator",
    "calibrate_logits",
    "CalibrationResult",
    # diagnostics
    "DiagnosticsReport",
    "compute_diagnostics",
    "__version__",
]

__version__ = "0.1.0"
