"""Multiclass CWoLa research prototype."""

from multiclass_cwola.api import MultiCWoLa
from multiclass_cwola.diagnostics import DiagnosticsReport, compute_diagnostics

__all__ = [
    "MultiCWoLa",
    "DiagnosticsReport",
    "compute_diagnostics",
    "__version__",
]

__version__ = "0.1.0"
