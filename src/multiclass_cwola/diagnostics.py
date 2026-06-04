"""Trust diagnostics for prior-free multiclass CWoLa recovery.

The method recovers latent classes purely from the geometry of the mixture
posterior ``g(x) = P(m | x)``. Its validity rests on three assumptions
(see the paper, Sec. 3):

  * **A2 -- full column rank** ``rank(Pi) = K``. If violated the recovered
    simplex collapses to fewer than ``K - 1`` dimensions.
  * **A3 -- separability / anchors**. Each class needs a region where it is
    (almost) the only contributor, so that points reach the simplex vertices.
    Weak anchors shrink the simplex and degrade identification.
  * **A1 -- shared class-conditionals**. If ``p_k`` differs across mixtures,
    the classifier can separate mixtures using per-mixture *artifacts*
    (file/batch provenance, acquisition conditions) rather than class
    composition, and the recovered "classes" become meaningless.

On a user's own data there are usually **no labels** to check accuracy
against, so these geometric/structural diagnostics are the primary trust
signal. This module turns each assumption into a numeric check with an
``ok`` / ``warn`` / ``fail`` status.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Check:
    """One assumption check with a traffic-light status."""

    name: str
    status: str  # "ok" | "warn" | "fail"
    value: float
    message: str
    assumption: str = ""


@dataclass
class DiagnosticsReport:
    """Bundle of trust diagnostics for one fitted recovery."""

    checks: list[Check] = field(default_factory=list)
    details: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when no check failed."""
        return all(c.status != "fail" for c in self.checks)

    @property
    def worst_status(self) -> str:
        order = {"ok": 0, "warn": 1, "fail": 2}
        return max((c.status for c in self.checks), key=lambda s: order[s], default="ok")

    def as_dict(self) -> dict[str, float]:
        return dict(self.details)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        symbol = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]"}
        width = max((len(c.name) for c in self.checks), default=0)
        lines = ["MultiCWoLa diagnostics", "=" * 22]
        for c in self.checks:
            tag = f" ({c.assumption})" if c.assumption else ""
            lines.append(f"{symbol[c.status]} {c.name.ljust(width)}  {c.message}{tag}")
        lines.append("-" * 22)
        lines.append(f"overall: {self.worst_status.upper()}")
        return "\n".join(lines)


def _condition_and_volume(vertices: np.ndarray) -> tuple[float, float, float]:
    """Return (condition number, simplex-volume proxy, min vertex distance)."""
    singular_values = np.linalg.svd(vertices, compute_uv=False)
    if singular_values[-1] < 1e-12:
        condition = float("inf")
    else:
        condition = float(singular_values[0] / singular_values[-1])
    # Volume of the (K-1)-simplex spanned by the centered vertices.
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    sign, logdet = np.linalg.slogdet(gram + 1e-12 * np.eye(len(vertices)))
    volume = float(np.exp(0.5 * logdet)) if sign > 0 else 0.0
    best = float("inf")
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            best = min(best, float(np.linalg.norm(vertices[i] - vertices[j])))
    min_distance = 0.0 if best == float("inf") else best
    return condition, volume, min_distance


def _mixture_confusion_diagonals(
    true_mixture: np.ndarray,
    pred_mixture: np.ndarray,
    pi: np.ndarray | None,
    mixture_priors: np.ndarray | None = None,
) -> tuple[float, float | None]:
    """Mean-diagonal recovery for empirical vs composition-limited confusion.

    See :func:`multiclass_cwola.visualization.plots.plot_mixture_confusion` for
    the geometric rationale: a classifier that separates mixtures far better
    than their class compositions allow is exploiting an A1 artifact.
    """
    true_mixture = np.asarray(true_mixture).astype(int)
    pred_mixture = np.asarray(pred_mixture).astype(int)
    M = int(max(true_mixture.max(initial=0), pred_mixture.max(initial=0))) + 1
    if pi is not None:
        M = max(M, np.asarray(pi).shape[0])
    cm = np.zeros((M, M), dtype=np.float64)
    np.add.at(cm, (true_mixture, pred_mixture), 1.0)
    cm /= np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    emp_diag = float(np.mean(np.diag(cm)))

    if pi is None:
        return emp_diag, None
    pi = np.asarray(pi, dtype=np.float64)
    Mp = pi.shape[0]
    if mixture_priors is None:
        counts = np.bincount(true_mixture, minlength=M).astype(np.float64)[:Mp]
        priors = counts / max(counts.sum(), 1.0)
    else:
        priors = np.asarray(mixture_priors, dtype=np.float64)
        priors = priors / max(priors.sum(), 1e-12)
    joint = priors[:, None] * pi
    best_m = joint.argmax(axis=0)
    ref = np.zeros((M, M), dtype=np.float64)
    for m in range(Mp):
        for k in range(pi.shape[1]):
            ref[m, best_m[k]] += pi[m, k]
    ref_diag = float(np.mean(np.diag(ref)[:Mp]))
    return emp_diag, ref_diag


def compute_diagnostics(
    *,
    source_posteriors: np.ndarray,
    class_posteriors: np.ndarray,
    vertices: np.ndarray,
    source_ids: np.ndarray,
    pi_hat: np.ndarray | None = None,
    reconstruction_error: float | None = None,
    condition_warn: float = 30.0,
    condition_fail: float = 1e3,
    anchor_warn: float = 0.8,
    anchor_fail: float = 0.5,
    leakage_warn: float = 0.10,
    leakage_fail: float = 0.25,
) -> DiagnosticsReport:
    """Compute assumption-level trust checks for a fitted recovery.

    Parameters
    ----------
    source_posteriors : (N, M) calibrated mixture posteriors ``g(x)``.
    class_posteriors : (N, K) decoded latent posteriors ``alpha(x)``.
    vertices : (K, M) fitted simplex vertices.
    source_ids : (N,) observed mixture identities.
    pi_hat : optional (M, K) recovered row-stochastic mixing matrix; enables
        the A1 artifact check.
    reconstruction_error : optional mean residual of the simplex fit.
    """
    report = DiagnosticsReport()
    K = class_posteriors.shape[1]

    # --- A2: rank / non-degeneracy of the recovered simplex --------------
    condition, volume, min_distance = _condition_and_volume(np.asarray(vertices))
    report.details |= {
        "vertex_condition_number": condition,
        "simplex_volume": volume,
        "min_vertex_distance": min_distance,
    }
    if not np.isfinite(condition) or min_distance < 1e-3 or condition >= condition_fail:
        status = "fail"
        msg = f"simplex near-degenerate (cond={condition:.1f}, min_dist={min_distance:.3g})"
    elif condition >= condition_warn:
        status = "warn"
        msg = f"ill-conditioned simplex (cond={condition:.1f}); check rank(Pi)>=K"
    else:
        status = "ok"
        msg = f"well-conditioned (cond={condition:.1f}, vol={volume:.3g})"
    report.checks.append(Check("rank / conditioning", status, condition, msg, "A2"))

    # --- A3: separability / anchor coverage ------------------------------
    # How close does the cloud get to each vertex? Per class, the max decoded
    # mass over points; a weak anchor means no point is near that corner.
    anchor_scores = class_posteriors.max(axis=0)  # (K,)
    min_anchor = float(anchor_scores.min())
    mean_max_alpha = float(class_posteriors.max(axis=1).mean())
    report.details |= {
        "min_anchor_score": min_anchor,
        "mean_anchor_score": float(anchor_scores.mean()),
        "mean_peak_posterior": mean_max_alpha,
    }
    if min_anchor < anchor_fail:
        status = "fail"
        msg = f"weak anchors: closest approach to a vertex is {min_anchor:.2f} (need separability)"
    elif min_anchor < anchor_warn:
        status = "warn"
        msg = f"moderate anchors (min vertex coverage {min_anchor:.2f}); simplex may be shrunk"
    else:
        status = "ok"
        msg = f"clear anchors (min vertex coverage {min_anchor:.2f})"
    report.checks.append(Check("separability / anchors", status, min_anchor, msg, "A3"))

    # --- A1: provenance / artifact leakage -------------------------------
    pred_mixture = np.asarray(source_posteriors).argmax(axis=1)
    emp_diag, ref_diag = _mixture_confusion_diagonals(
        np.asarray(source_ids), pred_mixture, pi_hat
    )
    report.details["mixture_recovery_empirical"] = emp_diag
    if ref_diag is None:
        report.checks.append(
            Check(
                "artifact leakage",
                "warn",
                emp_diag,
                "no pi_hat available; cannot compare against composition limit",
                "A1",
            )
        )
    else:
        leakage = emp_diag - ref_diag
        report.details["mixture_recovery_composition_limit"] = ref_diag
        report.details["artifact_leakage_gap"] = leakage
        if leakage >= leakage_fail:
            status = "fail"
            msg = (
                f"mixtures separated far beyond composition "
                f"(empirical {emp_diag:.2f} vs limit {ref_diag:.2f}); likely A1 artifact"
            )
        elif leakage >= leakage_warn:
            status = "warn"
            msg = f"some excess mixture separability (gap {leakage:+.2f}); inspect for artifacts"
        else:
            status = "ok"
            msg = f"separability consistent with composition (gap {leakage:+.2f})"
        report.checks.append(Check("artifact leakage", status, leakage, msg, "A1"))

    # --- Fit quality (informational) -------------------------------------
    if reconstruction_error is not None:
        report.details["reconstruction_error"] = float(reconstruction_error)
        report.checks.append(
            Check(
                "fit residual",
                "ok",
                float(reconstruction_error),
                f"mean simplex reconstruction error {reconstruction_error:.4f}",
            )
        )

    report.details["num_classes"] = float(K)
    return report
