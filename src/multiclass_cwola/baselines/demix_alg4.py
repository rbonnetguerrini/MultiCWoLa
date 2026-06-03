"""Finite-vector implementation of Algorithm 4 Demix baselines.

The routines in this module operate only on empirical distributions represented
as probability vectors over a shared finite basis.  The input baseline builds
that basis from raw inputs; the posterior baseline builds it from calibrated
source-posterior vectors supplied by the experiment pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linprog, minimize
from sklearn.cluster import KMeans
from sklearn.random_projection import GaussianRandomProjection

def _flatten_features(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim <= 2:
        return array
    return array.reshape(array.shape[0], -1)


class DemixError(RuntimeError):
    """Raised when finite Demix cannot produce a stable result."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass
class DemixConfig:
    """Configuration for finite Demix baselines."""

    method: str = "demix_alg4_input"
    label: str = "Demix Alg. 4 input"
    information_level: str = "fair"
    representation: str = "input_quantized_histogram"
    quantizer: str = "kmeans"
    num_bins: int = 128
    random_projection_dim: int | None = None
    eps: float = 1e-8
    face_eps: float = 1e-4
    max_face_iter: int = 100
    num_face_restarts: int = 20
    num_nonsquare_restarts: int = 20
    prior_mode: str = "reconstruction"
    failure_policy: str = "report_nan"
    random_state: int = 7


@dataclass
class DemixRunResult:
    """Predictions and diagnostics from one finite Demix baseline."""

    val_probabilities: np.ndarray
    test_probabilities: np.ndarray
    component_histograms: np.ndarray
    source_histograms: np.ndarray
    reconstruction_weights: np.ndarray | None
    diagnostics: dict[str, Any]


@dataclass
class QuantizedHistogramRepresentation:
    """Shared KMeans quantizer with optional unsupervised random projection."""

    num_bins: int
    random_projection_dim: int | None
    random_state: int
    quantizer_type: str = "kmeans"
    projection: GaussianRandomProjection | None = None
    quantizer: KMeans | None = None

    def fit(self, features: np.ndarray) -> QuantizedHistogramRepresentation:
        rows = _flatten_features(features)
        if (
            self.random_projection_dim is not None
            and self.random_projection_dim > 0
            and rows.shape[1] > self.random_projection_dim
        ):
            self.projection = GaussianRandomProjection(
                n_components=self.random_projection_dim,
                random_state=self.random_state,
            )
            rows = self.projection.fit_transform(rows)
        if self.quantizer_type != "kmeans":
            raise ValueError(f"Unsupported Demix quantizer: {self.quantizer_type}")
        n_clusters = min(int(self.num_bins), int(rows.shape[0]))
        if n_clusters < 1:
            raise ValueError("Cannot fit Demix quantizer on an empty feature array")
        self.quantizer = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        self.quantizer.fit(rows)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.quantizer is None:
            raise ValueError("Demix quantizer has not been fitted")
        rows = _flatten_features(features)
        if self.projection is not None:
            rows = self.projection.transform(rows)
        return self.quantizer.predict(rows)

    @property
    def fitted_bins(self) -> int:
        if self.quantizer is None:
            return 0
        return int(self.quantizer.n_clusters)


def _normalize_probability_vector(values: np.ndarray, eps: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).copy()
    vector[np.abs(vector) < eps] = 0.0
    if np.any(vector < -10.0 * eps):
        raise DemixError("Probability vector has substantial negative entries")
    vector = np.clip(vector, 0.0, None)
    total = float(vector.sum())
    if total <= eps or not np.isfinite(total):
        raise DemixError("Probability vector cannot be normalized")
    return vector / total


def kappa_star_vec(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """Compute kappa*(p | q) for probability vectors on a shared support."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = q > eps
    if not np.any(mask):
        raise DemixError("kappa_star_vec received a conditioning vector with empty support")
    kappa = float(np.min(p[mask] / q[mask]))
    if not np.isfinite(kappa):
        raise DemixError("kappa_star_vec produced a non-finite value")
    return float(np.clip(kappa, 0.0, 1.0 - eps))


def residue_vec(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return the finite-vector residue of p with respect to q."""
    kappa = kappa_star_vec(p, q, eps=eps)
    denom = 1.0 - kappa
    if denom <= eps:
        raise DemixError("residue_vec denominator is numerically unstable")
    return _normalize_probability_vector((np.asarray(p) - kappa * np.asarray(q)) / denom, eps)


def multi_residue_vec(
    p: np.ndarray,
    qs: list[np.ndarray] | np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the finite-vector multi-residue and LP diagnostics."""
    p = np.asarray(p, dtype=np.float64)
    q_array = np.asarray(qs, dtype=np.float64)
    if q_array.ndim != 2 or q_array.shape[0] < 1:
        raise DemixError("multi_residue_vec requires at least one conditioning vector")
    num_q = q_array.shape[0]
    c = -np.ones(num_q, dtype=np.float64)
    a_ub = np.vstack([q_array.T, np.ones((1, num_q), dtype=np.float64)])
    b_ub = np.concatenate([p, np.ones(1, dtype=np.float64)])
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0.0, None)] * num_q,
        method="highs",
    )
    diagnostics = {
        "status": int(result.status),
        "message": str(result.message),
        "objective": None if result.fun is None else float(result.fun),
        "success": bool(result.success),
    }
    if not result.success or result.x is None:
        raise DemixError("multi_residue_vec LP failed", {"lp_solver_status": diagnostics})
    coefficients = np.asarray(result.x, dtype=np.float64)
    total = float(coefficients.sum())
    diagnostics["coefficients"] = coefficients.tolist()
    diagnostics["coefficient_sum"] = total
    if total >= 1.0 - eps:
        raise DemixError("multi_residue_vec LP saturated the simplex", {"lp_solver_status": diagnostics})
    residual = p - coefficients @ q_array
    if np.min(residual) < -100.0 * eps:
        diagnostics["min_residual"] = float(np.min(residual))
        raise DemixError("multi_residue_vec produced a negative residual", {"lp_solver_status": diagnostics})
    return _normalize_probability_vector(residual / (1.0 - total), eps), diagnostics


def _kappa_matrix(distributions: np.ndarray, eps: float) -> np.ndarray:
    count = distributions.shape[0]
    matrix = np.ones((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(count):
            if i != j:
                matrix[i, j] = kappa_star_vec(distributions[i], distributions[j], eps=eps)
    return matrix


def face_test_vec(distributions: list[np.ndarray] | np.ndarray, face_eps: float = 1e-4, eps: float = 1e-8) -> bool:
    """Return true iff all off-diagonal kappa* values exceed face_eps."""
    array = np.asarray(distributions, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2:
        raise DemixError("face_test_vec requires at least two distributions")
    matrix = _kappa_matrix(array, eps=eps)
    off_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    return bool(np.all(off_diag > face_eps))


def find_face_vec(
    distributions: list[np.ndarray] | np.ndarray,
    *,
    eps: float = 1e-8,
    face_eps: float = 1e-4,
    max_face_iter: int = 100,
    num_face_restarts: int = 20,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Finite-vector FindFace subroutine from Algorithm 4."""
    array = np.asarray(distributions, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 3:
        raise DemixError("find_face_vec requires K >= 3 distributions")
    rng = np.random.default_rng() if rng is None else rng
    diagnostics: dict[str, Any] = {
        "face_test_outcomes": [],
        "num_face_restarts": int(num_face_restarts),
        "max_face_iter": int(max_face_iter),
        "selected_face_restart": None,
        "selected_face_iter": None,
    }
    tail = array[1:]
    for restart in range(int(num_face_restarts)):
        weights = rng.dirichlet(np.ones(tail.shape[0], dtype=np.float64))
        q = weights @ tail
        for n in range(2, int(max_face_iter) + 1):
            candidates = np.vstack(
                [
                    residue_vec((1.0 / n) * si + ((n - 1.0) / n) * q, array[0], eps=eps)
                    for si in tail
                ]
            )
            kappa = _kappa_matrix(candidates, eps=eps)
            passed = face_test_vec(candidates, face_eps=face_eps, eps=eps)
            diagnostics["face_test_outcomes"].append(
                {
                    "restart": restart,
                    "iteration": n,
                    "passed": bool(passed),
                    "kappa_matrix": kappa.tolist(),
                }
            )
            if passed:
                diagnostics["selected_face_restart"] = restart
                diagnostics["selected_face_iter"] = n
                return candidates, diagnostics
    raise DemixError("find_face_vec did not find a face", diagnostics)


def demix_square(
    distributions: list[np.ndarray] | np.ndarray,
    *,
    eps: float = 1e-8,
    face_eps: float = 1e-4,
    max_face_iter: int = 100,
    num_face_restarts: int = 20,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run finite-vector Algorithm 4 on exactly K observed distributions."""
    array = np.asarray(distributions, dtype=np.float64)
    if array.ndim != 2:
        raise DemixError("demix_square requires a 2D array of distributions")
    num_distributions = array.shape[0]
    if num_distributions < 2:
        raise DemixError("demix_square requires at least two distributions")
    rng = np.random.default_rng() if rng is None else rng
    diagnostics: dict[str, Any] = {
        "K": int(num_distributions),
        "kappa_matrix": _kappa_matrix(array, eps=eps).tolist(),
        "face_calls": [],
        "lp_solver_status": [],
    }
    if num_distributions == 2:
        components = np.vstack(
            [
                residue_vec(array[0], array[1], eps=eps),
                residue_vec(array[1], array[0], eps=eps),
            ]
        )
        diagnostics["component_kappa_matrix"] = _kappa_matrix(components, eps=eps).tolist()
        return components, diagnostics

    face, face_diag = find_face_vec(
        array,
        eps=eps,
        face_eps=face_eps,
        max_face_iter=max_face_iter,
        num_face_restarts=num_face_restarts,
        rng=rng,
    )
    diagnostics["face_calls"].append(face_diag)
    recursive_components, recursive_diag = demix_square(
        face,
        eps=eps,
        face_eps=face_eps,
        max_face_iter=max_face_iter,
        num_face_restarts=num_face_restarts,
        rng=rng,
    )
    average = np.mean(array, axis=0)
    final_component, lp_diag = multi_residue_vec(average, recursive_components, eps=eps)
    diagnostics["recursive"] = recursive_diag
    diagnostics["lp_solver_status"].append(lp_diag)
    components = np.vstack([recursive_components, final_component])
    diagnostics["component_kappa_matrix"] = _kappa_matrix(components, eps=eps).tolist()
    return components, diagnostics


def _fit_reconstruction_weights(
    observed: np.ndarray,
    components: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    weights = np.zeros((observed.shape[0], components.shape[0]), dtype=np.float64)
    reconstructed = np.zeros_like(observed, dtype=np.float64)
    row_status: list[dict[str, Any]] = []
    for row_idx, source_distribution in enumerate(observed):
        def objective(w: np.ndarray, target: np.ndarray = source_distribution) -> float:
            return float(np.sum((w @ components - target) ** 2))

        initial = np.full(components.shape[0], 1.0 / components.shape[0], dtype=np.float64)
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * components.shape[0],
            constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
            options={"maxiter": 500, "ftol": eps},
        )
        row_status.append(
            {
                "source_index": row_idx,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "objective": float(result.fun) if np.isfinite(result.fun) else None,
            }
        )
        if not result.success:
            raise DemixError("Reconstruction weight fitting failed", {"reconstruction_status": row_status})
        row_weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, None)
        row_sum = float(row_weights.sum())
        if row_sum <= eps or not np.isfinite(row_sum):
            raise DemixError("Reconstruction weights are numerically unstable")
        weights[row_idx] = row_weights / row_sum
        reconstructed[row_idx] = weights[row_idx] @ components
    error = float(np.sqrt(np.mean((observed - reconstructed) ** 2)))
    diagnostics = {"row_status": row_status, "reconstruction_error": error}
    return weights, error, diagnostics


def demix_nonsquare(
    distributions: list[np.ndarray] | np.ndarray,
    K: int,  # noqa: N803 - keep the Algorithm 4 notation in the public utility.
    *,
    eps: float = 1e-8,
    face_eps: float = 1e-4,
    max_face_iter: int = 100,
    num_face_restarts: int = 20,
    num_nonsquare_restarts: int = 20,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run square Demix directly when M=K, or use synthetic convex K-tuples when M>K."""
    array = np.asarray(distributions, dtype=np.float64)
    if array.ndim != 2:
        raise DemixError("demix_nonsquare requires a 2D array of distributions")
    if array.shape[0] < K:
        raise DemixError(f"Demix requires at least K observed distributions, got M={array.shape[0]}, K={K}")
    rng = np.random.default_rng() if rng is None else rng
    diagnostics: dict[str, Any] = {
        "num_observed_distributions": int(array.shape[0]),
        "num_components": int(K),
        "num_nonsquare_restarts": int(num_nonsquare_restarts),
        "candidate_reconstruction_errors": [],
        "selected_restart_index": None,
    }
    if array.shape[0] == K:
        components, square_diag = demix_square(
            array,
            eps=eps,
            face_eps=face_eps,
            max_face_iter=max_face_iter,
            num_face_restarts=num_face_restarts,
            rng=rng,
        )
        weights, error, recon_diag = _fit_reconstruction_weights(array, components, eps=eps)
        diagnostics["square"] = square_diag
        diagnostics["reconstruction"] = recon_diag
        diagnostics["selected_restart_index"] = 0
        diagnostics["candidate_reconstruction_errors"].append(error)
        diagnostics["best_reconstruction_error"] = float(error)
        return components, weights, diagnostics

    best: tuple[float, np.ndarray, np.ndarray, dict[str, Any]] | None = None
    failures: list[dict[str, Any]] = []
    for restart in range(int(num_nonsquare_restarts)):
        try:
            convex_weights = rng.dirichlet(np.ones(array.shape[0], dtype=np.float64), size=K)
            synthetic = convex_weights @ array
            components, square_diag = demix_square(
                synthetic,
                eps=eps,
                face_eps=face_eps,
                max_face_iter=max_face_iter,
                num_face_restarts=num_face_restarts,
                rng=rng,
            )
            weights, error, recon_diag = _fit_reconstruction_weights(array, components, eps=eps)
            diagnostics["candidate_reconstruction_errors"].append(error)
            candidate_diag = {
                "restart": restart,
                "synthetic_weights": convex_weights.tolist(),
                "square": square_diag,
                "reconstruction": recon_diag,
            }
            if best is None or error < best[0]:
                best = (error, components, weights, candidate_diag)
        except DemixError as exc:
            diagnostics["candidate_reconstruction_errors"].append(np.nan)
            failures.append({"restart": restart, "error": str(exc), "diagnostics": exc.diagnostics})
    diagnostics["failed_candidates"] = failures
    if best is None:
        raise DemixError("All non-square Demix restarts failed", diagnostics)
    error, components, weights, selected_diag = best
    diagnostics["selected_restart_index"] = int(selected_diag["restart"])
    diagnostics["selected_candidate"] = selected_diag
    diagnostics["reconstruction"] = selected_diag["reconstruction"]
    diagnostics["best_reconstruction_error"] = float(error)
    return components, weights, diagnostics


def _source_histograms(assignments: np.ndarray, sources: np.ndarray, num_sources: int, num_bins: int, eps: float) -> np.ndarray:
    histograms = np.zeros((num_sources, num_bins), dtype=np.float64)
    for source in range(num_sources):
        source_assignments = assignments[np.asarray(sources) == source]
        if source_assignments.size == 0:
            raise DemixError(f"Source {source} has no examples for Demix histogram fitting")
        counts = np.bincount(source_assignments, minlength=num_bins).astype(np.float64)
        histograms[source] = _normalize_probability_vector(counts, eps=eps)
    return histograms


def _predict_from_histograms(
    assignments: np.ndarray,
    components: np.ndarray,
    weights: np.ndarray | None,
    train_sources: np.ndarray,
    *,
    prior_mode: str,
    eps: float,
) -> np.ndarray:
    if prior_mode == "uniform" or weights is None:
        priors = np.full(components.shape[0], 1.0 / components.shape[0], dtype=np.float64)
    elif prior_mode == "reconstruction":
        source_counts = np.bincount(np.asarray(train_sources), minlength=weights.shape[0]).astype(np.float64)
        source_weights = source_counts / np.clip(source_counts.sum(), eps, None)
        priors = source_weights @ weights
        priors = _normalize_probability_vector(priors, eps=eps)
    else:
        raise ValueError(f"Unsupported Demix prior_mode: {prior_mode}")
    scores = components[:, assignments].T * priors[None, :]
    row_sums = scores.sum(axis=1, keepdims=True)
    collapsed = row_sums[:, 0] <= eps
    if np.any(collapsed):
        scores[collapsed] = priors[None, :]
        row_sums = scores.sum(axis=1, keepdims=True)
    return scores / np.clip(row_sums, eps, None)


def _failure_result(
    *,
    method_name: str,
    val_size: int,
    test_size: int,
    num_classes: int,
    diagnostics: dict[str, Any],
    failure_reason: str,
) -> DemixRunResult:
    val = np.full((val_size, num_classes), np.nan, dtype=np.float64)
    test = np.full((test_size, num_classes), np.nan, dtype=np.float64)
    diagnostics = dict(diagnostics)
    diagnostics["method"] = method_name
    diagnostics["failure_reason"] = failure_reason
    diagnostics["failed"] = True
    return DemixRunResult(
        val_probabilities=val,
        test_probabilities=test,
        component_histograms=np.full((num_classes, 0), np.nan),
        source_histograms=np.full((0, 0), np.nan),
        reconstruction_weights=None,
        diagnostics=diagnostics,
    )


def _run_demix_on_features(
    *,
    method_name: str,
    representation_type: str,
    train_features: np.ndarray,
    train_sources: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    num_sources: int,
    num_classes: int,
    config: DemixConfig,
) -> DemixRunResult:
    diagnostics: dict[str, Any] = {
        "method": method_name,
        "representation_type": representation_type,
        "num_bins": int(config.num_bins),
        "quantizer_type": config.quantizer,
        "random_projection_dim": config.random_projection_dim,
        "failed": False,
    }
    try:
        representation = QuantizedHistogramRepresentation(
            num_bins=int(config.num_bins),
            random_projection_dim=config.random_projection_dim,
            random_state=int(config.random_state),
            quantizer_type=str(config.quantizer),
        ).fit(train_features)
        train_assignments = representation.transform(train_features)
        val_assignments = representation.transform(val_features)
        test_assignments = representation.transform(test_features)
        source_histograms = _source_histograms(
            train_assignments,
            train_sources,
            num_sources,
            representation.fitted_bins,
            eps=float(config.eps),
        )
        rng = np.random.default_rng(int(config.random_state))
        components, weights, demix_diag = demix_nonsquare(
            source_histograms,
            num_classes,
            eps=float(config.eps),
            face_eps=float(config.face_eps),
            max_face_iter=int(config.max_face_iter),
            num_face_restarts=int(config.num_face_restarts),
            num_nonsquare_restarts=int(config.num_nonsquare_restarts),
            rng=rng,
        )
        val_probabilities = _predict_from_histograms(
            val_assignments,
            components,
            weights,
            train_sources,
            prior_mode=str(config.prior_mode),
            eps=float(config.eps),
        )
        test_probabilities = _predict_from_histograms(
            test_assignments,
            components,
            weights,
            train_sources,
            prior_mode=str(config.prior_mode),
            eps=float(config.eps),
        )
        diagnostics |= {
            "source_histogram_shapes": list(source_histograms.shape),
            "source_histograms": source_histograms.tolist(),
            "recovered_component_histograms": components.tolist(),
            "final_reconstruction_weights": weights.tolist(),
            "demix": demix_diag,
            "clipping_renormalization": "tiny negatives clipped during vector normalization",
            "lp_solver_status": demix_diag.get("square", {}).get("lp_solver_status")
            or demix_diag.get("selected_candidate", {}).get("square", {}).get("lp_solver_status"),
        }
        return DemixRunResult(
            val_probabilities=val_probabilities,
            test_probabilities=test_probabilities,
            component_histograms=components,
            source_histograms=source_histograms,
            reconstruction_weights=weights,
            diagnostics=diagnostics,
        )
    except DemixError as exc:
        diagnostics["demix_error_diagnostics"] = exc.diagnostics
        if str(config.failure_policy) == "report_nan":
            return _failure_result(
                method_name=method_name,
                val_size=val_features.shape[0],
                test_size=test_features.shape[0],
                num_classes=num_classes,
                diagnostics=diagnostics,
                failure_reason=str(exc),
            )
        raise


def demix_config_from_dict(config: dict[str, Any] | None, *, defaults: DemixConfig | None = None) -> DemixConfig:
    """Build a DemixConfig from a plain dict while ignoring unknown Hydra keys."""
    base = defaults or DemixConfig()
    if config is None:
        return base
    values = dict(base.__dict__)
    values.update({key: value for key, value in dict(config).items() if key in values})
    return DemixConfig(**values)


def run_demix_alg4_input(
    *,
    train_x: np.ndarray,
    train_source: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    num_sources: int,
    num_classes: int,
    config: dict[str, Any] | DemixConfig | None = None,
) -> DemixRunResult:
    """Run the direct-input Algorithm 4 baseline."""
    demix_cfg = config if isinstance(config, DemixConfig) else demix_config_from_dict(config)
    demix_cfg.method = "demix_alg4_input"
    demix_cfg.label = "Demix Alg. 4 input"
    demix_cfg.information_level = "fair"
    demix_cfg.representation = "input_quantized_histogram"
    return _run_demix_on_features(
        method_name="demix_alg4_input",
        representation_type="input_quantized_histogram",
        train_features=train_x,
        train_sources=train_source,
        val_features=val_x,
        test_features=test_x,
        num_sources=num_sources,
        num_classes=num_classes,
        config=demix_cfg,
    )


def run_demix_alg4_posterior(
    *,
    train_posteriors: np.ndarray,
    train_source: np.ndarray,
    val_posteriors: np.ndarray,
    test_posteriors: np.ndarray,
    num_sources: int,
    num_classes: int,
    config: dict[str, Any] | DemixConfig | None = None,
) -> DemixRunResult:
    """Run the posterior-space Algorithm 4 ablation."""
    defaults = DemixConfig(
        method="demix_alg4_posterior",
        label="Demix Alg. 4 posterior",
        representation="posterior_quantized_histogram",
        random_projection_dim=None,
    )
    demix_cfg = config if isinstance(config, DemixConfig) else demix_config_from_dict(config, defaults=defaults)
    demix_cfg.method = "demix_alg4_posterior"
    demix_cfg.label = "Demix Alg. 4 posterior"
    demix_cfg.information_level = "fair"
    demix_cfg.representation = "posterior_quantized_histogram"
    demix_cfg.random_projection_dim = None
    return _run_demix_on_features(
        method_name="demix_alg4_posterior",
        representation_type="posterior_quantized_histogram",
        train_features=train_posteriors,
        train_sources=train_source,
        val_features=val_posteriors,
        test_features=test_posteriors,
        num_sources=num_sources,
        num_classes=num_classes,
        config=demix_cfg,
    )
