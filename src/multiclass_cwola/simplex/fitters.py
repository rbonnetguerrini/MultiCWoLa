"""Simplex fitting estimators for calibrated source posteriors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import lsq_linear

from multiclass_cwola.simplex.preprocessing import AffineProjection, fit_affine_projection
from multiclass_cwola.simplex.projection import (
    batch_simplex_least_squares,
    batch_unconstrained_least_squares,
    project_rows_to_simplex,
)
from multiclass_cwola.simplex.selection import (
    SelectionCandidate,
    SelectionOutcome,
    select_stability_consensus,
    select_stability_medoid,
)


@dataclass
class SimplexDiagnostics:
    """Quality and degeneracy diagnostics."""

    raw_reconstruction_error: float
    reconstruction_error: float
    vertex_condition_number: float
    min_vertex_distance: float
    raw_barycentric_negative_fraction: float
    raw_barycentric_negative_mass: float
    raw_barycentric_sum_error: float
    barycentric_projection_gap: float
    mean_barycentric_entropy: float
    barycentric_collapse_fraction: float
    selection_score: float
    selected_candidate: str
    degenerate: bool


@dataclass
class SimplexFitResult:
    """Estimated simplex quantities."""

    vertices: np.ndarray
    barycentric_coordinates: np.ndarray
    pi_hat: np.ndarray | None
    diagnostics: SimplexDiagnostics
    preprocessing: AffineProjection | None = None
    candidates: list[SelectionCandidate] | None = None
    candidate_rows: list[dict[str, Any]] | None = None
    selection_summary: dict[str, Any] | None = None


@dataclass
class BarycentricDiagnostics:
    """Diagnostics before and after simplex projection."""

    raw_alpha: np.ndarray
    projected_alpha: np.ndarray
    raw_reconstruction_error: float
    negative_fraction: float
    negative_mass: float
    sum_error: float
    projection_gap: float
    mean_entropy: float
    collapse_fraction: float


def _pairwise_min_distance(vertices: np.ndarray) -> float:
    best = float("inf")
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            best = min(best, float(np.linalg.norm(vertices[i] - vertices[j])))
    return 0.0 if best == float("inf") else best


def _estimate_pi_from_vertices(vertices: np.ndarray) -> np.ndarray | None:
    try:
        fit = lsq_linear(vertices.T, np.ones(vertices.shape[1]), bounds=(0.0, np.inf))
        scales = np.clip(fit.x, 1e-8, None)
        pi_hat = vertices.T * scales[None, :]
        pi_hat /= np.clip(pi_hat.sum(axis=1, keepdims=True), 1e-8, None)
        return pi_hat
    except Exception:
        return None


def barycentric_diagnostics(points: np.ndarray, vertices: np.ndarray) -> BarycentricDiagnostics:
    """Measure how much unconstrained barycentric coordinates violate simplex structure."""
    raw_alpha = batch_unconstrained_least_squares(points, vertices)
    projected_alpha = project_rows_to_simplex(raw_alpha)
    entropies = -np.sum(projected_alpha * np.log(np.clip(projected_alpha, 1e-8, None)), axis=1)
    return BarycentricDiagnostics(
        raw_alpha=raw_alpha,
        projected_alpha=projected_alpha,
        raw_reconstruction_error=float(np.mean(np.linalg.norm(points - raw_alpha @ vertices, axis=1))),
        negative_fraction=float(np.mean(raw_alpha < 0.0)),
        negative_mass=float(np.mean(np.clip(-raw_alpha, 0.0, None).sum(axis=1))),
        sum_error=float(np.mean(np.abs(raw_alpha.sum(axis=1) - 1.0))),
        projection_gap=float(np.mean(np.abs(projected_alpha - raw_alpha).sum(axis=1))),
        mean_entropy=float(np.mean(entropies)),
        collapse_fraction=float(np.mean(projected_alpha.max(axis=1) >= 0.95)),
    )


def _selection_score(
    reconstruction_error: float,
    condition: float,
    min_distance: float,
    barycentric: BarycentricDiagnostics,
) -> float:
    condition_penalty = 0.0 if not np.isfinite(condition) else 0.02 * max(condition - 2.0, 0.0)
    distance_penalty = 0.05 / max(min_distance, 1e-3)
    pathology_penalty = (
        0.75 * barycentric.projection_gap
        + 0.5 * barycentric.negative_mass
        + 0.25 * barycentric.sum_error
    )
    degeneracy_penalty = 10.0 if (min_distance < 1e-3 or not np.isfinite(condition)) else 0.0
    return float(reconstruction_error + condition_penalty + distance_penalty + pathology_penalty + degeneracy_penalty)


def _diagnostics(
    points: np.ndarray,
    vertices: np.ndarray,
    alpha: np.ndarray,
    barycentric: BarycentricDiagnostics,
    selection_score: float,
    selected_candidate: str,
) -> SimplexDiagnostics:
    reconstruction = alpha @ vertices
    reconstruction_error = float(np.mean(np.linalg.norm(points - reconstruction, axis=1)))
    singular_values = np.linalg.svd(vertices, compute_uv=False)
    condition = (
        float("inf")
        if singular_values[-1] < 1e-8
        else float(singular_values[0] / singular_values[-1])
    )
    min_distance = _pairwise_min_distance(vertices)
    degenerate = bool(min_distance < 1e-3 or not np.isfinite(condition))
    return SimplexDiagnostics(
        raw_reconstruction_error=barycentric.raw_reconstruction_error,
        reconstruction_error=reconstruction_error,
        vertex_condition_number=condition,
        min_vertex_distance=min_distance,
        raw_barycentric_negative_fraction=barycentric.negative_fraction,
        raw_barycentric_negative_mass=barycentric.negative_mass,
        raw_barycentric_sum_error=barycentric.sum_error,
        barycentric_projection_gap=barycentric.projection_gap,
        mean_barycentric_entropy=barycentric.mean_entropy,
        barycentric_collapse_fraction=barycentric.collapse_fraction,
        selection_score=selection_score,
        selected_candidate=selected_candidate,
        degenerate=degenerate,
    )


def farthest_point_vertices(
    points: np.ndarray, num_vertices: int, first_index: int | None = None
) -> np.ndarray:
    """Pick far-apart points as a heuristic simplex initializer."""
    chosen: list[int] = []
    mean = points.mean(axis=0)
    first = (
        int(np.argmax(np.linalg.norm(points - mean[None, :], axis=1)))
        if first_index is None
        else int(first_index)
    )
    chosen.append(first)
    while len(chosen) < num_vertices:
        distances = []
        for index in range(len(points)):
            candidate_distance = min(
                float(np.linalg.norm(points[index] - points[selected])) for selected in chosen
            )
            distances.append(candidate_distance)
        chosen.append(int(np.argmax(distances)))
    return project_rows_to_simplex(points[np.asarray(chosen)])


def _entropy(values: np.ndarray) -> np.ndarray:
    return -np.sum(values * np.log(np.clip(values, 1e-8, None)), axis=1)


def _simplex_volume_proxy(vertices: np.ndarray) -> float:
    if len(vertices) <= 1:
        return 0.0
    edges = vertices[1:] - vertices[0]
    gram = edges @ edges.T
    determinant = float(np.linalg.det(gram))
    if determinant <= 0.0:
        return 0.0
    return float(np.sqrt(determinant))


def _split_count(total: int, num_buckets: int) -> list[int]:
    if num_buckets <= 0:
        return []
    base = total // num_buckets
    remainder = total % num_buckets
    counts = [base] * num_buckets
    for index in range(remainder):
        counts[index] += 1
    return counts


def _corner_pool_indices(
    points: np.ndarray,
    *,
    num_vertices: int,
    pool_fraction: float,
) -> np.ndarray:
    num_points = len(points)
    if num_points == 0:
        return np.empty(0, dtype=np.int64)
    pool_size = max(num_vertices * 4, int(round(pool_fraction * num_points)))
    pool_size = min(pool_size, num_points)
    entropies = _entropy(points)
    maxima = points.max(axis=1)
    chosen: set[int] = set(np.argsort(entropies)[:pool_size].tolist())
    chosen.update(np.argsort(-maxima)[:pool_size].tolist())
    per_dim = max(num_vertices, pool_size // max(2 * points.shape[1], 1))
    for dim in range(points.shape[1]):
        chosen.update(np.argsort(-points[:, dim])[:per_dim].tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def _random_unit_directions(
    dim: int,
    rng: np.random.Generator,
    *,
    num_directions: int,
) -> list[np.ndarray]:
    directions: list[np.ndarray] = []
    while len(directions) < num_directions:
        direction = rng.normal(size=dim)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            continue
        directions.append(direction / norm)
    return directions


def _spread_pool_indices(
    points: np.ndarray,
    rng: np.random.Generator,
    *,
    num_vertices: int,
    num_projections: int,
    top_per_direction: int,
) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    directions: list[np.ndarray] = []
    num_pca = min(centered.shape[1], max(num_vertices, num_projections // 2))
    directions.extend(vh[:num_pca])
    directions.extend(
        _random_unit_directions(
            centered.shape[1],
            rng,
            num_directions=max(num_projections - len(directions), 0),
        )
    )
    radial_scores = np.linalg.norm(centered, axis=1)
    chosen: set[int] = set(np.argsort(-radial_scores)[: max(num_vertices * 2, top_per_direction)].tolist())
    for direction in directions[:num_projections]:
        scores = centered @ direction
        top_count = min(top_per_direction, len(scores))
        chosen.update(np.argsort(-scores)[:top_count].tolist())
        chosen.update(np.argsort(scores)[:top_count].tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def _greedy_volume_vertices(
    points: np.ndarray,
    *,
    pool_indices: np.ndarray,
    num_vertices: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(pool_indices) < num_vertices:
        return farthest_point_vertices(points, num_vertices=num_vertices)
    pool = points[pool_indices]
    if num_vertices == 1:
        return project_rows_to_simplex(pool[[int(np.argmax(np.linalg.norm(pool - pool.mean(axis=0), axis=1)))]])
    pairwise = np.linalg.norm(pool[:, None, :] - pool[None, :, :], axis=-1)
    start_flat = int(np.argmax(pairwise))
    first_local, second_local = np.unravel_index(start_flat, pairwise.shape)
    chosen_local: list[int] = [int(first_local), int(second_local)]
    while len(chosen_local) < num_vertices:
        best_index: int | None = None
        best_score = -1.0
        for local_index in range(len(pool)):
            if local_index in chosen_local:
                continue
            vertices = pool[np.asarray(chosen_local + [local_index])]
            score = _simplex_volume_proxy(vertices)
            if score > best_score + 1e-12:
                best_score = score
                best_index = local_index
            elif abs(score - best_score) < 1e-12 and best_index is not None:
                current = float(
                    np.min(
                        np.linalg.norm(
                            pool[local_index][None, :] - pool[np.asarray(chosen_local)],
                            axis=1,
                        )
                    )
                )
                previous = float(
                    np.min(
                        np.linalg.norm(
                            pool[best_index][None, :] - pool[np.asarray(chosen_local)],
                            axis=1,
                        )
                    )
                )
                if current > previous:
                    best_index = local_index
        if best_index is None:
            remaining = [index for index in range(len(pool)) if index not in chosen_local]
            best_index = int(rng.choice(remaining))
        chosen_local.append(int(best_index))
    return project_rows_to_simplex(pool[np.asarray(chosen_local)])


def _corner_vertices(
    points: np.ndarray,
    num_vertices: int,
    rng: np.random.Generator,
    *,
    pool_fraction: float,
) -> np.ndarray:
    pool_indices = _corner_pool_indices(
        points,
        num_vertices=num_vertices,
        pool_fraction=pool_fraction,
    )
    if len(pool_indices) < num_vertices:
        return farthest_point_vertices(points, num_vertices=num_vertices)
    return _greedy_volume_vertices(
        points,
        pool_indices=pool_indices,
        num_vertices=num_vertices,
        rng=rng,
    )


def _spread_vertices(
    points: np.ndarray,
    num_vertices: int,
    rng: np.random.Generator,
    *,
    num_projections: int,
    top_per_direction: int,
) -> np.ndarray:
    pool_indices = _spread_pool_indices(
        points,
        rng,
        num_vertices=num_vertices,
        num_projections=num_projections,
        top_per_direction=top_per_direction,
    )
    return _greedy_volume_vertices(
        points,
        pool_indices=pool_indices,
        num_vertices=num_vertices,
        rng=rng,
    )


def _strategy_vertices(
    strategy: str,
    points: np.ndarray,
    num_vertices: int,
    rng: np.random.Generator,
    *,
    first_index: int | None,
    corner_pool_fraction: float,
    spread_num_projections: int,
    spread_top_per_direction: int,
) -> np.ndarray:
    if strategy == "farthest":
        return farthest_point_vertices(points, num_vertices=num_vertices, first_index=first_index)
    if strategy == "corner":
        return _corner_vertices(
            points,
            num_vertices=num_vertices,
            rng=rng,
            pool_fraction=corner_pool_fraction,
        )
    if strategy == "spread":
        return _spread_vertices(
            points,
            num_vertices=num_vertices,
            rng=rng,
            num_projections=spread_num_projections,
            top_per_direction=spread_top_per_direction,
        )
    raise ValueError(f"Unsupported archetypal initialization strategy: {strategy}")


def _fit_archetypal_from_init(
    g: np.ndarray,
    vertices: np.ndarray,
    max_iter: int,
    tol: float,
    ridge: float,
) -> np.ndarray:
    prev_error = float("inf")
    alpha = batch_simplex_least_squares(g, vertices)
    for _ in range(max_iter):
        alpha = batch_simplex_least_squares(g, vertices)
        lhs = alpha.T @ alpha + ridge * np.eye(vertices.shape[0])
        rhs = alpha.T @ g
        vertices = np.linalg.solve(lhs, rhs)
        vertices = project_rows_to_simplex(vertices)
        reconstruction_error = float(np.mean(np.linalg.norm(g - alpha @ vertices, axis=1)))
        if abs(prev_error - reconstruction_error) < tol:
            break
        prev_error = reconstruction_error
    return vertices


def _torch_vertices_from_pi(pi: torch.Tensor) -> torch.Tensor:
    col_sums = pi.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return (pi / col_sums).T


def _separation_penalty(vertices: torch.Tensor) -> torch.Tensor:
    penalties: list[torch.Tensor] = []
    for i in range(vertices.shape[0]):
        for j in range(i + 1, vertices.shape[0]):
            distance = torch.norm(vertices[i] - vertices[j], p=2)
            penalties.append(1.0 / (distance + 1e-3))
    if not penalties:
        return torch.zeros((), dtype=vertices.dtype, device=vertices.device)
    return torch.stack(penalties).mean()


def _candidate_rows_for_single_fit(
    candidate: SelectionCandidate,
    selected_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [
        {
            "candidate_name": candidate.name,
            "candidate_source": candidate.source,
            "num_fit_points": candidate.num_fit_points,
            "selection_score": candidate.selection_score,
            "reconstruction_error": candidate.reconstruction_error,
            "raw_reconstruction_error": candidate.raw_reconstruction_error,
            "vertex_condition_number": candidate.vertex_condition_number,
            "min_vertex_distance": candidate.min_vertex_distance,
            "raw_negative_fraction": candidate.raw_negative_fraction,
            "raw_negative_mass": candidate.raw_negative_mass,
            "projection_gap": candidate.projection_gap,
            "mean_barycentric_entropy": candidate.mean_entropy,
            "barycentric_collapse_fraction": candidate.collapse_fraction,
            "degenerate": candidate.degenerate,
            "avg_aligned_distance": np.nan,
            "distance_to_cluster_reference": np.nan,
            "support_fraction": np.nan,
            "cluster_member": True,
            "normalized_local_score": 0.0,
            "normalized_stability_distance": 0.0,
            "selected": True,
        }
    ]
    summary = {
        "selection_mode": "single_candidate",
        "selected_candidate": selected_name,
        "selected_source": candidate.source,
        "num_candidates": 1,
        "num_viable_candidates": int(not candidate.degenerate),
    }
    return rows, summary


def fit_constrained_pi_simplex(
    g: np.ndarray,
    num_vertices: int,
    *,
    max_outer_iters: int = 30,
    inner_steps: int = 20,
    lr: float = 0.05,
    separation_weight: float = 1e-3,
    seed: int = 0,
    selection_points: np.ndarray | None = None,
    weighted: bool = False,
    corner_weight: float = 2.0,
    init_mode: str = "archetypal",
) -> SimplexFitResult:
    """Fit a simplex induced by a valid row-stochastic Pi."""
    selection = g if selection_points is None else selection_points
    rng = np.random.default_rng(seed)
    if init_mode == "archetypal":
        init_fit = fit_archetypal_simplex(
            g,
            num_vertices=num_vertices,
            max_iter=40,
            tol=1e-5,
            ridge=1e-4,
            num_restarts=4,
            seed=seed,
            selection_points=selection,
            include_heuristic_candidate=True,
            selector="restart_only",
            num_subsamples=0,
        )
        pi_init = init_fit.pi_hat
    elif init_mode == "uniform":
        pi_init = None
    else:
        raise ValueError(f"Unsupported constrained Pi initialization mode: {init_mode}")

    if pi_init is None or pi_init.shape != (g.shape[1], num_vertices):
        raw = rng.uniform(size=(g.shape[1], num_vertices))
        pi_init = raw / raw.sum(axis=1, keepdims=True)

    points_t = torch.as_tensor(g, dtype=torch.float32)
    weights_np = np.ones(len(g), dtype=np.float32)
    if weighted:
        normalized_entropy = _entropy(g) / np.log(g.shape[1])
        extremeness = 1.0 - normalized_entropy
        weights_np = 1.0 + float(corner_weight) * extremeness.astype(np.float32)
        weights_np /= np.mean(weights_np)
    weights_t = torch.as_tensor(weights_np, dtype=torch.float32)

    logits = torch.nn.Parameter(torch.as_tensor(np.log(np.clip(pi_init, 1e-8, None)), dtype=torch.float32))
    optimizer = torch.optim.Adam([logits], lr=lr)
    history: list[dict[str, float]] = []
    alpha_np = np.full((len(g), num_vertices), 1.0 / num_vertices, dtype=np.float32)

    for outer in range(max_outer_iters):
        with torch.no_grad():
            pi_t = torch.softmax(logits, dim=1)
            vertices_np = _torch_vertices_from_pi(pi_t).cpu().numpy()
            alpha_np = batch_simplex_least_squares(g, vertices_np)
        alpha_t = torch.as_tensor(alpha_np, dtype=torch.float32)

        total_value = np.nan
        recon_value = np.nan
        sep_value = np.nan
        for _ in range(inner_steps):
            optimizer.zero_grad()
            pi_t = torch.softmax(logits, dim=1)
            vertices_t = _torch_vertices_from_pi(pi_t)
            recon = alpha_t @ vertices_t
            squared_error = torch.sum((recon - points_t) ** 2, dim=1)
            reconstruction_loss = torch.mean(weights_t * squared_error)
            separation_loss = separation_weight * _separation_penalty(vertices_t)
            total_loss = reconstruction_loss + separation_loss
            total_loss.backward()
            optimizer.step()
            total_value = float(total_loss.item())
            recon_value = float(reconstruction_loss.item())
            sep_value = float(separation_loss.item())
        history.append(
            {
                "outer_iter": float(outer),
                "total_loss": total_value,
                "reconstruction_loss": recon_value,
                "separation_loss": sep_value,
            }
        )

    with torch.no_grad():
        pi_t = torch.softmax(logits, dim=1)
        pi_hat = pi_t.cpu().numpy()
        vertices = _torch_vertices_from_pi(pi_t).cpu().numpy()
    alpha = batch_simplex_least_squares(g, vertices)
    candidate_name = "constrained_pi_weighted" if weighted else "constrained_pi"
    _, diagnostics = _evaluate_fit(g, selection, vertices, candidate_name=candidate_name)
    candidate = _candidate_record(
        name=candidate_name,
        source="full",
        vertices=vertices,
        diagnostics=diagnostics,
        num_fit_points=len(g),
    )
    candidate_rows, selection_summary = _candidate_rows_for_single_fit(
        candidate,
        selected_name=candidate_name,
    )
    selection_summary.update(
        {
            "fit_method": "constrained_pi",
            "weighted": weighted,
            "corner_weight": float(corner_weight),
            "init_mode": init_mode,
            "max_outer_iters": int(max_outer_iters),
            "inner_steps": int(inner_steps),
            "lr": float(lr),
            "separation_weight": float(separation_weight),
            "loss_history": history,
        }
    )
    return SimplexFitResult(
        vertices=vertices,
        barycentric_coordinates=alpha,
        pi_hat=pi_hat,
        diagnostics=diagnostics,
        preprocessing=None,
        candidates=[candidate],
        candidate_rows=candidate_rows,
        selection_summary=selection_summary,
    )


def _evaluate_fit(
    fit_points: np.ndarray,
    selection_points: np.ndarray,
    vertices: np.ndarray,
    candidate_name: str,
) -> tuple[np.ndarray, SimplexDiagnostics]:
    alpha = batch_simplex_least_squares(fit_points, vertices)
    fit_barycentric = barycentric_diagnostics(fit_points, vertices)
    selection_barycentric = barycentric_diagnostics(selection_points, vertices)
    selection_reconstruction = float(
        np.mean(
            np.linalg.norm(
                selection_points - selection_barycentric.projected_alpha @ vertices,
                axis=1,
            )
        )
    )
    singular_values = np.linalg.svd(vertices, compute_uv=False)
    condition = (
        float("inf")
        if singular_values[-1] < 1e-8
        else float(singular_values[0] / singular_values[-1])
    )
    min_distance = _pairwise_min_distance(vertices)
    selection_score = _selection_score(
        reconstruction_error=selection_reconstruction,
        condition=condition,
        min_distance=min_distance,
        barycentric=selection_barycentric,
    )
    diagnostics = _diagnostics(
        fit_points,
        vertices,
        alpha,
        fit_barycentric,
        selection_score=selection_score,
        selected_candidate=candidate_name,
    )
    return alpha, diagnostics


def _candidate_record(
    *,
    name: str,
    source: str,
    vertices: np.ndarray,
    diagnostics: SimplexDiagnostics,
    num_fit_points: int,
) -> SelectionCandidate:
    return SelectionCandidate(
        name=name,
        source=source,
        vertices=vertices,
        selection_score=diagnostics.selection_score,
        reconstruction_error=diagnostics.reconstruction_error,
        raw_reconstruction_error=diagnostics.raw_reconstruction_error,
        vertex_condition_number=diagnostics.vertex_condition_number,
        min_vertex_distance=diagnostics.min_vertex_distance,
        raw_negative_fraction=diagnostics.raw_barycentric_negative_fraction,
        raw_negative_mass=diagnostics.raw_barycentric_negative_mass,
        projection_gap=diagnostics.barycentric_projection_gap,
        mean_entropy=diagnostics.mean_barycentric_entropy,
        collapse_fraction=diagnostics.barycentric_collapse_fraction,
        degenerate=diagnostics.degenerate,
        num_fit_points=num_fit_points,
    )


def _subsample_indices(
    num_points: int,
    *,
    rng: np.random.Generator,
    subsample_fraction: float,
    use_bootstrap: bool,
    min_points: int,
) -> np.ndarray:
    size = max(min_points, int(round(subsample_fraction * num_points)))
    size = min(size, num_points if not use_bootstrap else max(size, min_points))
    if use_bootstrap:
        return rng.integers(0, num_points, size=size)
    return rng.choice(num_points, size=size, replace=False)


def _select_vertices_from_candidates(
    *,
    candidates: list[SelectionCandidate],
    selector: str,
    g: np.ndarray,
    selection_points: np.ndarray,
    max_iter: int,
    tol: float,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, SimplexDiagnostics, list[dict[str, Any]], dict[str, Any]]:
    if not candidates:
        raise ValueError("No simplex candidates available for selection.")
    if selector == "restart_only":
        selected = min(candidates, key=lambda candidate: candidate.selection_score)
        alpha, diagnostics = _evaluate_fit(
            g,
            selection_points,
            selected.vertices,
            candidate_name=selected.name,
        )
        rows = []
        for candidate in candidates:
            rows.append(
                {
                    "candidate_name": candidate.name,
                    "candidate_source": candidate.source,
                    "num_fit_points": candidate.num_fit_points,
                    "selection_score": candidate.selection_score,
                    "reconstruction_error": candidate.reconstruction_error,
                    "raw_reconstruction_error": candidate.raw_reconstruction_error,
                    "vertex_condition_number": candidate.vertex_condition_number,
                    "min_vertex_distance": candidate.min_vertex_distance,
                    "raw_negative_fraction": candidate.raw_negative_fraction,
                    "raw_negative_mass": candidate.raw_negative_mass,
                    "projection_gap": candidate.projection_gap,
                    "mean_barycentric_entropy": candidate.mean_entropy,
                    "barycentric_collapse_fraction": candidate.collapse_fraction,
                    "degenerate": candidate.degenerate,
                    "avg_aligned_distance": np.nan,
                    "distance_to_cluster_reference": np.nan,
                    "support_fraction": np.nan,
                    "cluster_member": False,
                    "normalized_local_score": np.nan,
                    "normalized_stability_distance": np.nan,
                    "selected": candidate.name == selected.name,
                }
            )
        return (
            selected.vertices,
            alpha,
            diagnostics,
            rows,
            {
                "selection_mode": "restart_only",
                "selected_candidate": selected.name,
                "selected_source": selected.source,
                "num_candidates": len(candidates),
                "num_viable_candidates": int(sum(not candidate.degenerate for candidate in candidates)),
            },
        )

    outcome: SelectionOutcome
    if selector == "stability_medoid":
        outcome = select_stability_medoid(candidates)
        alpha, diagnostics = _evaluate_fit(
            g,
            selection_points,
            outcome.selected_vertices,
            candidate_name=outcome.selected_name,
        )
        return (
            outcome.selected_vertices,
            alpha,
            diagnostics,
            outcome.candidate_rows,
            outcome.summary,
        )

    if selector == "stability_consensus":
        outcome = select_stability_consensus(
            candidates,
            cleanup_fn=lambda vertices: _fit_archetypal_from_init(
                g,
                vertices=vertices,
                max_iter=max(10, max_iter // 3),
                tol=tol,
                ridge=ridge,
            ),
        )
        alpha, diagnostics = _evaluate_fit(
            g,
            selection_points,
            outcome.selected_vertices,
            candidate_name=outcome.selected_name,
        )
        return (
            outcome.selected_vertices,
            alpha,
            diagnostics,
            outcome.candidate_rows,
            outcome.summary,
        )
    raise ValueError(f"Unsupported simplex selector: {selector}")


def fit_heuristic_simplex(g: np.ndarray, num_vertices: int) -> SimplexFitResult:
    """Use extreme points as simplex vertices and recover barycentric coordinates."""
    vertices = farthest_point_vertices(g, num_vertices=num_vertices)
    alpha, diagnostics = _evaluate_fit(g, g, vertices, candidate_name="heuristic")
    candidate = _candidate_record(
        name="heuristic",
        source="full",
        vertices=vertices,
        diagnostics=diagnostics,
        num_fit_points=len(g),
    )
    return SimplexFitResult(
        vertices=vertices,
        barycentric_coordinates=alpha,
        pi_hat=_estimate_pi_from_vertices(vertices),
        diagnostics=diagnostics,
        preprocessing=None,
        candidates=[candidate],
    )


def fit_archetypal_simplex(
    g: np.ndarray,
    num_vertices: int,
    max_iter: int = 60,
    tol: float = 1e-5,
    ridge: float = 1e-4,
    num_restarts: int = 8,
    seed: int = 0,
    selection_points: np.ndarray | None = None,
    include_heuristic_candidate: bool = True,
    selector: str = "restart_only",
    num_subsamples: int = 0,
    subsample_fraction: float = 0.7,
    use_bootstrap: bool = False,
    candidate_strategies: list[str] | None = None,
    corner_pool_fraction: float = 0.25,
    spread_num_projections: int = 12,
    spread_top_per_direction: int = 2,
) -> SimplexFitResult:
    """Alternating simplex factorization with row-simplex vertex constraints."""
    selection = g if selection_points is None else selection_points
    rng = np.random.default_rng(seed)
    candidates: list[SelectionCandidate] = []
    strategies = candidate_strategies or ["farthest"]
    single_strategy = len(strategies) == 1 and strategies[0] == "farthest"

    heuristic_vertices = farthest_point_vertices(g, num_vertices=num_vertices)
    if include_heuristic_candidate:
        alpha, diagnostics = _evaluate_fit(
            g,
            selection,
            heuristic_vertices,
            candidate_name="heuristic",
        )
        candidates.append(
            _candidate_record(
                name="heuristic",
                source="full",
                vertices=heuristic_vertices,
                diagnostics=diagnostics,
                num_fit_points=len(g),
            )
        )

    restart_counts = _split_count(num_restarts, len(strategies))
    for strategy, strategy_restarts in zip(strategies, restart_counts, strict=True):
        candidate_seeds = [None]
        if strategy == "farthest" and strategy_restarts > 1:
            candidate_seeds.extend(
                rng.choice(len(g), size=strategy_restarts - 1, replace=False).tolist()
            )
        else:
            candidate_seeds.extend([None] * max(strategy_restarts - 1, 0))

        for restart_index, first_index in enumerate(candidate_seeds):
            init_vertices = _strategy_vertices(
                strategy,
                g,
                num_vertices=num_vertices,
                rng=rng,
                first_index=first_index,
                corner_pool_fraction=corner_pool_fraction,
                spread_num_projections=spread_num_projections,
                spread_top_per_direction=spread_top_per_direction,
            )
            vertices = _fit_archetypal_from_init(
                g,
                vertices=init_vertices,
                max_iter=max_iter,
                tol=tol,
                ridge=ridge,
            )
            candidate_name = (
                f"archetypal_restart_{restart_index}"
                if single_strategy
                else f"{strategy}_restart_{restart_index}"
            )
            _, diagnostics = _evaluate_fit(g, selection, vertices, candidate_name=candidate_name)
            candidates.append(
                _candidate_record(
                    name=candidate_name,
                    source="full",
                    vertices=vertices,
                    diagnostics=diagnostics,
                    num_fit_points=len(g),
                )
            )

    if num_subsamples > 0:
        min_points = max(num_vertices * 4, num_vertices + 1)
        subsample_counts = _split_count(num_subsamples, len(strategies))
        for strategy, strategy_subsamples in zip(strategies, subsample_counts, strict=True):
            for subsample_index in range(strategy_subsamples):
                indices = _subsample_indices(
                    len(g),
                    rng=rng,
                    subsample_fraction=subsample_fraction,
                    use_bootstrap=use_bootstrap,
                    min_points=min_points,
                )
                subset = g[indices]
                init_first = (
                    int(rng.integers(0, len(subset)))
                    if strategy == "farthest" and len(subset) < len(g)
                    else None
                )
                init_vertices = _strategy_vertices(
                    strategy,
                    subset,
                    num_vertices=num_vertices,
                    rng=rng,
                    first_index=init_first,
                    corner_pool_fraction=corner_pool_fraction,
                    spread_num_projections=spread_num_projections,
                    spread_top_per_direction=spread_top_per_direction,
                )
                vertices = _fit_archetypal_from_init(
                    subset,
                    vertices=init_vertices,
                    max_iter=max_iter,
                    tol=tol,
                    ridge=ridge,
                )
                candidate_name = (
                    f"subsample_{subsample_index}"
                    if single_strategy
                    else f"{strategy}_subsample_{subsample_index}"
                )
                _, diagnostics = _evaluate_fit(g, selection, vertices, candidate_name=candidate_name)
                candidates.append(
                    _candidate_record(
                        name=candidate_name,
                        source="bootstrap" if use_bootstrap else "subsample",
                        vertices=vertices,
                        diagnostics=diagnostics,
                        num_fit_points=len(indices),
                    )
                )

    best_vertices, best_alpha, best_diagnostics, candidate_rows, selection_summary = (
        _select_vertices_from_candidates(
            candidates=candidates,
            selector=selector,
            g=g,
            selection_points=selection,
            max_iter=max_iter,
            tol=tol,
            ridge=ridge,
        )
    )
    return SimplexFitResult(
        vertices=best_vertices,
        barycentric_coordinates=best_alpha,
        pi_hat=_estimate_pi_from_vertices(best_vertices),
        diagnostics=best_diagnostics,
        preprocessing=None,
        candidates=candidates,
        candidate_rows=candidate_rows,
        selection_summary=selection_summary,
    )


def _kde_density(points: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    """Cheap Gaussian KDE evaluated at the input points.

    Uses Scott's rule for the bandwidth if not given. Distances are computed
    in the ambient space (already low-D after the optional affine projection).
    Returns un-normalized density (proportional to a Gaussian-smoothed point
    count) so it is robust to dimensionality.
    """
    n, d = points.shape
    if bandwidth is None:
        std = float(np.mean(np.std(points, axis=0)))
        bandwidth = max(std * (n ** (-1.0 / max(d + 4, 5))), 1e-3)
    diff = points[:, None, :] - points[None, :, :]
    sq = np.sum(diff * diff, axis=-1)
    weights = np.exp(-0.5 * sq / (bandwidth * bandwidth))
    return weights.sum(axis=1)


def _ransac_line_segment(
    points: np.ndarray,
    rng: np.random.Generator,
    *,
    num_iters: int,
    inlier_radius: float,
    min_inliers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Find a single dense line segment via RANSAC.

    Returns (anchor, direction, inlier_indices) on success, else None. The
    anchor and direction parameterise the infinite line; the segment is
    determined by the projections of inliers onto the direction.
    """
    n = len(points)
    if n < max(min_inliers, 2):
        return None
    best_inliers: np.ndarray | None = None
    for _ in range(num_iters):
        idx = rng.choice(n, size=2, replace=False)
        a = points[idx[0]]
        b = points[idx[1]]
        direction = b - a
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            continue
        direction = direction / norm
        diffs = points - a[None, :]
        proj = diffs @ direction
        residual = diffs - proj[:, None] * direction[None, :]
        dist = np.linalg.norm(residual, axis=1)
        inliers = np.where(dist < inlier_radius)[0]
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers
    if best_inliers is None or len(best_inliers) < min_inliers:
        return None
    inlier_pts = points[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    centered = inlier_pts - centroid[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    return centroid, direction, best_inliers


def _segment_endpoints(
    points: np.ndarray,
    anchor: np.ndarray,
    direction: np.ndarray,
    quantile: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two endpoints of the densely populated portion of a line."""
    proj = (points - anchor[None, :]) @ direction
    lo = float(np.quantile(proj, quantile))
    hi = float(np.quantile(proj, 1.0 - quantile))
    return anchor + lo * direction, anchor + hi * direction


def fit_archetypal_edges_simplex(
    g: np.ndarray,
    num_vertices: int,
    *,
    seed: int = 0,
    selection_points: np.ndarray | None = None,
    num_ransac_iters: int = 200,
    inlier_radius: float | None = None,
    min_inliers_fraction: float = 0.05,
    suppression_radius: float | None = None,
    refine_max_iter: int = 60,
    refine_tol: float = 1e-5,
    refine_ridge: float = 1e-4,
    fallback_archetypal: bool = True,
) -> SimplexFitResult:
    """E1: Edge-line RANSAC simplex fitter.

    Detects up to C(K,2) dense line segments via RANSAC in the affine span
    of the data, clusters their 2K(K-1) endpoints into K groups, and uses
    the cluster medoids as initial vertices. ALS refinement follows.
    Falls back to the standard archetypal fit on degenerate input.
    """
    selection = g if selection_points is None else selection_points
    rng = np.random.default_rng(seed)
    n = len(g)

    affine_dim = max(num_vertices - 1, 1)
    proj = fit_affine_projection(g, affine_dim=affine_dim) if g.shape[1] > affine_dim else None
    if proj is not None:
        centered = (g - proj.mean[None, :]) @ proj.basis
    else:
        centered = g - g.mean(axis=0, keepdims=True)

    radius = inlier_radius
    if radius is None:
        spread = float(np.mean(np.std(centered, axis=0)))
        radius = max(spread * 0.08, 1e-3)
    min_inliers = max(int(min_inliers_fraction * n), num_vertices + 1)

    target_lines = num_vertices * (num_vertices - 1) // 2
    if target_lines == 0:
        target_lines = 1

    available_mask = np.ones(n, dtype=bool)
    discovered: list[tuple[np.ndarray, np.ndarray]] = []

    for _ in range(target_lines):
        active_idx = np.where(available_mask)[0]
        if len(active_idx) < min_inliers:
            break
        result = _ransac_line_segment(
            centered[active_idx],
            rng,
            num_iters=num_ransac_iters,
            inlier_radius=radius,
            min_inliers=min_inliers,
        )
        if result is None:
            break
        anchor, direction, local_inliers = result
        ep0, ep1 = _segment_endpoints(centered[active_idx[local_inliers]], anchor, direction)
        discovered.append((ep0, ep1))
        # suppress *only* points well within the segment to avoid eating shared corners
        proj_scalar = (centered - anchor[None, :]) @ direction
        residual = (centered - anchor[None, :]) - proj_scalar[:, None] * direction[None, :]
        radial = np.linalg.norm(residual, axis=1)
        seg_lo, seg_hi = np.sort([float(proj_scalar[active_idx[local_inliers]].min()),
                                  float(proj_scalar[active_idx[local_inliers]].max())])
        margin = 0.15 * (seg_hi - seg_lo + 1e-12)
        on_segment = (
            (radial < radius)
            & (proj_scalar > seg_lo + margin)
            & (proj_scalar < seg_hi - margin)
        )
        available_mask &= ~on_segment

    if len(discovered) < num_vertices - 1:
        if fallback_archetypal:
            return fit_archetypal_simplex(
                g,
                num_vertices=num_vertices,
                seed=seed,
                selection_points=selection,
                num_restarts=4,
                candidate_strategies=["farthest", "spread"],
            )
        raise RuntimeError("Edge RANSAC failed to discover enough line segments")

    endpoint_arr = np.stack([p for ep in discovered for p in ep], axis=0)

    # cluster endpoints into K groups (vertices)
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=num_vertices, n_init=10, random_state=seed)
    kmeans.fit(endpoint_arr)
    centers_low = kmeans.cluster_centers_

    # lift back to ambient space
    if proj is not None:
        init_vertices = proj.mean[None, :] + centers_low @ proj.basis.T
    else:
        init_vertices = centers_low + g.mean(axis=0, keepdims=True)
    init_vertices = project_rows_to_simplex(init_vertices)

    refined = _fit_archetypal_from_init(
        g,
        vertices=init_vertices,
        max_iter=refine_max_iter,
        tol=refine_tol,
        ridge=refine_ridge,
    )

    alpha, diagnostics = _evaluate_fit(g, selection, refined, candidate_name="archetypal_edges")
    candidate = _candidate_record(
        name="archetypal_edges",
        source="full",
        vertices=refined,
        diagnostics=diagnostics,
        num_fit_points=len(g),
    )
    candidate_rows, selection_summary = _candidate_rows_for_single_fit(
        candidate, selected_name="archetypal_edges"
    )
    selection_summary.update(
        {
            "fit_method": "archetypal_edges",
            "num_segments_found": len(discovered),
            "ransac_iters": int(num_ransac_iters),
            "inlier_radius": float(radius),
        }
    )
    return SimplexFitResult(
        vertices=refined,
        barycentric_coordinates=alpha,
        pi_hat=_estimate_pi_from_vertices(refined),
        diagnostics=diagnostics,
        preprocessing=None,
        candidates=[candidate],
        candidate_rows=candidate_rows,
        selection_summary=selection_summary,
    )


def _fit_archetypal_weighted(
    g: np.ndarray,
    vertices: np.ndarray,
    weights: np.ndarray,
    max_iter: int,
    tol: float,
    ridge: float,
) -> np.ndarray:
    """Density-weighted ALS refinement.

    The vertex update minimises Sigma_i w_i ||g_i - alpha_i V||^2 so dense regions pull
    vertices more strongly, while the alpha update keeps the simplex
    constraint via projection.
    """
    prev_error = float("inf")
    w = weights.astype(np.float64)
    w = w / max(float(w.mean()), 1e-12)
    for _ in range(max_iter):
        alpha = batch_simplex_least_squares(g, vertices)
        wa = alpha * w[:, None]
        lhs = wa.T @ alpha + ridge * np.eye(vertices.shape[0])
        rhs = wa.T @ g
        vertices = np.linalg.solve(lhs, rhs)
        vertices = project_rows_to_simplex(vertices)
        residual = g - alpha @ vertices
        weighted_err = float(np.mean(w * np.linalg.norm(residual, axis=1)))
        if abs(prev_error - weighted_err) < tol:
            break
        prev_error = weighted_err
    return vertices


def fit_archetypal_density_simplex(
    g: np.ndarray,
    num_vertices: int,
    *,
    seed: int = 0,
    selection_points: np.ndarray | None = None,
    bandwidth: float | None = None,
    density_power: float = 1.0,
    init_strategies: list[str] | None = None,
    num_restarts: int = 3,
    refine_max_iter: int = 40,
    refine_tol: float = 1e-5,
    refine_ridge: float = 1e-4,
) -> SimplexFitResult:
    """E2: Density-weighted volume maximisation simplex fitter.

    1) Compute a KDE density at every point.
    2) Generate candidate initial vertices via several archetypal strategies.
    3) For each candidate, run ALS where the vertex update is weighted by
       density^density_power so dense edge mass pulls the vertices outward
       to enclose it.
    4) Pick the candidate with the lowest *unweighted* reconstruction-on-
       selection score (so we do not reward overfitting to dense regions).
    """
    selection = g if selection_points is None else selection_points
    rng = np.random.default_rng(seed)
    strategies = init_strategies or ["farthest", "spread", "corner"]
    weights = _kde_density(g, bandwidth=bandwidth) ** float(density_power)

    candidates: list[SelectionCandidate] = []
    for strategy in strategies:
        for restart in range(max(1, num_restarts // len(strategies))):
            init_first = (
                int(rng.integers(0, len(g))) if strategy == "farthest" and restart > 0 else None
            )
            init_vertices = _strategy_vertices(
                strategy,
                g,
                num_vertices=num_vertices,
                rng=rng,
                first_index=init_first,
                corner_pool_fraction=0.25,
                spread_num_projections=12,
                spread_top_per_direction=2,
            )
            vertices = _fit_archetypal_weighted(
                g,
                vertices=init_vertices,
                weights=weights,
                max_iter=refine_max_iter,
                tol=refine_tol,
                ridge=refine_ridge,
            )
            name = f"archetypal_density_{strategy}_{restart}"
            _, diag = _evaluate_fit(g, selection, vertices, candidate_name=name)
            candidates.append(
                _candidate_record(
                    name=name,
                    source="full",
                    vertices=vertices,
                    diagnostics=diag,
                    num_fit_points=len(g),
                )
            )

    selected = min(candidates, key=lambda c: c.selection_score)
    alpha, diagnostics = _evaluate_fit(
        g, selection, selected.vertices, candidate_name=selected.name
    )
    rows, summary = _candidate_rows_for_single_fit(selected, selected_name=selected.name)
    summary.update(
        {
            "fit_method": "archetypal_density",
            "num_candidates": len(candidates),
            "density_power": float(density_power),
        }
    )
    return SimplexFitResult(
        vertices=selected.vertices,
        barycentric_coordinates=alpha,
        pi_hat=_estimate_pi_from_vertices(selected.vertices),
        diagnostics=diagnostics,
        preprocessing=None,
        candidates=candidates,
        candidate_rows=rows,
        selection_summary=summary,
    )


def fit_archetypal_edges_density_simplex(
    g: np.ndarray,
    num_vertices: int,
    *,
    seed: int = 0,
    selection_points: np.ndarray | None = None,
    bandwidth: float | None = None,
    density_power: float = 1.0,
    edges_kwargs: dict | None = None,
) -> SimplexFitResult:
    """Combined: edge-RANSAC initial vertices + density-weighted refinement."""
    edges_kwargs = edges_kwargs or {}
    edges_fit = fit_archetypal_edges_simplex(
        g,
        num_vertices=num_vertices,
        seed=seed,
        selection_points=selection_points,
        **edges_kwargs,
    )
    weights = _kde_density(g, bandwidth=bandwidth) ** float(density_power)
    refined = _fit_archetypal_weighted(
        g,
        vertices=edges_fit.vertices.copy(),
        weights=weights,
        max_iter=80,
        tol=1e-5,
        ridge=1e-4,
    )
    selection = g if selection_points is None else selection_points
    alpha, diagnostics = _evaluate_fit(
        g, selection, refined, candidate_name="archetypal_edges_density"
    )
    candidate = _candidate_record(
        name="archetypal_edges_density",
        source="full",
        vertices=refined,
        diagnostics=diagnostics,
        num_fit_points=len(g),
    )
    rows, summary = _candidate_rows_for_single_fit(
        candidate, selected_name="archetypal_edges_density"
    )
    summary.update({"fit_method": "archetypal_edges_density"})
    return SimplexFitResult(
        vertices=refined,
        barycentric_coordinates=alpha,
        pi_hat=_estimate_pi_from_vertices(refined),
        diagnostics=diagnostics,
        preprocessing=None,
        candidates=[candidate],
        candidate_rows=rows,
        selection_summary=summary,
    )


def fit_simplex(
    g: np.ndarray,
    num_vertices: int,
    config: dict,
    selection_points: np.ndarray | None = None,
    seed: int = 0,
) -> SimplexFitResult:
    """Dispatch configured simplex fitter."""
    preprocessing: AffineProjection | None = None
    method_points = g
    method_selection = selection_points
    preprocess = str(config.get("preprocess", "none"))
    if preprocess == "affine":
        preprocess_dim = int(config.get("preprocess_dim", -1))
        preprocessing = fit_affine_projection(
            g,
            affine_dim=max(num_vertices - 1, 1) if preprocess_dim <= 0 else preprocess_dim,
        )
        if preprocessing is not None:
            method_points = preprocessing.transform(g)
            if selection_points is not None:
                method_selection = preprocessing.transform(selection_points)
    method = str(config["method"])
    if method == "heuristic":
        result = fit_heuristic_simplex(method_points, num_vertices=num_vertices)
        result.preprocessing = preprocessing
        return result
    if method == "archetypal":
        result = fit_archetypal_simplex(
            method_points,
            num_vertices=num_vertices,
            max_iter=int(config.get("max_iter", 60)),
            tol=float(config.get("tol", 1e-5)),
            ridge=float(config.get("ridge", 1e-4)),
            num_restarts=int(config.get("num_restarts", 8)),
            seed=seed,
            selection_points=method_selection,
            include_heuristic_candidate=bool(config.get("include_heuristic_candidate", True)),
            selector=str(config.get("selector", "restart_only")),
            num_subsamples=int(config.get("num_subsamples", 0)),
            subsample_fraction=float(config.get("subsample_fraction", 0.7)),
            use_bootstrap=bool(config.get("use_bootstrap", False)),
            candidate_strategies=list(config.get("candidate_strategies", ["farthest"])),
            corner_pool_fraction=float(config.get("corner_pool_fraction", 0.25)),
            spread_num_projections=int(config.get("spread_num_projections", 12)),
            spread_top_per_direction=int(config.get("spread_top_per_direction", 2)),
        )
        result.preprocessing = preprocessing
        return result
    if method == "archetypal_edges":
        result = fit_archetypal_edges_simplex(
            method_points,
            num_vertices=num_vertices,
            seed=seed,
            selection_points=method_selection,
            num_ransac_iters=int(config.get("num_ransac_iters", 200)),
            inlier_radius=config.get("inlier_radius", None),
            min_inliers_fraction=float(config.get("min_inliers_fraction", 0.05)),
            suppression_radius=config.get("suppression_radius", None),
            refine_max_iter=int(config.get("refine_max_iter", 60)),
            refine_tol=float(config.get("refine_tol", 1e-5)),
            refine_ridge=float(config.get("refine_ridge", 1e-4)),
            fallback_archetypal=bool(config.get("fallback_archetypal", True)),
        )
        result.preprocessing = preprocessing
        return result
    if method == "archetypal_density":
        result = fit_archetypal_density_simplex(
            method_points,
            num_vertices=num_vertices,
            seed=seed,
            selection_points=method_selection,
            bandwidth=config.get("bandwidth", None),
            density_power=float(config.get("density_power", 1.0)),
            init_strategies=list(config.get("init_strategies", ["farthest", "spread", "corner"])),
            num_restarts=int(config.get("num_restarts", 6)),
            refine_max_iter=int(config.get("refine_max_iter", 80)),
            refine_tol=float(config.get("refine_tol", 1e-5)),
            refine_ridge=float(config.get("refine_ridge", 1e-4)),
        )
        result.preprocessing = preprocessing
        return result
    if method == "archetypal_edges_density":
        result = fit_archetypal_edges_density_simplex(
            method_points,
            num_vertices=num_vertices,
            seed=seed,
            selection_points=method_selection,
            bandwidth=config.get("bandwidth", None),
            density_power=float(config.get("density_power", 1.0)),
            edges_kwargs=dict(config.get("edges_kwargs", {})),
        )
        result.preprocessing = preprocessing
        return result
    if method == "constrained_pi":
        result = fit_constrained_pi_simplex(
            method_points,
            num_vertices=num_vertices,
            max_outer_iters=int(config.get("max_outer_iters", 30)),
            inner_steps=int(config.get("inner_steps", 20)),
            lr=float(config.get("lr", 0.05)),
            separation_weight=float(config.get("separation_weight", 1e-3)),
            seed=seed,
            selection_points=method_selection,
            weighted=bool(config.get("weighted", False)),
            corner_weight=float(config.get("corner_weight", 2.0)),
            init_mode=str(config.get("init_mode", "archetypal")),
        )
        result.preprocessing = preprocessing
        return result
    raise ValueError(f"Unsupported simplex method: {method}")


def transform_simplex_points(points: np.ndarray, fit: SimplexFitResult) -> np.ndarray:
    """Apply the fitted simplex preprocessing to new points when present."""
    if fit.preprocessing is None:
        return points
    return fit.preprocessing.transform(points)


def decode_posteriors(g: np.ndarray, fit: SimplexFitResult) -> np.ndarray:
    """Decode source posteriors into latent-class posteriors via a fitted simplex.

    This is the canonical "apply a fitted simplex to new posteriors" step: fit the
    simplex once on a train/val posterior cloud, then call this on any held-out
    posteriors to get latent-class coordinates.

    Parameters
    ----------
    g : (N, M) source posteriors P(m | x), e.g. from your own M-way classifier.
    fit : SimplexFitResult returned by :func:`fit_simplex`.

    Returns
    -------
    (N, K) latent-class posteriors alpha(x) (barycentric coordinates against the
    fitted vertices, projected onto the simplex).
    """
    points = transform_simplex_points(g, fit)
    return batch_simplex_least_squares(points, fit.vertices)
