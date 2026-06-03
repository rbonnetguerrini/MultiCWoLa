"""Stability-aware selection utilities for simplex candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from multiclass_cwola.simplex.projection import project_rows_to_simplex


@dataclass
class SelectionCandidate:
    """One fitted simplex candidate with comparable diagnostics."""

    name: str
    source: str
    vertices: np.ndarray
    selection_score: float
    reconstruction_error: float
    raw_reconstruction_error: float
    vertex_condition_number: float
    min_vertex_distance: float
    raw_negative_fraction: float
    raw_negative_mass: float
    projection_gap: float
    mean_entropy: float
    collapse_fraction: float
    degenerate: bool
    num_fit_points: int


@dataclass
class SelectionOutcome:
    """Selected candidate or consensus simplex plus diagnostics."""

    selected_name: str
    selected_vertices: np.ndarray
    selected_source: str
    candidate_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def align_vertices(
    reference_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Align candidate vertices to a reference ordering by minimum distance."""
    cost = np.linalg.norm(
        reference_vertices[:, None, :] - candidate_vertices[None, :, :],
        axis=-1,
    )
    ref_idx, cand_idx = linear_sum_assignment(cost)
    aligned = np.zeros_like(candidate_vertices)
    permutation = np.zeros(len(candidate_vertices), dtype=np.int64)
    for ref_vertex, cand_vertex in zip(ref_idx, cand_idx, strict=True):
        aligned[ref_vertex] = candidate_vertices[cand_vertex]
        permutation[ref_vertex] = cand_vertex
    mean_distance = float(np.mean(np.linalg.norm(reference_vertices - aligned, axis=1)))
    return aligned, permutation, mean_distance


def _pairwise_candidate_distances(candidates: list[SelectionCandidate]) -> np.ndarray:
    matrix = np.zeros((len(candidates), len(candidates)), dtype=np.float64)
    for i, left in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            _, _, distance = align_vertices(left.vertices, candidates[j].vertices)
            matrix[i, j] = distance
            matrix[j, i] = distance
    return matrix


def _normalize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum - minimum < 1e-12:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def _candidate_rows(
    candidates: list[SelectionCandidate],
    pairwise_distances: np.ndarray,
    cluster_reference: int,
    cluster_members: np.ndarray,
    selected_name: str,
) -> list[dict[str, Any]]:
    nonzero = pairwise_distances[np.triu_indices_from(pairwise_distances, k=1)]
    positive = nonzero[nonzero > 0]
    support_radius = float(np.median(positive)) if len(positive) else 0.0
    avg_distances = pairwise_distances.mean(axis=1)
    local_scores = np.asarray([candidate.selection_score for candidate in candidates], dtype=np.float64)
    normalized_local = _normalize(local_scores)
    normalized_distance = _normalize(avg_distances)

    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        within_radius = (
            np.ones(len(candidates), dtype=bool)
            if support_radius <= 0.0
            else pairwise_distances[index] <= support_radius
        )
        support_fraction = float(np.mean(within_radius))
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
                "avg_aligned_distance": float(avg_distances[index]),
                "distance_to_cluster_reference": float(pairwise_distances[cluster_reference, index]),
                "support_fraction": support_fraction,
                "cluster_member": bool(cluster_members[index]),
                "normalized_local_score": float(normalized_local[index]),
                "normalized_stability_distance": float(normalized_distance[index]),
                "selected": candidate.name == selected_name,
            }
        )
    return rows


def _composite_scores(
    candidates: list[SelectionCandidate],
    pairwise_distances: np.ndarray,
    viability_mask: np.ndarray,
) -> np.ndarray:
    avg_distances = pairwise_distances.mean(axis=1)
    normalized_distance = _normalize(avg_distances)
    local_scores = np.asarray([candidate.selection_score for candidate in candidates], dtype=np.float64)
    normalized_local = _normalize(local_scores)
    scores = normalized_distance + 0.25 * normalized_local
    scores = scores.astype(np.float64)
    scores[~viability_mask] += 10.0
    return scores


def _cluster_members(pairwise_distances: np.ndarray, medoid_index: int) -> np.ndarray:
    distances = pairwise_distances[medoid_index]
    positive = distances[distances > 0]
    if len(positive) == 0:
        return np.ones(len(distances), dtype=bool)
    threshold = float(np.median(positive))
    members = distances <= threshold
    members[medoid_index] = True
    return members


def select_stability_medoid(candidates: list[SelectionCandidate]) -> SelectionOutcome:
    """Choose the most stable viable candidate by pairwise aligned distance."""
    if not candidates:
        raise ValueError("No simplex candidates were provided.")
    pairwise_distances = _pairwise_candidate_distances(candidates)
    viability_mask = np.asarray([not candidate.degenerate for candidate in candidates], dtype=bool)
    if not np.any(viability_mask):
        viability_mask = np.ones(len(candidates), dtype=bool)
    scores = _composite_scores(candidates, pairwise_distances, viability_mask)
    medoid_index = int(np.argmin(scores))
    members = _cluster_members(pairwise_distances, medoid_index)
    selected = candidates[medoid_index]
    rows = _candidate_rows(
        candidates,
        pairwise_distances=pairwise_distances,
        cluster_reference=medoid_index,
        cluster_members=members,
        selected_name=selected.name,
    )
    summary = {
        "selection_mode": "stability_medoid",
        "selected_candidate": selected.name,
        "selected_source": selected.source,
        "num_candidates": len(candidates),
        "num_viable_candidates": int(np.sum(viability_mask)),
        "cluster_reference": candidates[medoid_index].name,
        "cluster_size": int(np.sum(members)),
        "cluster_support_fraction": float(np.mean(members)),
        "selected_composite_score": float(scores[medoid_index]),
        "selected_avg_aligned_distance": float(pairwise_distances.mean(axis=1)[medoid_index]),
    }
    return SelectionOutcome(
        selected_name=selected.name,
        selected_vertices=selected.vertices,
        selected_source=selected.source,
        candidate_rows=rows,
        summary=summary,
    )


def select_stability_consensus(
    candidates: list[SelectionCandidate],
    cleanup_fn: callable | None = None,
) -> SelectionOutcome:
    """Average aligned candidates in the dominant stable cluster."""
    medoid = select_stability_medoid(candidates)
    medoid_name = medoid.summary["cluster_reference"]
    medoid_index = next(
        index for index, candidate in enumerate(candidates) if candidate.name == medoid_name
    )
    pairwise_distances = _pairwise_candidate_distances(candidates)
    members = _cluster_members(pairwise_distances, medoid_index)
    aligned_vertices = [candidates[medoid_index].vertices]
    member_names = [candidates[medoid_index].name]
    for index, candidate in enumerate(candidates):
        if index == medoid_index or not members[index]:
            continue
        aligned, _, _ = align_vertices(candidates[medoid_index].vertices, candidate.vertices)
        aligned_vertices.append(aligned)
        member_names.append(candidate.name)
    consensus_vertices = np.mean(np.stack(aligned_vertices, axis=0), axis=0)
    consensus_vertices = project_rows_to_simplex(consensus_vertices)
    if cleanup_fn is not None:
        consensus_vertices = cleanup_fn(consensus_vertices)
    rows = medoid.candidate_rows
    for row in rows:
        row["selected"] = False
    selected_name = "stability_consensus"
    summary = dict(medoid.summary)
    summary.update(
        {
            "selection_mode": "stability_consensus",
            "selected_candidate": selected_name,
            "selected_source": "consensus",
            "cluster_member_names": member_names,
            "cluster_size": len(member_names),
            "cluster_support_fraction": len(member_names) / max(len(candidates), 1),
        }
    )
    return SelectionOutcome(
        selected_name=selected_name,
        selected_vertices=consensus_vertices,
        selected_source="consensus",
        candidate_rows=rows,
        summary=summary,
    )
