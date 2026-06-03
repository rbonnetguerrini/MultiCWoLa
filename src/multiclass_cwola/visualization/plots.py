"""Matplotlib plotting utilities."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _project_arrays_2d(*arrays: np.ndarray | None) -> tuple[np.ndarray | None, ...]:
    """Project arrays into a shared 2D coordinate system for overlays."""
    non_empty = [array for array in arrays if array is not None and len(array) > 0]
    if not non_empty:
        return tuple(None for _ in arrays)
    feature_dim = non_empty[0].shape[1]
    if any(array.shape[1] != feature_dim for array in non_empty):
        raise ValueError("All arrays must have the same feature dimension for plotting.")
    if feature_dim == 1:
        return tuple(
            None if array is None else np.column_stack([array[:, 0], np.zeros(len(array))])
            for array in arrays
        )
    if feature_dim == 2:
        return tuple(None if array is None else array[:, :2] for array in arrays)
    if feature_dim == 3:
        # Equilateral-triangle barycentric projection for the probability simplex Delta^2.
        def _bary(a: np.ndarray) -> np.ndarray:
            a = a / np.clip(a.sum(axis=1, keepdims=True), 1e-8, None)
            return np.column_stack([a[:, 1] + 0.5 * a[:, 2], (np.sqrt(3) / 2) * a[:, 2]])
        return tuple(None if array is None else _bary(array) for array in arrays)
    projector = PCA(n_components=2).fit(np.vstack(non_empty))
    return tuple(None if array is None else projector.transform(array) for array in arrays)


def _decision_axes_Mspace(vertices: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (start, end) pairs in M-space for decision axes: barycenter to edge midpoint."""
    K = len(vertices)
    barycenter = vertices.mean(axis=0)
    axes = []
    for i in range(K):
        for j in range(i + 1, K):
            axes.append((barycenter, (vertices[i] + vertices[j]) / 2))
    return axes


def plot_pi_heatmap(
    pi: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "Pi_hat (column-stochastic)",
    class_labels: list[str] | None = None,
    source_labels: list[str] | None = None,
) -> None:
    """Save a heatmap of an MxK mixing matrix Pi."""
    M, K = pi.shape
    fig, ax = plt.subplots(figsize=(max(4, 0.5 * K + 2), max(3, 0.4 * M + 1.5)))
    im = ax.imshow(pi, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xlabel("latent class k")
    ax.set_ylabel("source m")
    ax.set_xticks(range(K))
    ax.set_yticks(range(M))
    if class_labels is not None:
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
    if source_labels is not None:
        ax.set_yticklabels(source_labels)
    for i in range(M):
        for j in range(K):
            ax.text(
                j, i, f"{pi[i, j]:.2f}", ha="center", va="center",
                color="white" if pi[i, j] < 0.5 else "black", fontsize=7,
            )
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _draw_simplex_edges(
    vertices_2d: np.ndarray,
    *,
    color: str,
    linestyle: str,
    marker: str,
    label: str,
) -> None:
    plt.scatter(vertices_2d[:, 0], vertices_2d[:, 1], c=color, marker=marker, s=100)
    if len(vertices_2d) < 2:
        plt.scatter([], [], c=color, marker=marker, label=label)
        return
    label_pending = True
    for i in range(len(vertices_2d)):
        for j in range(i + 1, len(vertices_2d)):
            plt.plot(
                [vertices_2d[i, 0], vertices_2d[j, 0]],
                [vertices_2d[i, 1], vertices_2d[j, 1]],
                color=color,
                linewidth=1.2,
                linestyle=linestyle,
                alpha=0.75,
                label=label if label_pending else None,
            )
            label_pending = False


def _project_kspace_2d(alpha: np.ndarray) -> np.ndarray:
    """Project K-dimensional simplex coordinates to 2D for visualization.

    K=2: direct; K=3: equilateral-triangle barycentric projection; K>3: PCA.
    """
    K = alpha.shape[1]
    if K == 2:
        return alpha[:, :2]
    if K == 3:
        row_sum = alpha.sum(axis=1, keepdims=True)
        a = alpha / np.maximum(row_sum, 1e-12)
        x = a[:, 1] + 0.5 * a[:, 2]
        y = (np.sqrt(3) / 2) * a[:, 2]
        return np.column_stack([x, y])
    return PCA(n_components=2).fit_transform(alpha)


def plot_kspace_scatter(
    alpha: np.ndarray,
    labels: np.ndarray | None,
    out_path: str | Path,
    max_points: int = 1500,
    title: str = "Decoded latent posterior",
    label_name: str = "class",
) -> None:
    """Scatter plot of decoded latent-class coordinates alpha(x) in Delta^{K-1}.

    Oracle vertices are the standard one-hot basis {e_1,...,e_K}.
    Projection: equilateral-triangle barycentric for K=3, PCA for K>3, direct for K=2.
    """
    K = alpha.shape[1]
    oracle_vertices = np.eye(K)
    subset = alpha[:max_points]
    subset_labels = labels[:max_points] if labels is not None else None

    points_2d = _project_kspace_2d(subset)
    oracle_2d = _project_kspace_2d(oracle_vertices)

    plt.figure(figsize=(6, 5))
    has_legend = False
    if subset_labels is None:
        plt.scatter(points_2d[:, 0], points_2d[:, 1], s=10, alpha=0.6)
    else:
        for class_id in np.unique(subset_labels):
            mask = subset_labels == class_id
            plt.scatter(
                points_2d[mask, 0], points_2d[mask, 1], s=10, alpha=0.6,
                label=f"{label_name} {class_id}",
            )
        has_legend = True
    _draw_simplex_edges(
        oracle_2d,
        color="#b23a48",
        linestyle="--",
        marker="D",
        label="oracle simplex",
    )
    has_legend = True
    if has_legend:
        plt.legend(frameon=False)
    plt.title(title)
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_mspace_simplex(
    posteriors: np.ndarray,
    fitted_vertices: np.ndarray,
    oracle_vertices: np.ndarray | None,
    labels: np.ndarray | None,
    out_path: str | Path,
    max_points: int = 2000,
    title: str = "Posterior geometry (M-space)",
    label_name: str = "class",
) -> None:
    """M-space simplex geometry plot (paper figure Panel a / Panel b).

    Post-hoc (Panel a): posteriors = g(x) (N,M), fitted_vertices = post-hoc fit (K,M),
      oracle_vertices = bundle.source_given_class (K,M).
    Bottleneck (Panel b): posteriors = alpha(x) @ V_hat (N,M), fitted_vertices = V_hat (K,M),
      oracle_vertices = None.

    Decision axes (K=3): barycenter to edge midpoints, computed in M-space before projection.
    Projection: equilateral-triangle barycentric for M=3, shared PCA for M>3.
    """
    subset = posteriors[:max_points]
    subset_labels = labels[:max_points] if labels is not None else None

    # Compute decision-axis endpoints in M-space before projecting.
    fitted_axes = _decision_axes_Mspace(fitted_vertices)
    oracle_axes = _decision_axes_Mspace(oracle_vertices) if oracle_vertices is not None else []

    # Stack all axis endpoints into arrays for shared projection.
    fitted_axis_starts = np.array([s for s, _ in fitted_axes]) if fitted_axes else None
    fitted_axis_ends = np.array([e for _, e in fitted_axes]) if fitted_axes else None
    oracle_axis_starts = np.array([s for s, _ in oracle_axes]) if oracle_axes else None
    oracle_axis_ends = np.array([e for _, e in oracle_axes]) if oracle_axes else None

    projected = _project_arrays_2d(
        subset,
        fitted_vertices,
        oracle_vertices,
        fitted_axis_starts,
        fitted_axis_ends,
        oracle_axis_starts,
        oracle_axis_ends,
    )
    (
        pts_2d,
        fitted_verts_2d,
        oracle_verts_2d,
        f_starts_2d,
        f_ends_2d,
        o_starts_2d,
        o_ends_2d,
    ) = projected

    fig, ax = plt.subplots(figsize=(6, 5))

    # Points colored by class label.
    if subset_labels is None:
        ax.scatter(pts_2d[:, 0], pts_2d[:, 1], s=8, alpha=0.5, rasterized=True)
    else:
        for class_id in np.unique(subset_labels):
            mask = subset_labels == class_id
            ax.scatter(pts_2d[mask, 0], pts_2d[mask, 1], s=8, alpha=0.5,
                       label=f"{label_name} {class_id}", rasterized=True)

    # Fitted decision axes (transparent, no legend entry).
    if f_starts_2d is not None and f_ends_2d is not None:
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for idx in range(len(f_starts_2d)):
            c = color_cycle[idx % len(color_cycle)]
            ax.plot(
                [f_starts_2d[idx, 0], f_ends_2d[idx, 0]],
                [f_starts_2d[idx, 1], f_ends_2d[idx, 1]],
                color=c, linewidth=0.8, linestyle="-", alpha=0.25,
            )

    # Fitted simplex edges and vertices.
    if fitted_verts_2d is not None:
        K = len(fitted_verts_2d)
        ax.scatter(fitted_verts_2d[:, 0], fitted_verts_2d[:, 1], marker="x",
                   s=100, color="tab:blue", zorder=5)
        label_pending = True
        for i in range(K):
            for j in range(i + 1, K):
                ax.plot(
                    [fitted_verts_2d[i, 0], fitted_verts_2d[j, 0]],
                    [fitted_verts_2d[i, 1], fitted_verts_2d[j, 1]],
                    color="tab:blue", linewidth=1.2, linestyle="-", alpha=0.85,
                    label="recovered simplex" if label_pending else None,
                )
                label_pending = False

    # Oracle decision axes (transparent black dashed, no legend entry).
    if o_starts_2d is not None and o_ends_2d is not None:
        for idx in range(len(o_starts_2d)):
            ax.plot(
                [o_starts_2d[idx, 0], o_ends_2d[idx, 0]],
                [o_starts_2d[idx, 1], o_ends_2d[idx, 1]],
                color="black", linewidth=0.8, linestyle="--", alpha=0.25,
            )

    # Oracle simplex edges and vertices.
    if oracle_verts_2d is not None:
        K = len(oracle_verts_2d)
        ax.scatter(oracle_verts_2d[:, 0], oracle_verts_2d[:, 1], marker="D",
                   s=60, color="black", zorder=5)
        label_pending = True
        for i in range(K):
            for j in range(i + 1, K):
                ax.plot(
                    [oracle_verts_2d[i, 0], oracle_verts_2d[j, 0]],
                    [oracle_verts_2d[i, 1], oracle_verts_2d[j, 1]],
                    color="black", linewidth=1.2, linestyle="--", alpha=0.85,
                    label="oracle simplex" if label_pending else None,
                )
                label_pending = False

    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_metric_curve(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: str | Path,
    title: str,
) -> None:
    """Line plot for sweep results."""
    plt.figure(figsize=(6, 4))
    plt.plot(frame[x_col], frame[y_col], marker="o")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

