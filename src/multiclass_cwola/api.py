"""High-level MultiCWoLa: bring-your-own-model recovery, diagnostics, and plotting.

`MultiCWoLa` never trains or owns a model. You supply the model; the class provides
the *structure* around the two recovery methods plus alignment, trust diagnostics, and
mixture-vs-latent plotting. The only supervision required is the mixture identity of
each example — no class labels (optional, for alignment/eval) and no mixture proportions.

Two recovery methods (paper Sec. 3.2):

  * **Post-hoc** (R1) -- train any M-way classifier yourself, then hand its source
    posteriors ``g(x) = P(m|x)`` here:

        model = MultiCWoLa(K=3).fit_posteriors(g_train, source, y=y_optional)
        alpha_eval = model.predict_proba(g_eval)     # decode held-out posteriors
        model.report();  model.plot("out/", latent_labels=y, true_pi=pi)

  * **Bottleneck** (R2) -- attach :class:`BottleneckSimplexHead` to your PyTorch
    backbone, train it on mixture labels, then extract and hand over the result:

        alpha = head.predict_latent(features).cpu().numpy()   # (N, K)
        pi    = head.pi().detach().cpu().numpy()              # (M, K) column-stochastic
        model = MultiCWoLa(K=3).fit_bottleneck(alpha, pi, source, y=y_optional)

For users who prefer functions over a class, the same primitives are exported at the
package root: ``fit_simplex``, ``decode_posteriors``, ``align_latent_classes``,
``fit_calibrator`` / ``calibrate_logits``, and ``BottleneckSimplexHead``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from multiclass_cwola.diagnostics import DiagnosticsReport, compute_diagnostics
from multiclass_cwola.evaluation.matching import align_latent_classes, align_probabilities
from multiclass_cwola.simplex.fitters import (
    _estimate_pi_from_vertices,
    decode_posteriors,
    fit_simplex,
)
from multiclass_cwola.simplex.projection import source_given_class_from_pi
from multiclass_cwola.visualization.plots import (
    plot_kspace_scatter,
    plot_mixture_confusion,
    plot_mspace_simplex,
    plot_pi_heatmap,
)

# Embedded simplex presets so the API needs no YAML files. "constrained_pi" is the
# paper's strongest prior-free post-hoc fitter ("Regular + constr. Pi_hat").
_SIMPLEX_PRESETS: dict[str, dict[str, Any]] = {
    "constrained_pi": {
        "name": "constrained_pi",
        "method": "constrained_pi",
        "max_outer_iters": 30,
        "inner_steps": 20,
        "lr": 0.05,
        "separation_weight": 1.0e-3,
        "weighted": False,
        "corner_weight": 2.0,
        "init_mode": "archetypal",
        "preprocess": "none",
        "preprocess_dim": -1,
    },
    "archetypal_corner": {
        "name": "archetypal_corner",
        "method": "archetypal",
        "max_iter": 60,
        "tol": 1.0e-5,
        "ridge": 1.0e-4,
        "num_restarts": 8,
        "include_heuristic_candidate": True,
        "selector": "restart_only",
        "num_subsamples": 0,
        "subsample_fraction": 0.7,
        "use_bootstrap": False,
        "candidate_strategies": ["corner"],
        "corner_pool_fraction": 0.25,
        "spread_num_projections": 12,
        "spread_top_per_direction": 2,
        "preprocess": "none",
        "preprocess_dim": -1,
    },
}


def _as_probs(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    array = np.clip(array, 1e-12, None)
    return array / array.sum(axis=1, keepdims=True)


def _validate_source(source: np.ndarray) -> np.ndarray:
    source = np.asarray(source).astype(int)
    if source.ndim != 1:
        raise ValueError("source must be a 1-D array of mixture identities.")
    return source


class MultiCWoLa:
    """Bring-your-own-model multiclass recovery, diagnostics, and plotting.

    Parameters
    ----------
    K : int
        Number of latent classes to recover (``K <= M``).
    simplex : str | dict, default ``"constrained_pi"``
        Post-hoc fitter: a preset name (``"constrained_pi"``, ``"archetypal_corner"``),
        a bare method name (e.g. ``"archetypal"``), or an explicit fitter config dict.
        Unused by :meth:`fit_bottleneck`.
    seed : int, default 0

    Attributes (set after ``fit_*``)
    --------------------------------
    class_posteriors_ : (N, K) latent posteriors alpha(x); permutation-aligned if ``y`` given.
    pi_ : (M, K) row-stochastic recovered mixing matrix Pi_hat (the science target).
    vertices_ : (K, M) simplex vertices in source-posterior space.
    source_posteriors_ : (N, M) source posteriors g(x) (reconstructed for the bottleneck).
    source_ids_ : (N,) observed mixture identities.
    """

    def __init__(self, K: int, *, simplex: str | dict[str, Any] = "constrained_pi", seed: int = 0) -> None:
        if int(K) < 2:
            raise ValueError("K must be >= 2.")
        self.K = int(K)
        self.simplex = simplex
        self.seed = int(seed)

    # ------------------------------------------------------------------ #
    # Recovery
    # ------------------------------------------------------------------ #
    def fit_posteriors(
        self, g: np.ndarray, source: np.ndarray, y: np.ndarray | None = None
    ) -> MultiCWoLa:
        """Post-hoc recovery from your classifier's source posteriors ``g`` (N, M).

        Fits the posterior simplex to ``g`` and decodes latent posteriors. ``y`` (true
        class labels) is optional and only used to resolve the latent-class permutation.
        """
        g = _as_probs(np.asarray(g))
        source = _validate_source(source)
        if self.K > g.shape[1]:
            raise ValueError(f"K={self.K} exceeds number of mixtures M={g.shape[1]}.")
        self.num_sources_ = int(g.shape[1])
        self.fit_ = fit_simplex(
            g, num_vertices=self.K, config=self._resolve_simplex_config(), seed=self.seed
        )
        self.vertices_ = self.fit_.vertices
        self.pi_ = self.fit_.pi_hat
        self.source_posteriors_ = g
        self.source_ids_ = source
        self._mode = "posthoc"
        self._finalize(decode_posteriors(g, self.fit_), y)
        return self

    def fit_bottleneck(
        self, alpha: np.ndarray, pi: np.ndarray, source: np.ndarray, y: np.ndarray | None = None
    ) -> MultiCWoLa:
        """Bottleneck recovery from your trained model's extracted outputs.

        Parameters
        ----------
        alpha : (N, K) latent posteriors, e.g. ``head.predict_latent(features)``.
        pi : (M, K) column-stochastic mixing matrix, e.g. ``head.pi()``.
        source : (N,) mixture identities.
        y : optional (N,) true class labels for permutation alignment.
        """
        alpha = _as_probs(np.asarray(alpha))
        pi = np.asarray(pi, dtype=np.float64)
        source = _validate_source(source)
        if alpha.shape[1] != self.K:
            raise ValueError(f"alpha has {alpha.shape[1]} columns, expected K={self.K}.")
        if pi.shape[1] != self.K:
            raise ValueError(f"pi has {pi.shape[1]} columns, expected K={self.K}.")
        self.num_sources_ = int(pi.shape[0])
        self.fit_ = None
        self.vertices_ = pi.T  # (K, M) -- each row is a column-stochastic vertex
        self.pi_ = _estimate_pi_from_vertices(self.vertices_)  # (M, K) row-stochastic
        self.source_posteriors_ = alpha @ self.vertices_  # reconstructed g(x)
        self.source_ids_ = source
        self._mode = "bottleneck"
        self._finalize(alpha, y)
        return self

    # ------------------------------------------------------------------ #
    # Decode / predict (post-hoc held-out)
    # ------------------------------------------------------------------ #
    def predict_proba(self, g_eval: np.ndarray) -> np.ndarray:
        """Decode held-out source posteriors into latent posteriors (N, K).

        Post-hoc only. For the bottleneck, extract alpha from your own model with
        ``head.predict_latent(features)``.
        """
        self._check_fitted()
        if self._mode != "posthoc":
            raise RuntimeError(
                "predict_proba is post-hoc only; for the bottleneck extract alpha from "
                "your model via head.predict_latent(features)."
            )
        alpha = decode_posteriors(_as_probs(np.asarray(g_eval)), self.fit_)
        if self.alignment_:
            alpha = align_probabilities(alpha, self.alignment_, num_classes=self.K)
        return alpha

    def predict(self, g_eval: np.ndarray) -> np.ndarray:
        """Hard latent-class predictions for held-out posteriors (post-hoc only)."""
        return self.predict_proba(g_eval).argmax(axis=1)

    def score(self, g_eval: np.ndarray, y: np.ndarray) -> float:
        """Permutation-aligned latent accuracy on held-out posteriors (post-hoc only)."""
        self._check_fitted()
        if self._mode != "posthoc":
            raise RuntimeError("score is post-hoc only; evaluate bottleneck outputs directly.")
        y = np.asarray(y).astype(int)
        alpha = decode_posteriors(_as_probs(np.asarray(g_eval)), self.fit_)
        aligned, _ = align_latent_classes(alpha, y, num_classes=self.K)
        return float((aligned.argmax(axis=1) == y).mean())

    # ------------------------------------------------------------------ #
    # Diagnostics + plotting
    # ------------------------------------------------------------------ #
    def report(self) -> DiagnosticsReport:
        """Compute A1/A2/A3 trust diagnostics for the fitted recovery."""
        self._check_fitted()
        if self.fit_ is not None and self.fit_.diagnostics is not None:
            recon = self.fit_.diagnostics.reconstruction_error
        else:
            reconstruction = self.class_posteriors_ @ self.vertices_
            recon = float(np.mean(np.linalg.norm(self.source_posteriors_ - reconstruction, axis=1)))
        return compute_diagnostics(
            source_posteriors=self.source_posteriors_,
            class_posteriors=self.class_posteriors_,
            vertices=self.vertices_,
            source_ids=self.source_ids_,
            pi_hat=self.pi_,
            reconstruction_error=recon,
        )

    def plot(
        self,
        outdir: str | Path,
        *,
        latent_labels: np.ndarray | None = None,
        true_pi: np.ndarray | None = None,
    ) -> dict[str, Path]:
        """Write the mixture-vs-latent comparison figures.

        Parameters
        ----------
        outdir : directory to write PNGs into (created if missing).
        latent_labels : optional (N,) true latent classes; when given, points are
            coloured by latent class instead of mixture id.
        true_pi : optional (M, K) true row-stochastic mixing matrix; when given, the
            oracle simplex / oracle Pi heatmap are overlaid for comparison.

        Returns
        -------
        dict mapping a short key to each written file path.
        """
        self._check_fitted()
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        labels = np.asarray(latent_labels) if latent_labels is not None else self.source_ids_
        label_name = "class" if latent_labels is not None else "source"
        m, k = self.num_sources_, self.K
        paths: dict[str, Path] = {}

        paths["kspace"] = out / "kspace_posterior.png"
        plot_kspace_scatter(
            self.class_posteriors_, labels, paths["kspace"],
            title=f"Decoded latent posterior (K={k})", label_name=label_name,
        )

        oracle_vertices = (
            source_given_class_from_pi(np.asarray(true_pi)) if true_pi is not None else None
        )
        paths["mspace"] = out / "mspace_posterior.png"
        plot_mspace_simplex(
            self.source_posteriors_, self.vertices_, oracle_vertices, labels, paths["mspace"],
            title=f"Source posterior geometry (M={m}, K={k})", label_name=label_name,
        )

        if self.pi_ is not None:
            paths["pi_hat"] = out / "pi_hat_heatmap.png"
            plot_pi_heatmap(self.pi_, paths["pi_hat"], title=f"Pi_hat (M={m}, K={k})")
        if true_pi is not None:
            paths["pi_oracle"] = out / "pi_oracle_heatmap.png"
            plot_pi_heatmap(np.asarray(true_pi), paths["pi_oracle"], title=f"Pi oracle (M={m}, K={k})")

        paths["mixture_confusion"] = out / "mixture_confusion.png"
        plot_mixture_confusion(
            self.source_ids_, self.source_posteriors_.argmax(axis=1), paths["mixture_confusion"],
            pi=self.pi_, title=f"Mixture recovery (M={m})",
        )
        return paths

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _finalize(self, raw_alpha: np.ndarray, y: np.ndarray | None) -> None:
        """Store decoded posteriors and resolve the latent permutation if labels given."""
        self.alignment_ = None
        if y is None:
            self.class_posteriors_ = raw_alpha
            return
        y = np.asarray(y).astype(int)
        aligned, mapping = align_latent_classes(raw_alpha, y, num_classes=self.K)
        self.alignment_ = mapping
        self.class_posteriors_ = aligned

    def _resolve_simplex_config(self) -> dict[str, Any]:
        if isinstance(self.simplex, dict):
            return dict(self.simplex)
        if self.simplex in _SIMPLEX_PRESETS:
            return dict(_SIMPLEX_PRESETS[self.simplex])
        # Treat any other string as a bare fitter method name.
        return {"name": str(self.simplex), "method": str(self.simplex)}

    def _check_fitted(self) -> None:
        if not hasattr(self, "class_posteriors_"):
            raise RuntimeError("MultiCWoLa is not fitted; call fit_posteriors(...) or fit_bottleneck(...).")
