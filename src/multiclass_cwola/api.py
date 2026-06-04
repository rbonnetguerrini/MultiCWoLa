"""High-level, sklearn-style API for prior-free multiclass CWoLa.

The estimator wraps the existing pipeline building blocks (source-classifier
training, posterior calibration, simplex fitting, barycentric decoding) behind
a small ``fit`` / ``predict`` surface so the method can be applied to a user's
own data without Hydra configs or benchmark scaffolding.

The only supervision required is the **mixture identity** of each example::

    from multiclass_cwola import MultiCWoLa

    model = MultiCWoLa(K=3).fit(X, source_ids)   # no class labels, no priors
    model.class_posteriors_     # alpha(x), shape (N, K)
    model.pi_                   # recovered mixing matrix Pi_hat, the science target
    model.predict(X_new)        # latent-class predictions (up to permutation)
    print(model.report())       # A1/A2/A3 trust diagnostics

Two recovery modes (paper Sec. 3.2):

  * ``mode="posthoc"`` (default, R1): train an unconstrained M-way classifier and
    fit a simplex to its posterior cloud after the fact. Works with any backbone.
  * ``mode="bottleneck"`` (R2): train a classifier whose head factorises as
    ``g(x) = Pi @ alpha(x)`` with ``Pi`` column-stochastic and ``alpha(x)`` on the
    (K-1)-simplex, so the geometry is enforced during training. Requires an
    internal trainable backbone (``"auto"``/``"mlp"``/``"cnn"``).

Backbone-agnostic input (the simplex machinery only needs ``g(x) = P(m|x)``):

  * ``backbone="auto"`` (default) trains an internal MLP (tabular ``X``) or CNN
    (image ``X``) source classifier.
  * ``backbone="mlp"`` / ``"cnn"`` force the internal architecture.
  * ``backbone="precomputed"`` treats ``X`` as already-computed source
    posteriors ``g`` (N, M) -- bring any model's outputs.
  * ``backbone=<estimator>`` uses any fitted object exposing
    ``predict_proba(X)`` (scikit-learn, XGBoost, ...).
  * ``backbone=<callable>`` uses ``g = backbone(X)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from multiclass_cwola.calibration.posthoc import calibrate_logits, fit_calibrator
from multiclass_cwola.diagnostics import DiagnosticsReport, compute_diagnostics
from multiclass_cwola.evaluation.matching import align_probabilities, match_label_permutation
from multiclass_cwola.simplex.fitters import (
    fit_simplex,
    transform_simplex_points,
)
from multiclass_cwola.simplex.projection import batch_simplex_least_squares

# Embedded simplex presets so the API needs no YAML files. "constrained_pi" is
# the paper's strongest prior-free post-hoc fitter ("Regular + constr. Pi_hat").
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

_DEFAULT_TRAIN = {
    "batch_size": 64,
    "epochs": 20,
    "lr": 1.0e-3,
    "weight_decay": 0.01,
    "early_stopping_patience": 5,
    "num_workers": 0,
    "bottleneck_warmup_epochs": 0,
    "scheduler": {"type": "cosine_warmup", "warmup_frac": 0.05, "min_lr_ratio": 0.0},
}


def _as_probs(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    array = np.clip(array, 1e-12, None)
    return array / array.sum(axis=1, keepdims=True)


class MultiCWoLa:
    """Prior-free multiclass classifier from unlabeled mixtures.

    Parameters
    ----------
    K : int
        Number of latent classes to recover (``K <= M``).
    backbone : str | object | callable, default ``"auto"``
        Source-posterior producer. See module docstring for the options.
    simplex : str | dict, default ``"constrained_pi"``
        Post-hoc fitter preset name or an explicit fitter config dict.
    calibration : str | None, default ``"temperature"``
        Posterior calibrator (``"temperature"``, ``"vector"``, or ``None``).
    device : str, default ``"auto"``
        Torch device for the internal classifier path.
    seed : int, default 0
    val_fraction : float, default 0.1
        Held-out fraction (by index) used to fit the calibrator.
    model_kwargs, train_kwargs : dict, optional
        Overrides for the internal classifier architecture / training loop.

    Attributes (set after ``fit``)
    ------------------------------
    source_posteriors_ : (N, M) calibrated ``g(x)``.
    class_posteriors_ : (N, K) decoded latent posteriors ``alpha(x)``.
    pi_ : (M, K) recovered row-stochastic mixing matrix (may be ``None``).
    vertices_ : (K, M) fitted simplex vertices.
    """

    def __init__(
        self,
        K: int,
        *,
        mode: str = "posthoc",
        backbone: str | Any = "auto",
        simplex: str | dict[str, Any] = "constrained_pi",
        calibration: str | None = "temperature",
        device: str = "auto",
        seed: int = 0,
        val_fraction: float = 0.1,
        model_kwargs: dict[str, Any] | None = None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if int(K) < 2:
            raise ValueError("K must be >= 2.")
        if mode not in {"posthoc", "bottleneck"}:
            raise ValueError("mode must be 'posthoc' or 'bottleneck'.")
        self.K = int(K)
        self.mode = mode
        self.backbone = backbone
        self.simplex = simplex
        self.calibration = calibration
        self.device = device
        self.seed = int(seed)
        self.val_fraction = float(val_fraction)
        self.model_kwargs = dict(model_kwargs or {})
        self.train_kwargs = dict(train_kwargs or {})

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, source: np.ndarray, y: np.ndarray | None = None) -> MultiCWoLa:
        """Fit the recovery from features/posteriors ``X`` and mixture ids ``source``.

        ``y`` (true class labels) is optional and used only to align the
        permutation-arbitrary latent classes for :meth:`predict` / :meth:`score`.
        """
        from multiclass_cwola.utils.repro import seed_everything

        seed_everything(self.seed)
        source = np.asarray(source).astype(int)
        if source.ndim != 1:
            raise ValueError("source must be a 1-D array of mixture identities.")
        self.num_sources_ = int(source.max()) + 1
        if self.K > self.num_sources_:
            raise ValueError(f"K={self.K} exceeds number of mixtures M={self.num_sources_}.")

        if self.mode == "bottleneck":
            self._fit_bottleneck(X, source, y)
            return self

        logits = self._fit_producer(X, source)  # (N, M) logits or log-probs
        calibrated = self._fit_and_apply_calibration(logits, source)
        self.source_posteriors_ = calibrated
        self._fit_geometry(calibrated, source, y)
        return self

    def fit_posteriors(
        self, g: np.ndarray, source: np.ndarray, y: np.ndarray | None = None
    ) -> MultiCWoLa:
        """Fit directly from precomputed source posteriors ``g`` (N, M).

        Convenience wrapper for ``backbone="precomputed"``.
        """
        self.backbone = "precomputed"
        return self.fit(g, source, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return latent posteriors ``alpha(x)`` (N, K), permutation-aligned if ``y`` was given."""
        self._check_fitted()
        alpha = self._alpha_for(X)
        if getattr(self, "alignment_", None):
            alpha = align_probabilities(alpha, self.alignment_, num_classes=self.K)
        return alpha

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard latent-class predictions (N,)."""
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Permutation-aligned latent accuracy against true labels ``y``."""
        self._check_fitted()
        y = np.asarray(y).astype(int)
        alpha = self._alpha_for(X)
        _, mapping = match_label_permutation(y, alpha.argmax(axis=1), num_classes=self.K)
        aligned = align_probabilities(alpha, mapping, num_classes=self.K)
        return float((aligned.argmax(axis=1) == y).mean())

    def report(self) -> DiagnosticsReport:
        """Compute A1/A2/A3 trust diagnostics for the fitted recovery."""
        self._check_fitted()
        if self.fit_ is not None and self.fit_.diagnostics is not None:
            recon = self.fit_.diagnostics.reconstruction_error
        else:
            # Bottleneck has no post-hoc fit: residual of g vs alpha @ vertices.
            reconstruction = self.class_posteriors_ @ self.vertices_
            recon = float(
                np.mean(np.linalg.norm(self.source_posteriors_ - reconstruction, axis=1))
            )
        return compute_diagnostics(
            source_posteriors=self.source_posteriors_,
            class_posteriors=self.class_posteriors_,
            vertices=self.vertices_,
            source_ids=self.source_ids_,
            pi_hat=self.pi_,
            reconstruction_error=recon,
        )

    # ------------------------------------------------------------------ #
    # Producer (backbone-agnostic) -- returns logits / log-probs (N, M)
    # ------------------------------------------------------------------ #
    def _fit_producer(self, X: np.ndarray, source: np.ndarray) -> np.ndarray:
        backbone = self.backbone
        if isinstance(backbone, str) and backbone in {"auto", "mlp", "cnn"}:
            return self._fit_internal_classifier(X, source, backbone)
        if isinstance(backbone, str) and backbone == "precomputed":
            self._producer_kind = "precomputed"
            return np.log(_as_probs(np.asarray(X)))
        if callable(getattr(backbone, "predict_proba", None)):
            self._producer_kind = "estimator"
            return np.log(_as_probs(backbone.predict_proba(X)))
        if callable(backbone):
            self._producer_kind = "callable"
            return np.log(_as_probs(backbone(X)))
        raise TypeError(
            "backbone must be 'auto'/'mlp'/'cnn'/'precomputed', a callable, "
            "or an object with predict_proba."
        )

    def _fit_internal_classifier(
        self, X: np.ndarray, source: np.ndarray, backbone: str
    ) -> np.ndarray:
        self._producer_kind = "internal"
        return self._train_internal(X, source, enable_bottleneck=False).logits

    def _train_internal(self, X: np.ndarray, source: np.ndarray, *, enable_bottleneck: bool):
        """Build + train the internal classifier; return its inference outputs.

        Sets ``self._model`` and ``self._device``. With ``enable_bottleneck`` the
        head factorises ``g(x) = Pi @ alpha(x)`` and returns ``log q(x)``.
        """
        from multiclass_cwola.models.backbones import build_classifier
        from multiclass_cwola.training.trainer import infer_classifier, train_classifier
        from multiclass_cwola.utils.repro import choose_device

        if isinstance(self.backbone, str) and self.backbone in {"mlp", "cnn"}:
            backbone = self.backbone
        else:
            backbone = "auto"
        X = np.asarray(X, dtype=np.float32)
        chosen = "mlp" if (backbone == "auto" and X.ndim == 2) else (
            "cnn" if backbone == "auto" else backbone
        )
        input_shape = X.shape[1:] if X.ndim > 1 else (X.shape[-1],)
        model_cfg: dict[str, Any] = {"backbone": chosen}
        model_cfg |= self.model_kwargs
        if enable_bottleneck:
            bottleneck_cfg = {"enabled": True}
            bottleneck_cfg |= dict(model_cfg.get("bottleneck") or {})
            bottleneck_cfg["enabled"] = True
            model_cfg["bottleneck"] = bottleneck_cfg
        self._device = choose_device(self.device)

        model = build_classifier(
            model_cfg,
            input_shape=tuple(input_shape),
            num_outputs=self.num_sources_,
            num_classes=self.K if enable_bottleneck else None,
            enable_bottleneck=enable_bottleneck,
        )
        train_idx, val_idx = self._split_indices(len(X), source)
        train_cfg = dict(_DEFAULT_TRAIN) | self.train_kwargs
        train_classifier(
            model,
            X[train_idx],
            source[train_idx],
            X[val_idx],
            source[val_idx],
            train_cfg,
            device=self._device,
            num_classes_K=self.K if enable_bottleneck else None,
        )
        self._model = model
        return infer_classifier(
            model, X, batch_size=int(train_cfg["batch_size"]), device=self._device
        )

    def _fit_bottleneck(self, X: np.ndarray, source: np.ndarray, y: np.ndarray | None) -> None:
        """Recovery R2: read alpha(x) and Pi straight off the trained bottleneck head."""
        import torch

        from multiclass_cwola.simplex.fitters import _estimate_pi_from_vertices

        if not (isinstance(self.backbone, str) and self.backbone in {"auto", "mlp", "cnn"}):
            raise ValueError(
                "mode='bottleneck' requires a trainable backbone "
                "('auto'/'mlp'/'cnn'); got a precomputed/external producer."
            )
        self._producer_kind = "internal"
        outputs = self._train_internal(X, source, enable_bottleneck=True)
        head = self._model.head
        if not hasattr(head, "alpha") or not hasattr(head, "pi"):
            raise RuntimeError(
                "bottleneck head missing; ensure model.bottleneck stays enabled."
            )
        with torch.no_grad():
            embeddings = torch.as_tensor(outputs.embeddings, dtype=torch.float32).to(self._device)
            alpha = head.alpha(embeddings).cpu().numpy()
            pi_col = head.pi().detach().cpu().numpy()  # (M, K) column-stochastic

        self._calibration = None
        self.source_posteriors_ = _as_probs(outputs.probabilities)  # q(x) = exp(log q)
        self.fit_ = None
        self.vertices_ = pi_col.T  # (K, M)
        self.pi_ = _estimate_pi_from_vertices(self.vertices_)  # (M, K) row-stochastic
        self.class_posteriors_ = alpha
        self.source_ids_ = source
        self.alignment_ = None
        if y is not None:
            y = np.asarray(y).astype(int)
            _, mapping = match_label_permutation(
                y, alpha.argmax(axis=1), num_classes=self.K
            )
            self.alignment_ = mapping

    def _produce_posteriors(self, X: np.ndarray) -> np.ndarray:
        """Map new ``X`` to calibrated source posteriors using the fitted producer."""
        kind = self._producer_kind
        if kind == "precomputed":
            # X is already a (N, M) posterior; re-decode it (optionally recalibrated).
            logits = np.log(_as_probs(np.asarray(X)))
        elif kind == "internal":
            from multiclass_cwola.training.trainer import infer_classifier

            logits = infer_classifier(
                self._model,
                np.asarray(X, dtype=np.float32),
                batch_size=int((dict(_DEFAULT_TRAIN) | self.train_kwargs)["batch_size"]),
                device=self._device,
            ).logits
        elif kind == "estimator":
            logits = np.log(_as_probs(self.backbone.predict_proba(X)))
        elif kind == "callable":
            logits = np.log(_as_probs(self.backbone(X)))
        else:  # pragma: no cover - guarded by _fit_producer
            raise RuntimeError(f"Unknown producer kind: {kind}")
        if self._calibration is None:
            return _as_probs(logits)
        return calibrate_logits(logits, self._calibration)

    # ------------------------------------------------------------------ #
    # Calibration + geometry
    # ------------------------------------------------------------------ #
    def _fit_and_apply_calibration(self, logits: np.ndarray, source: np.ndarray) -> np.ndarray:
        if self.calibration is None:
            self._calibration = None
            return _as_probs(logits)
        _, val_idx = self._split_indices(len(logits), source)
        self._calibration = fit_calibrator(
            logits[val_idx], source[val_idx], method=self.calibration
        )
        return calibrate_logits(logits, self._calibration)

    def _fit_geometry(
        self, g: np.ndarray, source: np.ndarray, y: np.ndarray | None
    ) -> None:
        config = self._resolve_simplex_config()
        self.fit_ = fit_simplex(g, num_vertices=self.K, config=config, seed=self.seed)
        self.vertices_ = self.fit_.vertices
        self.pi_ = self.fit_.pi_hat
        self.class_posteriors_ = self._decode(g)
        self.source_ids_ = source
        self.alignment_ = None
        if y is not None:
            y = np.asarray(y).astype(int)
            _, mapping = match_label_permutation(
                y, self.class_posteriors_.argmax(axis=1), num_classes=self.K
            )
            self.alignment_ = mapping

    def _decode(self, g: np.ndarray) -> np.ndarray:
        points = transform_simplex_points(g, self.fit_)
        return batch_simplex_least_squares(points, self.fit_.vertices)

    def _alpha_for(self, X: np.ndarray) -> np.ndarray:
        """Map new ``X`` to (unaligned) latent posteriors ``alpha(x)``."""
        if self.mode == "bottleneck":
            import torch

            from multiclass_cwola.training.trainer import infer_classifier

            batch_size = int((dict(_DEFAULT_TRAIN) | self.train_kwargs)["batch_size"])
            embeddings = infer_classifier(
                self._model,
                np.asarray(X, dtype=np.float32),
                batch_size=batch_size,
                device=self._device,
            ).embeddings
            with torch.no_grad():
                tensor = torch.as_tensor(embeddings, dtype=torch.float32).to(self._device)
                return self._model.head.alpha(tensor).cpu().numpy()
        return self._decode(self._produce_posteriors(X))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _resolve_simplex_config(self) -> dict[str, Any]:
        if isinstance(self.simplex, dict):
            return dict(self.simplex)
        if self.simplex in _SIMPLEX_PRESETS:
            return dict(_SIMPLEX_PRESETS[self.simplex])
        raise ValueError(
            f"Unknown simplex preset '{self.simplex}'. "
            f"Choose from {sorted(_SIMPLEX_PRESETS)} or pass a config dict."
        )

    def _split_indices(self, n: int, source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic train/val split; falls back to all-as-val for tiny n."""
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(n)
        n_val = max(1, int(round(self.val_fraction * n)))
        if n_val >= n:
            return order, order
        return order[n_val:], order[:n_val]

    def _check_fitted(self) -> None:
        if not hasattr(self, "class_posteriors_"):
            raise RuntimeError("MultiCWoLa is not fitted yet; call fit(...) first.")
