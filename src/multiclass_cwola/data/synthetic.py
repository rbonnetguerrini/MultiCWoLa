"""Synthetic latent-class mixture data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from multiclass_cwola.data.common import MixtureDatasetBundle, SplitData, source_given_class_from_pi
from multiclass_cwola.data.mixtures import MixingMatrixInfo, generate_mixing_matrix


class ConditionalDistribution(Protocol):
    """Sampling and density API for latent class conditionals."""

    def sample(
        self, rng: np.random.Generator, num_samples: int, source_id: int | None = None
    ) -> np.ndarray:
        """Draw samples."""

    def log_prob(self, x: np.ndarray, source_id: int | None = None) -> np.ndarray:
        """Evaluate log density if available."""


@dataclass
class GaussianConditional:
    """Single Gaussian latent class."""

    mean: np.ndarray
    cov: np.ndarray

    def sample(
        self, rng: np.random.Generator, num_samples: int, source_id: int | None = None
    ) -> np.ndarray:
        return rng.multivariate_normal(self.mean, self.cov, size=num_samples)

    def log_prob(self, x: np.ndarray, source_id: int | None = None) -> np.ndarray:
        return multivariate_normal.logpdf(x, mean=self.mean, cov=self.cov)


@dataclass
class MixtureGaussianConditional:
    """Mixture-of-Gaussians latent class."""

    weights: np.ndarray
    means: np.ndarray
    covs: np.ndarray

    def sample(
        self, rng: np.random.Generator, num_samples: int, source_id: int | None = None
    ) -> np.ndarray:
        component_ids = rng.choice(len(self.weights), size=num_samples, p=self.weights)
        chunks = []
        for component_id in component_ids:
            chunks.append(
                rng.multivariate_normal(self.means[component_id], self.covs[component_id])
            )
        return np.asarray(chunks, dtype=np.float32)

    def log_prob(self, x: np.ndarray, source_id: int | None = None) -> np.ndarray:
        pieces = []
        for weight, mean, cov in zip(self.weights, self.means, self.covs, strict=True):
            pieces.append(
                np.log(weight + 1e-12) + multivariate_normal.logpdf(x, mean=mean, cov=cov)
            )
        return logsumexp(np.stack(pieces, axis=0), axis=0)


@dataclass
class ManifoldToyConditional:
    """Simple nonlinear toy distribution on a noisy circle arc."""

    radius: float
    angle_offset: float
    noise_scale: float

    def sample(
        self, rng: np.random.Generator, num_samples: int, source_id: int | None = None
    ) -> np.ndarray:
        angles = self.angle_offset + rng.uniform(-0.7, 0.7, size=num_samples)
        x = np.stack([self.radius * np.cos(angles), self.radius * np.sin(angles)], axis=1)
        x += rng.normal(scale=self.noise_scale, size=x.shape)
        return x.astype(np.float32)

    def log_prob(self, x: np.ndarray, source_id: int | None = None) -> np.ndarray:
        raise NotImplementedError("True posteriors are not implemented for manifold toys.")


def _random_covariance(
    rng: np.random.Generator, dim: int, scale: float, anisotropic: bool
) -> np.ndarray:
    if not anisotropic:
        return np.eye(dim) * (scale**2)
    base = rng.normal(size=(dim, dim))
    cov = base.T @ base
    cov /= np.trace(cov) / dim
    return cov * (scale**2)


def build_latent_conditionals(
    family: str,
    num_classes: int,
    input_dim: int,
    class_sep: float,
    noise_scale: float,
    mixture_components: int,
    seed: int,
) -> list[ConditionalDistribution]:
    """Create latent class conditional objects."""
    rng = np.random.default_rng(seed)
    conditionals: list[ConditionalDistribution] = []
    if family == "nonlinear_manifold":
        if input_dim != 2:
            raise ValueError("nonlinear_manifold requires input_dim=2")
        for index in range(num_classes):
            angle = 2.0 * np.pi * index / num_classes
            conditionals.append(
                ManifoldToyConditional(
                    radius=class_sep, angle_offset=angle, noise_scale=noise_scale
                )
            )
        return conditionals

    class_means = rng.normal(size=(num_classes, input_dim))
    class_means /= np.linalg.norm(class_means, axis=1, keepdims=True) + 1e-8
    class_means *= class_sep

    if family in {"isotropic_gaussian", "anisotropic_gaussian"}:
        anisotropic = family == "anisotropic_gaussian"
        for index in range(num_classes):
            cov = _random_covariance(rng, input_dim, noise_scale, anisotropic=anisotropic)
            conditionals.append(GaussianConditional(mean=class_means[index], cov=cov))
        return conditionals

    if family == "mog":
        for index in range(num_classes):
            weights = rng.dirichlet(np.ones(mixture_components))
            offsets = rng.normal(scale=0.5, size=(mixture_components, input_dim))
            means = class_means[index][None, :] + offsets
            covs = np.stack(
                [
                    _random_covariance(rng, input_dim, noise_scale, anisotropic=True)
                    for _ in range(mixture_components)
                ],
                axis=0,
            )
            conditionals.append(MixtureGaussianConditional(weights=weights, means=means, covs=covs))
        return conditionals

    raise ValueError(f"Unsupported synthetic family: {family}")


def _apply_shift(
    x: np.ndarray,
    source: np.ndarray,
    y: np.ndarray,
    shift_cfg: dict,
) -> np.ndarray:
    if not shift_cfg.get("enabled", False):
        return x
    shift_type = shift_cfg.get("type", "none")
    scale = float(shift_cfg.get("scale", 0.0))
    shifted = np.array(x, copy=True)
    if shift_type == "covariate":
        shifted += scale * source[:, None]
    elif shift_type == "class_conditional":
        shifted += scale * y[:, None]
    elif shift_type == "spurious":
        nuisance = np.zeros((len(x), 1), dtype=np.float32)
        nuisance[:, 0] = scale * (source.astype(np.float32) - source.mean())
        shifted = np.concatenate([shifted, nuisance], axis=1)
    return shifted


def _sample_split(
    rng: np.random.Generator,
    pi: np.ndarray,
    num_per_source: int,
    conditionals: list[ConditionalDistribution],
    shift_cfg: dict,
) -> SplitData:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    for source_id, source_priors in enumerate(pi):
        latent_y = rng.choice(len(conditionals), size=num_per_source, p=source_priors)
        x_chunks = []
        for class_id in latent_y:
            x_chunks.append(conditionals[class_id].sample(rng, 1, source_id=source_id)[0])
        x_source = np.asarray(x_chunks, dtype=np.float32)
        xs.append(x_source)
        ys.append(latent_y.astype(np.int64))
        sources.append(np.full(num_per_source, source_id, dtype=np.int64))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    source = np.concatenate(sources, axis=0)
    x = _apply_shift(x, source, y, shift_cfg)
    return SplitData(x=x.astype(np.float32), y=y, source=source)


def _true_posteriors(
    x: np.ndarray,
    pi: np.ndarray,
    conditionals: list[ConditionalDistribution],
) -> np.ndarray | None:
    try:
        class_priors = pi.mean(axis=0)
        logits = []
        for prior, conditional in zip(class_priors, conditionals, strict=True):
            logits.append(np.log(prior + 1e-12) + conditional.log_prob(x))
        log_probs = np.stack(logits, axis=1)
    except NotImplementedError:
        return None
    log_probs -= logsumexp(log_probs, axis=1, keepdims=True)
    return np.exp(log_probs)


def _true_source_posteriors(
    x: np.ndarray,
    pi: np.ndarray,
    conditionals: list[ConditionalDistribution],
) -> np.ndarray | None:
    try:
        logits = []
        for source_priors in pi:
            class_terms = []
            for prior, conditional in zip(source_priors, conditionals, strict=True):
                class_terms.append(np.log(prior + 1e-12) + conditional.log_prob(x))
            logits.append(logsumexp(np.stack(class_terms, axis=1), axis=1))
        log_probs = np.stack(logits, axis=1)
    except NotImplementedError:
        return None
    log_probs -= logsumexp(log_probs, axis=1, keepdims=True)
    return np.exp(log_probs)


def build_synthetic_dataset(config: dict) -> MixtureDatasetBundle:
    """Build a synthetic dataset bundle from config."""
    info: MixingMatrixInfo = generate_mixing_matrix(
        num_sources=int(config["M"]),
        num_classes=int(config["K"]),
        mode=str(config.get("pi_mode", "dirichlet")),
        dirichlet_alpha=float(config.get("pi_dirichlet_alpha", 0.8)),
        similarity=float(config.get("pi_similarity", 0.9)),
        condition_strength=float(config.get("condition_strength", 0.25)),
        seed=int(config.get("seed", 0)),
    )
    conditionals = build_latent_conditionals(
        family=str(config["family"]),
        num_classes=int(config["K"]),
        input_dim=int(config["input_dim"]),
        class_sep=float(config["class_sep"]),
        noise_scale=float(config["noise_scale"]),
        mixture_components=int(config.get("mixture_components", 2)),
        seed=int(config.get("seed", 0)),
    )
    rng = np.random.default_rng(int(config.get("seed", 0)))
    train = _sample_split(
        rng, info.pi, int(config["train_per_source"]), conditionals, dict(config["shift"])
    )
    val = _sample_split(
        rng, info.pi, int(config["val_per_source"]), conditionals, dict(config["shift"])
    )
    test = _sample_split(
        rng, info.pi, int(config["test_per_source"]), conditionals, dict(config["shift"])
    )
    train.true_posteriors = _true_posteriors(train.x, info.pi, conditionals)
    val.true_posteriors = _true_posteriors(val.x, info.pi, conditionals)
    test.true_posteriors = _true_posteriors(test.x, info.pi, conditionals)
    train.true_source_posteriors = _true_source_posteriors(train.x, info.pi, conditionals)
    val.true_source_posteriors = _true_source_posteriors(val.x, info.pi, conditionals)
    test.true_source_posteriors = _true_source_posteriors(test.x, info.pi, conditionals)

    return MixtureDatasetBundle(
        train=train,
        val=val,
        test=test,
        pi=info.pi,
        source_given_class=source_given_class_from_pi(info.pi),
        metadata={
            "family": config["family"],
            "rank": info.rank,
            "condition_number": info.condition_number,
            "row_similarity": info.row_similarity,
        },
        input_shape=(train.x.shape[1],),
        task_type="tabular",
        num_classes=int(config["K"]),
        num_sources=int(config["M"]),
    )
