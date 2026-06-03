"""Regularizers for stage-1 source-classifier training under an unfrozen
backbone. Each piece is an opt-in flag that defaults to a no-op so existing
runs are unaffected.

Pieces:
  - build_train_transform: torchvision augmentation pipelines suitable for
    image inputs (e.g. Galaxy10 -- class-symmetry-respecting transforms).
  - intra_mixture_mixup: in-batch Mixup that blends only same-mixture samples,
    so the mixture label is preserved exactly (fair signal). Returns
    blended inputs and either the original hard target (soft_targets=False)
    or a soft target stack ready for soft cross-entropy.
  - smooth_one_hot: produce smoothed mixture targets for label smoothing.
  - SAM: Sharpness-Aware Minimization wrapper over a base optimizer.
  - geometric_subspace_residual: 1 - (energy in top K-1 singular components)
    for a stack of M-dim posteriors. Increases as the source classifier
    overfits the mixture-ID and drifts off the (K-1) class-simplex.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn


# --- Augmentation -----------------------------------------------------------


def build_train_transform(kind: str | None) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Return a torchvision-style transform applied per batch on-device, or None.

    Inputs are already-normalized float tensors with shape (B, C, H, W) in
    [0, 1] (the ResNet50 backbone applies ImageNet normalization downstream).
    """
    if not kind or str(kind).lower() in {"none", "off", "false"}:
        return None
    kind = str(kind).lower()

    try:
        from torchvision.transforms import v2 as T
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torchvision.transforms.v2 is required for augmentation. Install a recent torchvision."
        ) from exc

    if kind == "symmetry":
        # Pure label-symmetry augmentation: rotation + h/v flip only.
        # No crops or color jitter -- preserves pixel statistics so source
        # posteriors keep their full range and simplex vertices stay extremal.
        pipeline = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=180),
            ]
        )
    elif kind == "symmetry_plus":
        # Posterior-preserving extension of `symmetry`. Adds class-invariant
        # geometric jitter (translate, shear) and physics-motivated stochastic
        # ops (small Gaussian blur, additive noise) that keep pixel mean fixed
        # so simplex vertices stay extremal. No mixup, no smoothing, no
        # photometric jitter -- those would compress the simplex.
        geometric = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomAffine(
                    degrees=180,
                    translate=(0.06, 0.06),
                    shear=4,
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
            ]
        )
        noise_std = 0.015

        def _apply_symmetry_plus(batch: torch.Tensor) -> torch.Tensor:
            if batch.dim() != 4:
                return batch
            out = geometric(batch)
            out = out + noise_std * torch.randn_like(out)
            return out.clamp_(0.0, 1.0)

        return _apply_symmetry_plus
    elif kind == "galaxy":
        # Galaxy class identity is exactly invariant to rotation/flip/crop and
        # approximately invariant under modest brightness/contrast jitter.
        pipeline = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=180),
                T.RandomResizedCrop(size=224, scale=(0.7, 1.0), antialias=True),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            ]
        )
    elif kind == "mild":
        pipeline = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomResizedCrop(size=224, scale=(0.85, 1.0), antialias=True),
            ]
        )
    else:
        raise ValueError(f"Unsupported augmentation kind: {kind}")

    def _apply(batch: torch.Tensor) -> torch.Tensor:
        if batch.dim() != 4:
            return batch
        return pipeline(batch)

    return _apply


# --- Intra-mixture Mixup ----------------------------------------------------


def intra_mixture_mixup(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    alpha: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend each sample with another sample from the SAME mixture in the
    same batch. Mixture label is preserved exactly; returns (mixed_x, batch_y).

    If a sample has no same-mixture partner in the batch (unique mixture in
    this batch), it is left unmodified.
    """
    if alpha <= 0.0:
        return batch_x, batch_y
    device = batch_x.device
    bs = batch_x.shape[0]
    perm = torch.arange(bs, device=device)
    # Group indices by mixture id and shuffle within each group.
    unique_ids = torch.unique(batch_y)
    for mid in unique_ids:
        idx = (batch_y == mid).nonzero(as_tuple=False).flatten()
        if idx.numel() <= 1:
            continue
        shuffled = idx[torch.randperm(idx.numel(), device=device, generator=generator)]
        perm[idx] = shuffled
    lam = float(np.random.default_rng().beta(alpha, alpha))
    mixed = lam * batch_x + (1.0 - lam) * batch_x[perm]
    return mixed, batch_y


# --- Label smoothing --------------------------------------------------------


def smooth_one_hot(labels: torch.Tensor, num_classes: int, epsilon: float) -> torch.Tensor:
    """Convert int labels to a smoothed one-hot target."""
    if epsilon <= 0.0:
        oh = torch.zeros((labels.shape[0], num_classes), device=labels.device)
        oh.scatter_(1, labels.long().unsqueeze(1), 1.0)
        return oh
    off = epsilon / max(num_classes, 1)
    on = 1.0 - epsilon + off
    oh = torch.full((labels.shape[0], num_classes), off, device=labels.device)
    oh.scatter_(1, labels.long().unsqueeze(1), on)
    return oh


# --- SAM --------------------------------------------------------------------


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al. 2021) over a base optimizer.

    Usage:
        loss = criterion(model(x), y);  loss.backward()
        opt.first_step(zero_grad=True)
        criterion(model(x), y).backward()
        opt.second_step(zero_grad=True)
    """

    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs):
        if rho <= 0:
            raise ValueError("SAM rho must be > 0")
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        # Keep group dicts in sync (lr scheduler updates ours; mirror to base).
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def _grad_norm(self) -> torch.Tensor:
        device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm(p=2).to(device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm


# --- Geometric early-stopping monitor ---------------------------------------


def geometric_subspace_residual(posteriors: np.ndarray, num_classes: int) -> float:
    """Fraction of variance NOT captured by the top (K-1) right singular
    vectors of the mixture-posterior cloud. Theory says g(x) lives on a
    (K-1)-dim affine subspace of the (M-1)-simplex; this metric grows when
    the source classifier overfits the mixture ID and the cloud expands
    toward simplex corners.
    """
    if posteriors.size == 0 or num_classes <= 1:
        return float("nan")
    centered = posteriors - posteriors.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    total = float(np.square(s).sum())
    if total <= 0:
        return float("nan")
    keep = max(0, int(num_classes) - 1)
    captured = float(np.square(s[:keep]).sum()) if keep > 0 else 0.0
    return 1.0 - captured / total
