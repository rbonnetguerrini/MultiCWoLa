"""Training loops for source and latent classifiers."""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from multiclass_cwola.data.common import ArrayDataset
from multiclass_cwola.training.regularization import (
    SAM,
    build_train_transform,
    geometric_subspace_residual,
    intra_mixture_mixup,
    smooth_one_hot,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainHistory:
    """Training history summary."""

    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int
    runtime_seconds: float


@dataclass
class InferenceOutputs:
    """Model outputs extracted from a dataloader or array."""

    logits: np.ndarray
    probabilities: np.ndarray
    embeddings: np.ndarray


def _soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def _soft_nll(log_probs: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy when the model already returns log-probabilities."""
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def low_rank_tail_nuclear_penalty(
    outputs: torch.Tensor,
    regularizer_cfg: dict,
    *,
    returns_log_probs: bool = False,
) -> torch.Tensor:
    """Penalty for posterior mass outside the configured low-rank geometry."""
    if not bool(regularizer_cfg.get("enabled", False)):
        return outputs.new_tensor(0.0)
    weight = float(regularizer_cfg.get("weight", 0.0))
    if weight == 0.0:
        return outputs.new_tensor(0.0)

    centered = bool(regularizer_cfg.get("centered", True))
    target_rank_value = regularizer_cfg.get("target_rank", "auto")
    if target_rank_value in (None, "auto"):
        num_classes = regularizer_cfg.get("num_classes")
        if num_classes is None:
            raise ValueError(
                "low_rank_regularization requires num_classes when target_rank is auto"
            )
        target_rank = int(num_classes) - 1 if centered else int(num_classes)
    else:
        target_rank = int(target_rank_value)
    if target_rank < 0:
        raise ValueError("low_rank_regularization.target_rank must be non-negative")

    posteriors = outputs.exp() if returns_log_probs else torch.softmax(outputs, dim=-1)
    if centered:
        posteriors = posteriors - posteriors.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(posteriors)
    if target_rank >= singular_values.numel():
        return outputs.new_tensor(0.0)

    tail = singular_values[target_rank:].sum()
    normalize_by = str(regularizer_cfg.get("normalize_by", "sqrt_batch"))
    if normalize_by == "sqrt_batch":
        tail = tail / max(float(posteriors.shape[0]) ** 0.5, 1.0)
    elif normalize_by == "batch_size":
        tail = tail / max(float(posteriors.shape[0]), 1.0)
    elif normalize_by in {"none", "false"}:
        pass
    else:
        raise ValueError(f"Unknown low-rank normalization: {normalize_by}")
    return outputs.new_tensor(weight) * tail


def _build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_frac: float,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay to ``min_lr_ratio * lr``."""
    warmup_steps = max(1, int(total_steps * warmup_frac))
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / decay_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _make_loader(
    x: np.ndarray,
    labels: np.ndarray | None = None,
    soft_labels: np.ndarray | None = None,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = ArrayDataset(
        x=x,
        labels=np.zeros(len(x), dtype=np.int64) if labels is None else labels,
        soft_labels=soft_labels,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def train_classifier(
    model: nn.Module,
    train_x: np.ndarray,
    train_labels: np.ndarray,
    val_x: np.ndarray,
    val_labels: np.ndarray,
    train_cfg: dict,
    checkpoint_path: str | Path | None = None,
    reuse_checkpoint: bool = False,
    soft_labels: bool = False,
    device: torch.device | None = None,
    num_classes_K: int | None = None,
) -> TrainHistory:
    """Train a classifier with early stopping.

    Optional regularizers (each opt-in via ``train_cfg`` flags):
      - augmentation.kind: "galaxy" / "mild" -- torchvision transform on
        training batches (image inputs only). Default: None.
      - mixup.alpha: Beta(alpha, alpha) for intra-mixture Mixup. Default: 0.
      - sam.rho: enables SAM if > 0. Default: 0.
      - label_smoothing: eps for mixture-target smoothing. Default: 0.
      - early_stopping_metric: "val_loss" (default) or "geometric_residual"
        -- the latter monitors how well posteriors fit a (K-1) subspace
        and requires ``num_classes_K`` to be passed.
    """
    if device is None:
        device = torch.device("cpu")
    model.to(device)
    if reuse_checkpoint and checkpoint_path is not None and Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return TrainHistory(train_loss=[], val_loss=[], best_epoch=-1, runtime_seconds=0.0)
    backbone_lr = float(train_cfg.get("backbone_lr", train_cfg["lr"]))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    backbone_obj = getattr(model, "backbone", None)
    if backbone_obj is not None and hasattr(backbone_obj, "optimizer_param_groups"):
        param_groups = backbone_obj.optimizer_param_groups(
            head_lr=float(train_cfg["lr"]), backbone_lr=backbone_lr
        )
        head_module = getattr(model, "head", None)
        if head_module is not None:
            param_groups.append(
                {"params": list(head_module.parameters()), "lr": float(train_cfg["lr"])}
            )
    else:
        param_groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": float(train_cfg["lr"]),
            }
        ]

    # SAM wraps AdamW; default optimizer is plain AdamW.
    sam_cfg = (train_cfg.get("sam") or {}) if isinstance(train_cfg.get("sam"), dict) else {}
    sam_rho = float(sam_cfg.get("rho", 0.0)) if sam_cfg.get("enabled", True) else 0.0
    use_sam = sam_rho > 0.0
    if use_sam:
        optimizer = SAM(
            param_groups,
            torch.optim.AdamW,
            rho=sam_rho,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    returns_log_probs = bool(getattr(model, "returns_log_probs", False))

    # Label smoothing on mixture targets converts hard labels into soft
    # targets so we use the soft-CE / soft-NLL path uniformly.
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0) or 0.0)
    use_soft_targets = soft_labels or label_smoothing > 0.0
    if use_soft_targets:
        criterion = _soft_nll if returns_log_probs else _soft_cross_entropy
    else:
        criterion = nn.NLLLoss() if returns_log_probs else nn.CrossEntropyLoss()
    low_rank_cfg = dict(train_cfg.get("low_rank_regularization") or {})

    # Augmentation pipeline (image inputs only).
    aug_cfg = (
        (train_cfg.get("augmentation") or {})
        if isinstance(train_cfg.get("augmentation"), dict)
        else {}
    )
    train_transform = (
        build_train_transform(aug_cfg.get("kind"))
        if aug_cfg.get("enabled", False)
        else None
    )

    # Intra-mixture Mixup.
    mixup_cfg = (
        (train_cfg.get("mixup") or {}) if isinstance(train_cfg.get("mixup"), dict) else {}
    )
    mixup_alpha = float(mixup_cfg.get("alpha", 0.0)) if mixup_cfg.get("enabled", True) else 0.0

    # Early-stopping signal.
    es_metric = str(train_cfg.get("early_stopping_metric", "val_loss")).lower()
    if es_metric not in {"val_loss", "geometric_residual"}:
        raise ValueError(f"Unsupported early_stopping_metric: {es_metric}")
    if es_metric == "geometric_residual" and num_classes_K is None:
        # Caller didn't pass K -> fall back to val_loss rather than crashing.
        LOGGER.warning("geometric_residual ES requested without num_classes_K; falling back to val_loss")
        es_metric = "val_loss"

    bottleneck_head = getattr(model, "head", None)
    pi_warmup_epochs = int(train_cfg.get("bottleneck_warmup_epochs", 0))
    can_freeze_pi = (
        pi_warmup_epochs > 0
        and bottleneck_head is not None
        and hasattr(bottleneck_head, "set_pi_frozen")
    )
    if can_freeze_pi:
        bottleneck_head.set_pi_frozen(True)

    train_loader_for_count = _make_loader(
        train_x,
        labels=None if soft_labels else train_labels,
        soft_labels=train_labels if soft_labels else None,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    steps_per_epoch = max(1, len(train_loader_for_count))
    scheduler_cfg = train_cfg.get("scheduler") or {}
    scheduler_type = str(scheduler_cfg.get("type", "none")).lower()
    if scheduler_type == "cosine_warmup":
        scheduler = _build_cosine_warmup_scheduler(
            optimizer,
            total_steps=steps_per_epoch * int(train_cfg["epochs"]),
            warmup_frac=float(scheduler_cfg.get("warmup_frac", 0.05)),
            min_lr_ratio=float(scheduler_cfg.get("min_lr_ratio", 0.0)),
        )
    else:
        scheduler = None

    train_loader = _make_loader(
        train_x,
        labels=None if soft_labels else train_labels,
        soft_labels=train_labels if soft_labels else None,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    val_loader = _make_loader(
        val_x,
        labels=None if soft_labels else val_labels,
        soft_labels=val_labels if soft_labels else None,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    history_train: list[float] = []
    history_val: list[float] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    patience = int(train_cfg.get("early_stopping_patience", 5))
    no_improve = 0
    start_time = time.perf_counter()

    def _compute_loss(batch_x_in: torch.Tensor, batch_y_in) -> torch.Tensor:
        logits = model(batch_x_in)
        loss = criterion(logits, batch_y_in)
        if bottleneck_head is not None and hasattr(bottleneck_head, "regularization_loss"):
            reg = bottleneck_head.regularization_loss()
            if reg.requires_grad or bool(reg):
                loss = loss + reg
        loss = loss + low_rank_tail_nuclear_penalty(
            logits,
            low_rank_cfg,
            returns_log_probs=returns_log_probs,
        )
        return loss

    num_outputs = int(getattr(model, "num_outputs", 0)) or None

    for epoch in range(int(train_cfg["epochs"])):
        if can_freeze_pi and epoch == pi_warmup_epochs:
            bottleneck_head.set_pi_frozen(False)
        model.train()
        running_train = 0.0
        total_train = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if train_transform is not None:
                batch_x = train_transform(batch_x)
            if mixup_alpha > 0.0 and not soft_labels:
                batch_x, batch_y = intra_mixture_mixup(batch_x, batch_y, alpha=mixup_alpha)
            if use_soft_targets and not soft_labels:
                if num_outputs is None:
                    raise RuntimeError("model.num_outputs missing; cannot smooth labels")
                batch_y_use = smooth_one_hot(batch_y, num_outputs, label_smoothing)
            else:
                batch_y_use = batch_y

            if use_sam:
                # SAM requires two forward+backward passes. The second pass
                # uses the perturbed weights so the augmentation/mixup must
                # be re-applied or held constant -- we re-use the same x,y.
                optimizer.zero_grad(set_to_none=True)
                loss = _compute_loss(batch_x, batch_y_use)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                _compute_loss(batch_x, batch_y_use).backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = _compute_loss(batch_x, batch_y_use)
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running_train += float(loss.item()) * len(batch_x)
            total_train += len(batch_x)
        train_loss = running_train / max(total_train, 1)

        model.eval()
        running_val = 0.0
        total_val = 0
        val_probs_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                if use_soft_targets and not soft_labels:
                    if num_outputs is None:
                        raise RuntimeError("model.num_outputs missing; cannot smooth labels")
                    batch_y_use = smooth_one_hot(batch_y, num_outputs, label_smoothing)
                else:
                    batch_y_use = batch_y
                loss = criterion(logits, batch_y_use) + low_rank_tail_nuclear_penalty(
                    logits,
                    low_rank_cfg,
                    returns_log_probs=returns_log_probs,
                )
                running_val += float(loss.item()) * len(batch_x)
                total_val += len(batch_x)
                if es_metric == "geometric_residual":
                    probs = logits.exp() if returns_log_probs else torch.softmax(logits, dim=-1)
                    val_probs_chunks.append(probs.cpu().numpy())
        val_loss = running_val / max(total_val, 1)
        if es_metric == "geometric_residual":
            cloud = np.concatenate(val_probs_chunks, axis=0) if val_probs_chunks else np.zeros((0, 0))
            es_signal = geometric_subspace_residual(cloud, int(num_classes_K))
        else:
            es_signal = val_loss

        history_train.append(train_loss)
        history_val.append(val_loss)
        LOGGER.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f es_signal=%.6f",
            epoch + 1,
            train_loss,
            val_loss,
            float(es_signal),
        )

        if es_signal + 1e-6 < best_val_loss:
            best_val_loss = es_signal
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    runtime_seconds = time.perf_counter() - start_time
    return TrainHistory(
        train_loss=history_train,
        val_loss=history_val,
        best_epoch=best_epoch,
        runtime_seconds=runtime_seconds,
    )


@torch.no_grad()
def infer_classifier(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> InferenceOutputs:
    """Run forward inference and return logits, probs, and embeddings."""
    loader = _make_loader(
        x, labels=np.zeros(len(x), dtype=np.int64), batch_size=batch_size, shuffle=False
    )
    model.to(device)
    model.eval()
    returns_log_probs = bool(getattr(model, "returns_log_probs", False))
    logits_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []
    embeddings_list: list[np.ndarray] = []
    for batch_x, _ in loader:
        batch_x = batch_x.to(device)
        out = model(batch_x)
        if returns_log_probs:
            # `out` is log q(x); the "logits" slot stores log-probs so calibration
            # (which softmaxes them) leaves the simplex point invariant up to scale.
            probs = out.exp()
            logits = out
        else:
            logits = out
            probs = torch.softmax(logits, dim=-1)
        embeddings = model.extract_embeddings(batch_x)
        logits_list.append(logits.cpu().numpy())
        probs_list.append(probs.cpu().numpy())
        embeddings_list.append(embeddings.cpu().numpy())
    return InferenceOutputs(
        logits=np.concatenate(logits_list, axis=0),
        probabilities=np.concatenate(probs_list, axis=0),
        embeddings=np.concatenate(embeddings_list, axis=0),
    )
