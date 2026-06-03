"""Wei et al. (ICLR 2024) oracle-prior baselines.

Implements two members of the "Consistent Multi-Class Classification from Multiple
Unlabeled Datasets" baseline family for our same-mixture, same-backbone fairness
protocol:

  - CCM (Classifier-Consistent Method): train a K-class classifier f by minimising
    the cross-entropy of the *induced* mixture posterior, h_m(x) ~ Theta_m^T f(x),
    against the observed mixture identity. The forward correction Theta is the
    true class-prior matrix (M x K) supplied to the baseline at training time.

  - RCM (Risk-Consistent Method / unbiased risk estimator): re-weight per-class
    cross-entropy contributions of every example x in mixture m by the m-th
    column of the pseudo-inverse Theta+, giving an unbiased estimator of the
    supervised K-class risk (clipped at zero in practice).

Both methods receive the true class-prior matrix Theta = bundle.pi (M x K, rows
sum to 1). Our own method does NOT receive Theta -- it only sees mixture IDs.
The reporting layer must always label these baselines as oracle-prior.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from multiclass_cwola.data.common import ArrayDataset
from multiclass_cwola.models.backbones import build_classifier
from multiclass_cwola.training.trainer import infer_classifier

LOGGER = logging.getLogger(__name__)


def _validate_theta(theta: np.ndarray, num_classes: int | None = None) -> None:
    if not isinstance(theta, np.ndarray):
        raise TypeError(f"theta must be a numpy array, got {type(theta)!r}")
    if theta.ndim != 2:
        raise ValueError(f"theta must be 2D (M x K); got shape {theta.shape}")
    if theta.shape[0] < 1 or theta.shape[1] < 1:
        raise ValueError(f"theta must be non-empty; got shape {theta.shape}")
    if num_classes is not None and int(theta.shape[1]) != int(num_classes):
        raise ValueError(
            f"theta has K={theta.shape[1]} columns but num_classes={num_classes}"
        )
    if not np.isfinite(theta).all():
        raise ValueError("theta contains NaN/inf")
    if (theta < -1e-8).any():
        raise ValueError("theta contains negative entries")

    row_sums = theta.sum(axis=1)
    if (row_sums <= 0).any():
        raise ValueError("theta has an all-zero row (invalid mixture priors)")
    if not np.allclose(row_sums, 1.0, atol=5e-4, rtol=0.0):
        raise ValueError(
            "theta rows are expected to sum to 1 (P(class|mixture)); "
            f"row_sums in [{row_sums.min():.4f}, {row_sums.max():.4f}]"
        )


def _validate_mixture_labels(labels: np.ndarray, num_sources: int, name: str) -> None:
    if not isinstance(labels, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(labels)!r}")
    if labels.ndim != 1:
        raise ValueError(f"{name} must be 1D; got shape {labels.shape}")
    if len(labels) == 0:
        raise ValueError(f"{name} is empty")
    if labels.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be integer mixture IDs; got dtype {labels.dtype}")
    if labels.min() < 0 or labels.max() >= num_sources:
        raise ValueError(
            f"{name} must lie in [0, {num_sources}); got range [{int(labels.min())}, {int(labels.max())}]"
        )


@dataclass
class WeiTrainResult:
    val_probabilities: np.ndarray
    test_probabilities: np.ndarray
    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int
    runtime_seconds: float


def _make_loader(
    x: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = ArrayDataset(x=x, labels=labels.astype(np.int64))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _column_normalize(theta: np.ndarray) -> np.ndarray:
    """Return V (M x K) whose k-th column is Theta[:, k] / c_k with c_k = sum_m Theta[m, k].

    Under uniform mixture sampling, V[:, k] is the simplex vertex v_k from the
    paper (Bayes-optimal source posterior at a 'pure' class-k point), and
    p(m | x) = sum_k V[m, k] f_k(x) with f the Bayes-optimal class posterior.
    """
    c = theta.sum(axis=0, keepdims=True)
    c = np.where(c > 1e-12, c, 1.0)
    return (theta / c).astype(np.float32)


def _ccm_loss(logits: torch.Tensor, mixture: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    # logits: (B, K). vertices: (M, K), columns sum to 1. mixture: (B,) in [0, M).
    f = torch.softmax(logits, dim=-1)               # (B, K)
    v_rows = vertices[mixture]                      # (B, K) = (Theta_{m,k} / c_k)
    h = (v_rows * f).sum(dim=-1).clamp_min(1e-12)   # (B,) approximates p(m|x)
    return -torch.log(h).mean()


def _train_with_loss(
    *,
    model: nn.Module,
    train_x: np.ndarray,
    train_mixture: np.ndarray,
    val_x: np.ndarray,
    val_mixture: np.ndarray,
    train_cfg: dict,
    device: torch.device,
    loss_fn,
    checkpoint_path: str | Path | None = None,
    reuse_checkpoint: bool = False,
) -> tuple[nn.Module, list[float], list[float], int, float]:
    model.to(device)
    if reuse_checkpoint and checkpoint_path is not None and Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return model, [], [], -1, 0.0
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    train_loader = _make_loader(
        train_x, train_mixture, int(train_cfg["batch_size"]), True,
        int(train_cfg.get("num_workers", 0)),
    )
    val_loader = _make_loader(
        val_x, val_mixture, int(train_cfg["batch_size"]), False,
        int(train_cfg.get("num_workers", 0)),
    )
    history_train: list[float] = []
    history_val: list[float] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    patience = int(train_cfg.get("early_stopping_patience", 5))
    no_improve = 0
    start = time.perf_counter()

    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        running, total = 0.0, 0
        for batch_x, batch_m in train_loader:
            batch_x = batch_x.to(device)
            batch_m = batch_m.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_m)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(batch_x)
            total += len(batch_x)
        train_loss = running / max(total, 1)

        model.eval()
        running, total = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_m in val_loader:
                batch_x = batch_x.to(device)
                batch_m = batch_m.to(device)
                logits = model(batch_x)
                loss = loss_fn(logits, batch_m)
                running += float(loss.item()) * len(batch_x)
                total += len(batch_x)
        val_loss = running / max(total, 1)

        history_train.append(train_loss)
        history_val.append(val_loss)
        LOGGER.info("wei epoch=%d train=%.4f val=%.4f", epoch + 1, train_loss, val_loss)

        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
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
    return model, history_train, history_val, best_epoch, time.perf_counter() - start


def _build_kclass_model(model_cfg: dict, input_shape, num_classes: int) -> nn.Module:
    return build_classifier(model_cfg, input_shape=input_shape, num_outputs=num_classes)


def run_wei_ccm(
    *,
    model_cfg: dict,
    train_cfg: dict,
    input_shape,
    train_x: np.ndarray,
    train_mixture: np.ndarray,
    val_x: np.ndarray,
    val_mixture: np.ndarray,
    test_x: np.ndarray,
    theta: np.ndarray,
    num_classes: int,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
    reuse_checkpoint: bool = False,
) -> WeiTrainResult:
    """Wei CCM: forward-corrected NLL on mixture identity with known Theta."""
    _validate_theta(theta, num_classes=num_classes)
    _validate_mixture_labels(train_mixture, int(theta.shape[0]), "train_mixture")
    _validate_mixture_labels(val_mixture, int(theta.shape[0]), "val_mixture")
    vertices = _column_normalize(theta)
    theta_t = torch.as_tensor(vertices, dtype=torch.float32, device=device)

    def loss_fn(logits, mixture):
        return _ccm_loss(logits, mixture, theta_t)

    model = _build_kclass_model(model_cfg, input_shape, num_classes)
    model, train_loss, val_loss, best_epoch, rt = _train_with_loss(
        model=model,
        train_x=train_x,
        train_mixture=train_mixture,
        val_x=val_x,
        val_mixture=val_mixture,
        train_cfg=train_cfg,
        device=device,
        loss_fn=loss_fn,
        checkpoint_path=checkpoint_path,
        reuse_checkpoint=reuse_checkpoint,
    )
    val_out = infer_classifier(model, val_x, int(train_cfg["batch_size"]), device)
    test_out = infer_classifier(model, test_x, int(train_cfg["batch_size"]), device)
    return WeiTrainResult(
        val_probabilities=val_out.probabilities,
        test_probabilities=test_out.probabilities,
        train_loss=train_loss,
        val_loss=val_loss,
        best_epoch=best_epoch,
        runtime_seconds=rt,
    )


