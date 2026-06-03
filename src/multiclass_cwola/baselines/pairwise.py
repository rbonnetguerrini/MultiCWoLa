"""Simple one-vs-rest source-reduction baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

from multiclass_cwola.models.backbones import build_classifier
from multiclass_cwola.training.trainer import infer_classifier, train_classifier
from multiclass_cwola.utils.repro import seed_everything


def run_source_ovr_baseline(
    model_cfg: dict,
    train_cfg: dict,
    input_shape: tuple[int, ...],
    train_x: np.ndarray,
    train_source: np.ndarray,
    val_x: np.ndarray,
    val_source: np.ndarray,
    test_x: np.ndarray,
    num_sources: int,
    num_classes: int,
    device: torch.device,
    seed: int,
    checkpoint_dir: str | Path | None = None,
    reuse_checkpoints: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Train source one-vs-rest models, then cluster their outputs."""
    seed_everything(seed)
    logits_train: list[np.ndarray] = []
    logits_val: list[np.ndarray] = []
    logits_test: list[np.ndarray] = []
    for source_id in range(num_sources):
        binary_train = (train_source == source_id).astype(np.int64)
        binary_val = (val_source == source_id).astype(np.int64)
        model = build_classifier(model_cfg, input_shape=input_shape, num_outputs=2)
        checkpoint_path = None
        if checkpoint_dir is not None:
            checkpoint_path = Path(checkpoint_dir) / f"ovr_source_{source_id}.pt"
        train_classifier(
            model,
            train_x=train_x,
            train_labels=binary_train,
            val_x=val_x,
            val_labels=binary_val,
            train_cfg=train_cfg,
            checkpoint_path=checkpoint_path,
            reuse_checkpoint=reuse_checkpoints,
            device=device,
        )
        train_outputs = infer_classifier(
            model, x=train_x, batch_size=int(train_cfg["batch_size"]), device=device
        )
        val_outputs = infer_classifier(
            model, x=val_x, batch_size=int(train_cfg["batch_size"]), device=device
        )
        test_outputs = infer_classifier(
            model, x=test_x, batch_size=int(train_cfg["batch_size"]), device=device
        )
        logits_train.append(train_outputs.probabilities[:, 1:2])
        logits_val.append(val_outputs.probabilities[:, 1:2])
        logits_test.append(test_outputs.probabilities[:, 1:2])
    train_features = np.concatenate(logits_train, axis=1)
    val_features = np.concatenate(logits_val, axis=1)
    test_features = np.concatenate(logits_test, axis=1)
    kmeans = KMeans(n_clusters=num_classes, random_state=seed, n_init=10)
    kmeans.fit(train_features)
    val_distances = np.linalg.norm(
        val_features[:, None, :] - kmeans.cluster_centers_[None, :, :], axis=-1
    )
    distances = np.linalg.norm(
        test_features[:, None, :] - kmeans.cluster_centers_[None, :, :], axis=-1
    )
    val_soft = 1.0 / np.clip(val_distances, 1e-6, None)
    soft = 1.0 / np.clip(distances, 1e-6, None)
    val_soft /= val_soft.sum(axis=1, keepdims=True)
    soft /= soft.sum(axis=1, keepdims=True)
    return val_soft, soft
