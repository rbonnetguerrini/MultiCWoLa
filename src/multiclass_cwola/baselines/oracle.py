"""Oracle supervised baseline."""

from __future__ import annotations

import numpy as np
import torch

from multiclass_cwola.models.backbones import build_classifier
from multiclass_cwola.training.trainer import InferenceOutputs, infer_classifier, train_classifier


def run_oracle_supervised(
    model_cfg: dict,
    train_cfg: dict,
    input_shape: tuple[int, ...],
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    device: torch.device,
    checkpoint_path: str | None = None,
    reuse_checkpoint: bool = False,
) -> tuple[InferenceOutputs, dict]:
    """Train directly on hidden true labels as an oracle upper bound."""
    model = build_classifier(
        model_cfg, input_shape=input_shape, num_outputs=len(np.unique(train_y))
    )
    history = train_classifier(
        model,
        train_x=train_x,
        train_labels=train_y,
        val_x=val_x,
        val_labels=val_y,
        train_cfg=train_cfg,
        checkpoint_path=checkpoint_path,
        reuse_checkpoint=reuse_checkpoint,
        device=device,
    )
    outputs = infer_classifier(
        model, x=test_x, batch_size=int(train_cfg["batch_size"]), device=device
    )
    return outputs, history.__dict__
