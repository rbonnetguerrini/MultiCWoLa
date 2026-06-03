"""Shared dataclasses and dataset helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SplitData:
    """One split of data with optional labels and posteriors."""

    x: np.ndarray
    source: np.ndarray
    y: np.ndarray | None = None
    true_posteriors: np.ndarray | None = None
    true_source_posteriors: np.ndarray | None = None


@dataclass
class MixtureDatasetBundle:
    """Container returned by synthetic and real-data builders."""

    train: SplitData
    val: SplitData
    test: SplitData
    pi: np.ndarray | None = None
    source_given_class: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    input_shape: tuple[int, ...] = ()
    task_type: str = "tabular"
    num_classes: int = 0
    num_sources: int = 0


class ArrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch dataset backed by NumPy arrays."""

    def __init__(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        soft_labels: np.ndarray | None = None,
    ) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.labels = torch.as_tensor(labels)
        self.soft_labels = None
        if soft_labels is not None:
            self.soft_labels = torch.as_tensor(soft_labels, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.soft_labels is not None:
            return self.x[index], self.soft_labels[index]
        return self.x[index], self.labels[index].long()


def source_given_class_from_pi(pi: np.ndarray) -> np.ndarray:
    """Convert row-stochastic Pi[m, k] into source-given-class vertices V[k, m]."""
    col_sums = np.clip(pi.sum(axis=0, keepdims=True), 1e-8, None)
    return (pi / col_sums).T
