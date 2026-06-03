"""Post-hoc calibration for source posteriors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class TemperatureModule(nn.Module):
    """Scalar temperature scaling."""

    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.exp(self.log_temperature).clamp(min=1e-4)
        return logits / temperature


class VectorScalingModule(nn.Module):
    """Diagonal vector scaling plus bias."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(self.log_scale).clamp(min=1e-4)
        return logits * scale + self.bias


@dataclass
class CalibrationResult:
    """Fitted calibrator and calibration metadata."""

    module: nn.Module
    method: str
    nll_before: float
    nll_after: float


def _nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.CrossEntropyLoss()(logits, labels)


def fit_calibrator(
    logits: np.ndarray,
    labels: np.ndarray,
    method: str = "temperature",
    max_iter: int = 300,
    lr: float = 0.05,
) -> CalibrationResult:
    """Fit a temperature or vector scaling calibrator."""
    tensor_logits = torch.as_tensor(logits, dtype=torch.float32)
    tensor_labels = torch.as_tensor(labels, dtype=torch.long)
    if method == "temperature":
        module: nn.Module = TemperatureModule()
    elif method == "vector":
        module = VectorScalingModule(num_classes=logits.shape[1])
    else:
        raise ValueError(f"Unsupported calibration method: {method}")
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    before = float(_nll(tensor_logits, tensor_labels).item())
    for _ in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        loss = _nll(module(tensor_logits), tensor_labels)
        loss.backward()
        optimizer.step()
    after = float(_nll(module(tensor_logits), tensor_labels).item())
    return CalibrationResult(
        module=module.eval(), method=method, nll_before=before, nll_after=after
    )


@torch.no_grad()
def calibrate_logits(logits: np.ndarray, calibration: CalibrationResult) -> np.ndarray:
    """Apply a fitted calibrator and return probabilities."""
    tensor_logits = torch.as_tensor(logits, dtype=torch.float32)
    calibrated_logits = calibration.module(tensor_logits)
    return torch.softmax(calibrated_logits, dim=-1).cpu().numpy()
