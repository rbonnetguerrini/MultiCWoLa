"""Reproducibility, logging, and lightweight persistence helpers."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(device: str = "auto") -> torch.device:
    """Choose a CPU or CUDA device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def setup_logging(level: int = logging.INFO) -> None:
    """Configure project logging once."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save JSON with stable formatting."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_text(path: str | Path, text: str) -> None:
    """Save a UTF-8 text file."""
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(text)


def environment_summary() -> dict[str, Any]:
    """Return a small runtime summary."""
    return {
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
