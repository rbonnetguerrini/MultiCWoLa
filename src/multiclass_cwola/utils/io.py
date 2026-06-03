"""Small I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf


def save_dataframe(path: str | Path, frame: pd.DataFrame) -> None:
    """Persist a dataframe to CSV without an index."""
    frame.to_csv(path, index=False)


def save_config(path: str | Path, config: Any) -> None:
    """Save a Hydra or OmegaConf config."""
    Path(path).write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")
