"""Source-labeled array datasets for unlabeled-class data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from multiclass_cwola.data.common import MixtureDatasetBundle, SplitData, source_given_class_from_pi


def _required_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing required array '{key}' in NPZ dataset.")
    return np.asarray(data[key])


def _optional_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data:
        return None
    return np.asarray(data[key])


def _split_from_npz(data: np.lib.npyio.NpzFile, split: str) -> SplitData:
    return SplitData(
        x=_required_array(data, f"{split}_x").astype(np.float32),
        source=_required_array(data, f"{split}_source").astype(np.int64),
        y=(
            None
            if f"{split}_y" not in data
            else np.asarray(data[f"{split}_y"], dtype=np.int64)
        ),
        true_posteriors=_optional_array(data, f"{split}_true_posteriors"),
        true_source_posteriors=_optional_array(data, f"{split}_true_source_posteriors"),
    )


def _infer_num_classes(config: dict[str, Any], data: np.lib.npyio.NpzFile) -> int:
    if "K" in config and config["K"] is not None:
        return int(config["K"])
    for key in ("source_given_class", "pi", "train_true_posteriors", "val_true_posteriors"):
        if key in data:
            array = np.asarray(data[key])
            if key == "source_given_class":
                return int(array.shape[0])
            return int(array.shape[1])
    raise ValueError("Array datasets without labels or priors must set data.K explicitly.")


def _infer_num_sources(config: dict[str, Any], train_source: np.ndarray) -> int:
    if "M" in config and config["M"] is not None:
        return int(config["M"])
    return int(np.max(train_source)) + 1


def build_array_dataset(config: dict[str, Any]) -> MixtureDatasetBundle:
    """Load train/val/test source-labeled splits from a NumPy NPZ file.

    Required arrays:
      - train_x, train_source
      - val_x, val_source
      - test_x, test_source

    Optional arrays:
      - train_y, val_y, test_y
      - train_true_posteriors, val_true_posteriors, test_true_posteriors
      - train_true_source_posteriors, val_true_source_posteriors, test_true_source_posteriors
      - pi
      - source_given_class
    """
    path = Path(str(config["path"]))
    if not path.exists():
        raise FileNotFoundError(f"Array dataset not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        train = _split_from_npz(data, "train")
        val = _split_from_npz(data, "val")
        test = _split_from_npz(data, "test")
        pi = _optional_array(data, "pi")
        source_given_class = _optional_array(data, "source_given_class")
        if source_given_class is None and pi is not None:
            source_given_class = source_given_class_from_pi(pi)
        num_classes = _infer_num_classes(config, data)

    num_sources = _infer_num_sources(config, train.source)
    task_type = str(config.get("task_type", "auto"))
    if task_type == "auto":
        task_type = "image" if train.x.ndim == 4 else "tabular"
    if bool(config.get("flatten", False)) and train.x.ndim > 2:
        train.x = train.x.reshape(len(train.x), -1)
        val.x = val.x.reshape(len(val.x), -1)
        test.x = test.x.reshape(len(test.x), -1)
        task_type = "tabular"

    return MixtureDatasetBundle(
        train=train,
        val=val,
        test=test,
        pi=pi,
        source_given_class=source_given_class,
        metadata={
            "dataset_name": str(config.get("dataset_name", path.stem)),
            "dataset_path": str(path),
            "dataset_split_strategy": "user_provided_npz",
            "has_true_labels": train.y is not None and val.y is not None and test.y is not None,
            "has_oracle_mixing": pi is not None and source_given_class is not None,
        },
        input_shape=tuple(train.x.shape[1:]),
        task_type=task_type,
        num_classes=num_classes,
        num_sources=num_sources,
    )
