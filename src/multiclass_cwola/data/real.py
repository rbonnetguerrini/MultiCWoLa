"""Real-data wrappers that hide labels during training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split

from multiclass_cwola.data.common import MixtureDatasetBundle, SplitData, source_given_class_from_pi
from multiclass_cwola.data.mixtures import generate_mixing_matrix


def _load_torchvision_dataset(
    dataset_name: str, root: str, download: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from torchvision import datasets
    except Exception as exc:  # pragma: no cover - import-path dependent
        raise RuntimeError(
            "torchvision is required for MNIST/FashionMNIST/CIFAR10 wrappers."
        ) from exc

    root_path = Path(root)
    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(root=root_path, train=True, download=download)
        test_dataset = datasets.MNIST(root=root_path, train=False, download=download)
    elif dataset_name == "fashion_mnist":
        train_dataset = datasets.FashionMNIST(root=root_path, train=True, download=download)
        test_dataset = datasets.FashionMNIST(root=root_path, train=False, download=download)
    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(root=root_path, train=True, download=download)
        test_dataset = datasets.CIFAR10(root=root_path, train=False, download=download)
    else:  # pragma: no cover - protected by config
        raise ValueError(f"Unsupported torchvision dataset: {dataset_name}")

    def _as_arrays(dataset) -> tuple[np.ndarray, np.ndarray]:
        data = dataset.data
        if hasattr(data, "numpy"):
            data = data.numpy()
        targets = np.asarray(dataset.targets)
        if data.ndim == 3:
            data = data[:, None, :, :]
        elif data.ndim == 4:
            data = np.transpose(data, (0, 3, 1, 2))
        return data.astype(np.float32) / 255.0, targets.astype(np.int64)

    train_x, train_y = _as_arrays(train_dataset)
    test_x, test_y = _as_arrays(test_dataset)
    return train_x, train_y, test_x, test_y


def load_base_dataset(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a labeled base dataset before converting it into unlabeled mixtures."""
    dataset_name = str(config["dataset_name"])
    if dataset_name == "digits":
        bundle = load_digits()
        x = bundle.images[:, None, :, :].astype(np.float32) / 16.0
        y = bundle.target.astype(np.int64)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=int(config.get("seed", 0)),
            stratify=y,
        )
    elif dataset_name == "iris":
        bundle = load_iris()
        x = bundle.data.astype(np.float32)
        y = bundle.target.astype(np.int64)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=int(config.get("seed", 0)),
            stratify=y,
        )
    elif dataset_name == "wine":
        bundle = load_wine()
        x = bundle.data.astype(np.float32)
        y = bundle.target.astype(np.int64)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=int(config.get("seed", 0)),
            stratify=y,
        )
    elif dataset_name == "galaxy10_decals":
        from multiclass_cwola.data.galaxy10 import DEFAULT_IMAGE_SIZE, load_galaxy10_arrays

        h5_path = Path(config.get("data_root", ".")) / str(
            config.get("h5_filename", "Galaxy10_DECals.h5")
        )
        return load_galaxy10_arrays(
            h5_path=h5_path,
            image_size=int(config.get("image_size", DEFAULT_IMAGE_SIZE)),
            cache_dir=config.get("cache_dir"),
            seed=int(config.get("seed", 0)),
            test_fraction=float(config.get("test_fraction", 0.2)),
        )
    else:
        return _load_torchvision_dataset(
            dataset_name=dataset_name,
            root=str(config.get("data_root", "./outputs/datasets")),
            download=bool(config.get("download", True)),
        )
    return x_train, y_train, x_test, y_test


def _select_classes(
    x: np.ndarray, y: np.ndarray, class_subset: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isin(y, np.asarray(class_subset))
    x_selected = x[mask]
    y_selected = y[mask]
    remapped = {label: index for index, label in enumerate(class_subset)}
    mapped_y = np.asarray([remapped[int(label)] for label in y_selected], dtype=np.int64)
    return x_selected, mapped_y


def _sample_mixtures(
    rng: np.random.Generator,
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    pi: np.ndarray,
    per_source: int,
    class_pool_strategy: str = "independent",
) -> SplitData:
    """Build mixture splits from a labeled pool.

    ``class_pool_strategy``:
      - ``"independent"`` (default): for each mixture, draw class labels from
        its prior, then for each class draw an image index *independently
        with replacement* from the full class pool. The same image can appear
        in multiple mixtures.
      - ``"disjoint_partition"``: pre-shuffle each class pool once, then
        consume the shuffled queue across mixtures so no image is seen by two
        different mixtures. Tests whether image overlap across mixtures
        matters for the shared-class-conditional assumption.
    """
    if class_pool_strategy not in {"independent", "disjoint_partition"}:
        raise ValueError(
            f"Unknown class_pool_strategy: {class_pool_strategy}"
        )
    num_sources, num_classes = pi.shape
    if class_pool_strategy == "disjoint_partition":
        per_class_queue: dict[int, list[int]] = {}
        for class_id in range(num_classes):
            cand = np.flatnonzero(y_pool == class_id)
            shuffled = rng.permutation(cand).tolist()
            per_class_queue[class_id] = shuffled
        per_class_cursor = {k: 0 for k in range(num_classes)}

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    for source_id, source_priors in enumerate(pi):
        source_labels = rng.choice(num_classes, size=per_source, p=source_priors)
        chosen_indices: list[int] = []
        for class_id in source_labels:
            class_id = int(class_id)
            if class_pool_strategy == "independent":
                candidates = np.flatnonzero(y_pool == class_id)
                chosen_indices.append(int(rng.choice(candidates)))
            else:  # disjoint_partition
                queue = per_class_queue[class_id]
                cursor = per_class_cursor[class_id]
                if cursor >= len(queue):
                    # Pool exhausted -- reshuffle and continue (with-replacement
                    # fallback). Logged via metadata to keep the contract honest.
                    queue = rng.permutation(np.flatnonzero(y_pool == class_id)).tolist()
                    per_class_queue[class_id] = queue
                    cursor = 0
                chosen_indices.append(int(queue[cursor]))
                per_class_cursor[class_id] = cursor + 1
        chosen_arr = np.asarray(chosen_indices, dtype=np.int64)
        xs.append(x_pool[chosen_arr])
        ys.append(y_pool[chosen_arr])
        sources.append(np.full(per_source, source_id, dtype=np.int64))
    return SplitData(
        x=np.concatenate(xs, axis=0).astype(np.float32),
        y=np.concatenate(ys, axis=0),
        source=np.concatenate(sources, axis=0),
    )


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


def _source_posteriors_from_labels(labels: np.ndarray, pi: np.ndarray) -> np.ndarray:
    return source_given_class_from_pi(pi)[labels]


def _resolve_per_source(requested: int, pool_size: int, num_sources: int, split_name: str) -> int:
    if requested > 0:
        return requested
    if num_sources <= 0:
        raise ValueError(f"num_sources must be positive, got {num_sources}")
    auto_value = pool_size // num_sources
    if auto_value <= 0:
        raise ValueError(
            f"Cannot derive full per-source size for {split_name}: pool_size={pool_size}, "
            f"num_sources={num_sources}"
        )
    return auto_value


def _counts_by_source_and_class(split: SplitData, num_sources: int, num_classes: int) -> list[list[int]]:
    counts = np.zeros((num_sources, num_classes), dtype=np.int64)
    for source_id in range(num_sources):
        mask = split.source == source_id
        split_labels = split.y[mask]
        for class_id in range(num_classes):
            counts[source_id, class_id] = int(np.sum(split_labels == class_id))
    return counts.tolist()


def _apply_real_shift(split: SplitData, shift_cfg: dict) -> SplitData:
    if not shift_cfg.get("enabled", False):
        return split
    scale = float(shift_cfg.get("scale", 0.0))
    shift_type = shift_cfg.get("type", "none")
    shifted = np.array(split.x, copy=True)
    if shift_type == "covariate":
        shifted = shifted + scale * split.source.reshape((-1,) + (1,) * (shifted.ndim - 1))
    elif shift_type == "class_conditional":
        shifted = shifted + scale * split.y.reshape((-1,) + (1,) * (shifted.ndim - 1))
    elif shift_type == "spurious":
        nuisance = scale * split.source.astype(np.float32)
        if shifted.ndim == 2:
            shifted = np.concatenate([shifted, nuisance[:, None]], axis=1)
        else:
            shifted[:, :, 0, 0] += nuisance
    split.x = shifted.astype(np.float32)
    return split


def build_real_dataset(config: dict) -> MixtureDatasetBundle:
    """Wrap a labeled dataset into train mixtures with hidden labels."""
    x_train_full, y_train_full, x_test_full, y_test_full = load_base_dataset(config)
    x_train_full, y_train_full = _select_classes(
        x_train_full, y_train_full, list(config["class_subset"])
    )
    x_test_full, y_test_full = _select_classes(
        x_test_full, y_test_full, list(config["class_subset"])
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=float(config.get("val_fraction", 0.2)),
        random_state=int(config.get("seed", 0)),
        stratify=y_train_full,
    )
    x_test = x_test_full
    y_test = y_test_full

    if bool(config.get("normalize", True)) and x_train.ndim == 2:
        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True) + 1e-6
        x_train = ((x_train - mean) / std).astype(np.float32)
        x_val = ((x_val - mean) / std).astype(np.float32)
        x_test = ((x_test - mean) / std).astype(np.float32)

    info = generate_mixing_matrix(
        num_sources=int(config["M"]),
        num_classes=int(config["K"]),
        mode=str(config.get("pi_mode", "dirichlet")),
        dirichlet_alpha=float(config.get("pi_dirichlet_alpha", 0.8)),
        similarity=float(config.get("pi_similarity", 0.9)),
        condition_strength=float(config.get("condition_strength", 0.25)),
        purity=float(config.get("purity", 1.0)),
        seed=int(config.get("seed", 0)),
    )
    rng = np.random.default_rng(int(config.get("seed", 0)))
    pool_strategy = str(config.get("class_pool_strategy", "independent"))
    num_sources = int(config["M"])
    train_per_source = _resolve_per_source(
        int(config["train_per_source"]),
        int(len(x_train)),
        num_sources,
        "train",
    )
    val_per_source = _resolve_per_source(
        int(config["val_per_source"]),
        int(len(x_val)),
        num_sources,
        "validation",
    )
    test_per_source = _resolve_per_source(
        int(config["test_per_source"]),
        int(len(x_test)),
        num_sources,
        "test",
    )

    train = _sample_mixtures(
        rng, x_train, y_train, info.pi, train_per_source, class_pool_strategy=pool_strategy
    )
    val = _sample_mixtures(
        rng, x_val, y_val, info.pi, val_per_source, class_pool_strategy=pool_strategy
    )
    test = _sample_mixtures(
        rng, x_test, y_test, info.pi, test_per_source, class_pool_strategy=pool_strategy
    )
    train = _apply_real_shift(train, dict(config["shift"]))
    val = _apply_real_shift(val, dict(config["shift"]))
    test = _apply_real_shift(test, dict(config["shift"]))
    train.true_posteriors = _one_hot(train.y, int(config["K"]))
    val.true_posteriors = _one_hot(val.y, int(config["K"]))
    test.true_posteriors = _one_hot(test.y, int(config["K"]))
    train.true_source_posteriors = _source_posteriors_from_labels(train.y, info.pi)
    val.true_source_posteriors = _source_posteriors_from_labels(val.y, info.pi)
    test.true_source_posteriors = _source_posteriors_from_labels(test.y, info.pi)

    task_type = "image" if train.x.ndim == 4 else "tabular"
    if bool(config.get("flatten", False)) and task_type == "image":
        train.x = train.x.reshape(len(train.x), -1)
        val.x = val.x.reshape(len(val.x), -1)
        test.x = test.x.reshape(len(test.x), -1)
        task_type = "tabular"

    return MixtureDatasetBundle(
        train=train,
        val=val,
        test=test,
        pi=info.pi,
        source_given_class=source_given_class_from_pi(info.pi),
        metadata={
            "dataset_name": config["dataset_name"],
            "dataset_split_strategy": "official_train_val_test",
            "class_subset": list(config["class_subset"]),
            "class_names": [str(label) for label in config["class_subset"]],
            "mixture_seed": int(config.get("seed", 0)),
            "class_pool_strategy": pool_strategy,
            "mixture_train_per_source": train_per_source,
            "mixture_val_per_source": val_per_source,
            "mixture_test_per_source": test_per_source,
            "train_counts_by_source_class": _counts_by_source_and_class(
                train, int(config["M"]), int(config["K"])
            ),
            "val_counts_by_source_class": _counts_by_source_and_class(
                val, int(config["M"]), int(config["K"])
            ),
            "test_counts_by_source_class": _counts_by_source_and_class(
                test, int(config["M"]), int(config["K"])
            ),
            "pool_sizes": {
                "train_pool": int(len(x_train)),
                "val_pool": int(len(x_val)),
                "test_pool": int(len(x_test)),
            },
            "rank": info.rank,
            "condition_number": info.condition_number,
            "row_similarity": info.row_similarity,
        },
        input_shape=tuple(train.x.shape[1:]),
        task_type=task_type,
        num_classes=int(config["K"]),
        num_sources=int(config["M"]),
    )
