"""Galaxy10 DECaLS h5 loader.

Returns labeled `(x_train, y_train, x_test, y_test)` arrays in CHW float32
[0, 1]; mixture construction (Dirichlet Pi, label hiding, bundle assembly)
is handled by the existing path in `multiclass_cwola.data.real`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

GALAXY10_NUM_CLASSES = 10
DEFAULT_IMAGE_SIZE = 224


def _resize_batch(
    images_uint8: np.ndarray, size: int, chunk_size: int = 128
) -> np.ndarray:
    """Resize (N, H, W, 3) uint8 -> (N, 3, size, size) float32 in [0, 1].

    Streams in `chunk_size`-image chunks so peak memory stays bounded at
    ~chunk_size x 3 x max(H,size)^2 x 4 bytes rather than allocating the full
    float32 stack up-front.
    """
    import torch
    import torch.nn.functional as F

    n = images_uint8.shape[0]
    out = np.empty((n, 3, size, size), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_u8 = torch.from_numpy(images_uint8[start:end])  # uint8 view
        chunk = chunk_u8.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        if chunk.shape[-1] != size or chunk.shape[-2] != size:
            chunk = F.interpolate(chunk, size=(size, size), mode="bilinear", align_corners=False)
        out[start:end] = chunk.numpy()
    return out


def _cache_key(h5_path: Path, size: int) -> str:
    """Cache key keyed on (path, file size, image size) -- seed-independent."""
    stat = h5_path.stat()
    payload = f"{h5_path.resolve()}|{stat.st_size}|{size}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def load_galaxy10_arrays(
    h5_path: str | Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    cache_dir: str | Path | None = None,
    seed: int = 0,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Galaxy10 DECaLS, resize, and stratified-split into train/test.

    Returns float32 arrays in CHW [0, 1] and int64 labels in [0, 9].
    Caches the resized arrays to disk so repeated runs skip the 17k resizes.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"Galaxy10 h5 not found at {h5_path}")

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"galaxy10_pool_{_cache_key(h5_path, image_size)}.npz"

    if cache_path is not None and cache_path.exists():
        # mmap so concurrent jobs share the same on-disk file without each
        # pulling 10 GB into RAM.
        blob = np.load(cache_path, mmap_mode="r")
        x_pool = blob["x"]
        y_pool = np.asarray(blob["y"])
    else:
        import h5py

        with h5py.File(h5_path, "r") as f:
            images = np.asarray(f["images"][...], dtype=np.uint8)
            y_pool = np.asarray(f["ans"][...], dtype=np.int64)
        x_pool = _resize_batch(images, image_size)
        del images
        if cache_path is not None:
            # np.savez auto-appends .npz only if missing; use an explicit .npz tmp.
            tmp_path = cache_path.with_name(cache_path.name + ".tmp.npz")
            np.savez(tmp_path, x=x_pool, y=y_pool)
            tmp_path.replace(cache_path)

    rng = np.random.default_rng(seed)
    train_idx_list: list[np.ndarray] = []
    test_idx_list: list[np.ndarray] = []
    for k in range(GALAXY10_NUM_CLASSES):
        class_idx = np.flatnonzero(y_pool == k)
        rng.shuffle(class_idx)
        n_test = max(1, int(round(len(class_idx) * test_fraction)))
        test_idx_list.append(class_idx[:n_test])
        train_idx_list.append(class_idx[n_test:])
    train_idx = np.concatenate(train_idx_list)
    test_idx = np.concatenate(test_idx_list)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    # Materialize only the per-split slices into RAM (mmap -> in-memory copy).
    x_train = np.ascontiguousarray(x_pool[train_idx])
    y_train = y_pool[train_idx].astype(np.int64, copy=False)
    x_test = np.ascontiguousarray(x_pool[test_idx])
    y_test = y_pool[test_idx].astype(np.int64, copy=False)
    return x_train, y_train, x_test, y_test
