"""Local CIFAR-10 loading, official split, stratified validation, and EDA data."""

from collections import defaultdict
from pathlib import Path
import pickle

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

CLASS_NAMES = np.array(
    ["Avión", "Automóvil", "Pájaro", "Gato", "Ciervo", "Perro", "Rana", "Caballo", "Barco", "Camión"]
)


def _read_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as file:
        batch = pickle.load(file, encoding="bytes")
    pixels = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    images = pixels.transpose(0, 2, 3, 1).copy()
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    if len(images) != len(labels):
        raise ValueError(f"Mismatched images and labels in {path}")
    return images, labels


def load_cifar10(directory: Path = Path("data/cifar-10-batches-py")) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load only the local Python-version batches; never trigger a download."""
    directory = Path(directory)
    required = [*(directory / f"data_batch_{i}" for i in range(1, 6)), directory / "test_batch", directory / "batches.meta"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing local CIFAR-10 files: " + ", ".join(missing))
    train_parts = [_read_batch(directory / f"data_batch_{i}") for i in range(1, 6)]
    train_images = np.concatenate([part[0] for part in train_parts])
    train_labels = np.concatenate([part[1] for part in train_parts])
    test_images, test_labels = _read_batch(directory / "test_batch")
    with (directory / "batches.meta").open("rb") as file:
        metadata = pickle.load(file, encoding="bytes")
    if len(metadata[b"label_names"]) != len(CLASS_NAMES):
        raise ValueError("Unexpected CIFAR-10 class metadata")
    return train_images, train_labels, test_images, test_labels, CLASS_NAMES.copy()


def make_splits(labels: np.ndarray, output_path: Path = Path("results/splits.npz"), test_count: int = 10000) -> dict[str, np.ndarray]:
    """Reference 80/20 split plus separate official-test row indices."""
    train, validation = train_test_split(
        np.arange(len(labels)), test_size=0.20, stratify=labels, random_state=42
    )
    splits = {"train": train, "validation": validation, "test": np.arange(test_count)}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **splits)
    return splits


def normalize(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32) / 255.0


def dataset_summary(train_labels: np.ndarray, test_labels: np.ndarray, splits: dict[str, np.ndarray], output_path: Path = Path("results/dataset_summary.csv")) -> pd.DataFrame:
    rows = []
    for class_id, name in enumerate(CLASS_NAMES):
        rows.append({
            "class_id": class_id,
            "class": name,
            "train": int(np.sum(train_labels[splits["train"]] == class_id)),
            "validation": int(np.sum(train_labels[splits["validation"]] == class_id)),
            "test": int(np.sum(test_labels == class_id)),
        })
    frame = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def image_statistics(images: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    pixels = images.astype(np.float32).reshape(len(images), -1) / 255.0
    return pd.DataFrame({
        "class_id": labels,
        "class": CLASS_NAMES[labels],
        "mean_brightness": pixels.mean(axis=1),
        "contrast_std": pixels.std(axis=1),
    })


def near_duplicate_train_test(train_images: np.ndarray, test_images: np.ndarray, max_hamming_distance: int = 4, max_examples: int = 8) -> dict:
    """Screen all official train/test pairs with pHash and an exact Hamming check.

    Five independent hash chunks ensure any 64-bit pair at distance <= 4
    shares at least one chunk (pigeonhole principle). Candidates are then
    checked with the complete hash. Matches are candidates, not proven duplicates.
    """
    if not 0 <= max_hamming_distance <= 4:
        raise ValueError("This five-chunk index supports thresholds from 0 to 4")
    train_hashes = [int(str(imagehash.phash(Image.fromarray(image))), 16) for image in train_images]
    test_hashes = [int(str(imagehash.phash(Image.fromarray(image))), 16) for image in test_images]
    widths = (13, 13, 13, 13, 12)
    shifts = (0, 13, 26, 39, 52)
    index: dict[tuple[int, int], list[int]] = defaultdict(list)
    for train_id, value in enumerate(train_hashes):
        for chunk_id, (width, shift) in enumerate(zip(widths, shifts)):
            index[(chunk_id, (value >> shift) & ((1 << width) - 1))].append(train_id)
    count = 0
    examples = []
    for test_id, value in enumerate(test_hashes):
        candidates = set()
        for chunk_id, (width, shift) in enumerate(zip(widths, shifts)):
            candidates.update(index[(chunk_id, (value >> shift) & ((1 << width) - 1))])
        for train_id in candidates:
            distance = (train_hashes[train_id] ^ value).bit_count()
            if distance <= max_hamming_distance:
                count += 1
                if len(examples) < max_examples:
                    examples.append((train_id, test_id, distance))
    return {"train_count": len(train_images), "test_count": len(test_images), "threshold": max_hamming_distance, "candidate_pair_count": count, "examples": examples}
