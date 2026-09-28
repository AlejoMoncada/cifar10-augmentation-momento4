"""Resumable CIFAR-10 scenario training and untouched-test evaluation."""

import gc
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .augmentation import SCENARIOS, build_augmentation
from .data import CLASS_NAMES
from .model import build_cnn_model


class EpochTimer(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.seconds = []

    def on_epoch_begin(self, epoch, logs=None):
        self.started = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.seconds.append(time.perf_counter() - self.started)


def _dataset(images: np.ndarray, labels: np.ndarray, batch_size: int, seed: int | None = None) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if seed is not None:
        dataset = dataset.shuffle(10000, seed=seed)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def balanced_subset_indices(labels: np.ndarray, per_class: int, seed: int = 42) -> np.ndarray:
    """Choose the same reproducible balanced training subset for all scenarios."""
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    rng = np.random.default_rng(seed)
    pieces = []
    for class_id in range(len(CLASS_NAMES)):
        available = np.flatnonzero(labels == class_id)
        if len(available) < per_class:
            raise ValueError(f"Class {class_id} has only {len(available)} training rows")
        pieces.append(rng.choice(available, size=per_class, replace=False))
    return np.sort(np.concatenate(pieces))


def train_scenario(
    scenario: str,
    seed: int,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    output_root: Path = Path("results"),
    epochs: int = 30,
    batch_size: int = 128,
) -> dict:
    """Train once with the exact Momento 3 optimizer, callbacks, and data pipeline."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    run_dir = Path(output_root) / "runs" / f"{scenario}_seed{seed}"
    history_path, metrics_path, predictions_path = (run_dir / name for name in ("history.json", "metrics.json", "predictions.npz"))
    if all(path.is_file() for path in (history_path, metrics_path, predictions_path)):
        print(f"Skipping completed run: {run_dir}")
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)
    tf.keras.backend.clear_session()
    augmentation = build_augmentation(scenario, seed=seed)
    model = build_cnn_model(augmentation)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    timer = EpochTimer()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        timer,
    ]
    train_ds = _dataset(train_images, train_labels, batch_size, seed=seed)
    val_ds = _dataset(val_images, val_labels, batch_size)
    test_ds = _dataset(test_images, test_labels, batch_size)
    started = time.perf_counter()
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks, verbose=1).history
    total_seconds = time.perf_counter() - started
    probability = model.predict(test_ds, verbose=0).astype(np.float32)
    prediction = probability.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, prediction, labels=np.arange(len(CLASS_NAMES)), zero_division=0
    )
    per_class = [
        {"class_id": int(i), "class": str(CLASS_NAMES[i]), "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
        for i in range(len(CLASS_NAMES))
    ]
    best_epoch_index = int(np.argmin(history["val_loss"]))
    metrics = {
        "scenario": scenario,
        "seed": int(seed),
        "accuracy": float(accuracy_score(test_labels, prediction)),
        "macro_f1": float(f1_score(test_labels, prediction, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(test_labels, prediction, labels=np.arange(len(CLASS_NAMES))).tolist(),
        "best_epoch": best_epoch_index + 1,
        "best_epoch_train_accuracy": float(history["accuracy"][best_epoch_index]),
        "best_epoch_val_accuracy": float(history["val_accuracy"][best_epoch_index]),
        "generalization_gap": float(history["accuracy"][best_epoch_index] - history["val_accuracy"][best_epoch_index]),
        "final_train_loss": float(history["loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "epochs_trained": len(history["loss"]),
        "epoch_seconds": [float(value) for value in timer.seconds],
        "fit_seconds": float(total_seconds),
    }
    _json(history_path, {"history": {key: [float(value) for value in values] for key, values in history.items()}, "epoch_seconds": metrics["epoch_seconds"]})
    np.savez_compressed(predictions_path, prediction=prediction, probability=probability, truth=test_labels)
    _json(metrics_path, metrics)
    print(f"{scenario} seed={seed}: accuracy={metrics['accuracy']:.4f}, epochs={metrics['epochs_trained']}, epoch_seconds={metrics['epoch_seconds']}")
    del model
    gc.collect()
    return metrics


def run_experiments(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    scenarios: tuple[str, ...] = SCENARIOS,
    seeds: tuple[int, ...] = (42,),
    epochs: int = 30,
    subset_per_class: int | None = None,
    output_root: Path = Path("results"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run or resume selected scenarios; always keep full validation and test sets."""
    if subset_per_class is not None:
        selected = balanced_subset_indices(train_labels, subset_per_class)
        train_images, train_labels = train_images[selected], train_labels[selected]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records = [
        train_scenario(scenario, seed, train_images, train_labels, val_images, val_labels, test_images, test_labels, output_root, epochs)
        for scenario in scenarios for seed in seeds
    ]
    rows = [{key: record[key] for key in ("scenario", "seed", "accuracy", "macro_f1", "epochs_trained", "fit_seconds")} for record in records]
    run_metrics = pd.DataFrame(rows)
    run_metrics.to_csv(output_root / "run_metrics.csv", index=False)
    summary = run_metrics.groupby("scenario", sort=False)[["accuracy", "macro_f1"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.fillna(0.0).reset_index()
    summary.to_csv(output_root / "metrics.csv", index=False)
    gaps = pd.DataFrame([{key: record[key] for key in ("scenario", "seed", "best_epoch", "best_epoch_train_accuracy", "best_epoch_val_accuracy", "generalization_gap", "final_train_loss", "final_val_loss", "epochs_trained")} for record in records])
    gaps.to_csv(output_root / "generalization_gaps.csv", index=False)
    per_class_rows = [{"scenario": record["scenario"], "seed": record["seed"], **item} for record in records for item in record["per_class"]]
    per_class_frame = pd.DataFrame(per_class_rows)
    per_class_frame.to_csv(output_root / "per_class_metrics.csv", index=False)
    return summary, gaps, per_class_frame
