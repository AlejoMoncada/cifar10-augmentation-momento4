"""CIFAR-10 EDA and comparison figures, saved as 200-dpi PNG files."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .augmentation import SCENARIOS, build_augmentation
from .data import CLASS_NAMES


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_class_balance(summary: pd.DataFrame, path: Path) -> None:
    frame = summary.melt(id_vars=["class_id", "class"], value_vars=["train", "validation", "test"], var_name="split", value_name="count")
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(data=frame, x="class", y="count", hue="split", ax=ax)
    ax.set(xlabel="Clase", ylabel="Imágenes", title="Balance de clases por partición")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, path)


def plot_image_statistics(statistics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for ax, column, title in zip(axes, ("mean_brightness", "contrast_std"), ("Brillo medio por clase", "Contraste por clase")):
        sns.boxplot(data=statistics, x="class", y=column, color="#78a9bd", ax=ax)
        ax.set(title=title, xlabel="Clase", ylabel="Valor normalizado")
        ax.tick_params(axis="x", rotation=35)
    _save(fig, path)


def plot_class_samples(images: np.ndarray, labels: np.ndarray, path: Path, per_class: int = 5) -> None:
    fig, axes = plt.subplots(len(CLASS_NAMES), per_class, figsize=(2 * per_class, 1.65 * len(CLASS_NAMES)))
    for class_id, name in enumerate(CLASS_NAMES):
        indices = np.flatnonzero(labels == class_id)[:per_class]
        for column, ax in enumerate(axes[class_id]):
            if column < len(indices):
                ax.imshow(images[indices[column]])
            ax.axis("off")
            if column == 0:
                ax.set_ylabel(name, rotation=0, labelpad=38, va="center")
    fig.suptitle("Muestras de entrenamiento por clase")
    _save(fig, path)


def plot_near_duplicate_examples(train_images: np.ndarray, test_images: np.ndarray, examples: list[tuple[int, int, int]], path: Path) -> None:
    count = max(1, len(examples))
    fig, axes = plt.subplots(count, 2, figsize=(5, 2.2 * count), squeeze=False)
    for row, pair in enumerate(examples):
        train_id, test_id, distance = pair
        axes[row, 0].imshow(train_images[train_id])
        axes[row, 1].imshow(test_images[test_id])
        axes[row, 0].set_title(f"Train #{train_id} (d={distance})", fontsize=9)
        axes[row, 1].set_title(f"Test #{test_id}", fontsize=9)
    if not examples:
        axes[0, 0].text(0.5, 0.5, "Sin candidatos", ha="center", va="center")
        axes[0, 1].text(0.5, 0.5, "Sin candidatos", ha="center", va="center")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Pares candidatos a duplicados próximos (pHash)")
    _save(fig, path)


def plot_augmentations(train_images: np.ndarray, train_labels: np.ndarray, path: Path, scenarios: tuple[str, ...] = SCENARIOS, seed: int = 42) -> None:
    """Apply each scenario to the same six train-only images."""
    indices = [int(np.flatnonzero(train_labels == class_id)[0]) for class_id in range(6)]
    originals = train_images[indices]
    fig, axes = plt.subplots(len(scenarios), len(indices), figsize=(2 * len(indices), 1.8 * len(scenarios)), squeeze=False)
    for row, scenario in enumerate(scenarios):
        augmented = build_augmentation(scenario, seed=seed)(originals, training=True).numpy()
        for column, ax in enumerate(axes[row]):
            ax.imshow(np.clip(augmented[column], 0, 1))
            ax.axis("off")
            if row == 0:
                ax.set_title(CLASS_NAMES[train_labels[indices[column]]], fontsize=9)
            if column == 0:
                ax.set_ylabel(scenario, rotation=0, labelpad=55, va="center")
    fig.suptitle("Las mismas seis imágenes de entrenamiento bajo cada condición")
    _save(fig, path)


def _history(output_root: Path, scenario: str, seed: int) -> dict:
    path = Path(output_root) / "runs" / f"{scenario}_seed{seed}" / "history.json"
    return json.loads(path.read_text(encoding="utf-8"))["history"]


def plot_learning_curves(output_root: Path, seed: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for scenario, label in (("reference", "Referencia"), ("full", "Aumento completo")):
        history = _history(output_root, scenario, seed)
        epochs = np.arange(1, len(history["loss"]) + 1)
        axes[0].plot(epochs, history["loss"], label=f"{label} train")
        axes[0].plot(epochs, history["val_loss"], linestyle="--", label=f"{label} val")
        axes[1].plot(epochs, history["accuracy"], label=f"{label} train")
        axes[1].plot(epochs, history["val_accuracy"], linestyle="--", label=f"{label} val")
    for ax, title, ylabel in zip(axes, ("Pérdida", "Exactitud"), ("Pérdida", "Exactitud")):
        ax.set(title=title, xlabel="Época", ylabel=ylabel)
        ax.legend(fontsize=8)
    _save(fig, path)


def _predictions(output_root: Path, scenario: str, seed: int):
    return np.load(Path(output_root) / "runs" / f"{scenario}_seed{seed}" / "predictions.npz")


def plot_confusions(output_root: Path, seed: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for ax, scenario, title in zip(axes, ("reference", "full"), ("Referencia", "Aumento completo")):
        record = _predictions(output_root, scenario, seed)
        truth, prediction = record["truth"], record["prediction"]
        matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.float64)
        np.add.at(matrix, (truth, prediction), 1)
        matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set(title=title, xlabel="Predicción", ylabel="Clase real")
        ax.tick_params(axis="x", rotation=45)
    _save(fig, path)


def plot_per_class_f1(per_class: pd.DataFrame, path: Path) -> None:
    summary = per_class.groupby(["scenario", "class"], as_index=False)["f1"].mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.barplot(data=summary, x="class", y="f1", hue="scenario", hue_order=[scenario for scenario in SCENARIOS if scenario in summary.scenario.unique()], ax=ax)
    ax.set(title="F1 por clase y condición", xlabel="Clase", ylabel="F1", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=35)
    ax.legend(ncol=4, fontsize=8)
    _save(fig, path)


def plot_delta_f1(per_class: pd.DataFrame, path: Path) -> None:
    summary = per_class.groupby(["scenario", "class_id"], as_index=False)["f1"].mean().pivot(index="class_id", columns="scenario", values="f1")
    delta = summary["full"] - summary["reference"]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(CLASS_NAMES, delta, color=np.where(delta >= 0, "#478f7d", "#ad6565"))
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(title="Cambio F1 por clase: aumento completo − referencia", xlabel="Clase", ylabel="ΔF1")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, path)


def plot_prediction_examples(test_images: np.ndarray, output_root: Path, scenario: str, seed: int, path: Path) -> None:
    """One highest-confidence correct and mistake per true class, not first-N."""
    record = _predictions(output_root, scenario, seed)
    truth, prediction, probability = record["truth"], record["prediction"], record["probability"]
    confidence = probability.max(axis=1)
    fig, axes = plt.subplots(2, len(CLASS_NAMES), figsize=(2.1 * len(CLASS_NAMES), 5.4))
    for class_id in range(len(CLASS_NAMES)):
        for row, correct in enumerate((True, False)):
            candidates = np.flatnonzero((truth == class_id) & ((prediction == truth) == correct))
            ax = axes[row, class_id]
            if len(candidates):
                index = int(candidates[np.argmax(confidence[candidates])])
                ax.imshow(test_images[index])
                ax.set_title(f"{CLASS_NAMES[truth[index]]} → {CLASS_NAMES[prediction[index]]}\n{confidence[index]:.2f}", fontsize=8)
            else:
                ax.text(0.5, 0.5, "Sin casos", ha="center", va="center")
            ax.axis("off")
    axes[0, 0].set_ylabel("Correctos", rotation=0, labelpad=45)
    axes[1, 0].set_ylabel("Errores", rotation=0, labelpad=45)
    fig.suptitle(f"Predicciones de alta confianza por clase real: {scenario}")
    _save(fig, path)
