"""Build compact report figures from the saved runs (no training)."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.augmentation import build_augmentation
from src.data import load_cifar10, make_splits, normalize

OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SCENARIOS = ("reference", "full", "flip_h", "rotation", "zoom", "flip_v", "blur")
LABELS = {"reference": "referencia", "full": "full (M3)", "flip_h": "flip_h", "rotation": "rotation",
          "zoom": "zoom", "flip_v": "flip_v", "blur": "blur"}
COLORS = {"reference": "#1f3864", "full": "#c0504d", "flip_h": "#4f81bd", "rotation": "#9bbb59",
          "zoom": "#8064a2", "flip_v": "#f79646", "blur": "#7f7f7f"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})


def augmentation_strip():
    train_raw, train_labels, test_raw, _, class_names = load_cifar10(ROOT / "data" / "cifar-10-batches-py")
    splits = make_splits(train_labels, ROOT / "results" / "splits.npz", test_count=len(test_raw))
    x = normalize(train_raw[splits["train"]])
    y = train_labels[splits["train"]]
    picks = [int(np.flatnonzero(y == c)[0]) for c in (0, 1, 3)]
    images = x[picks]
    fig, axes = plt.subplots(len(picks), len(SCENARIOS), figsize=(7.2, 3.3))
    for col, scenario in enumerate(SCENARIOS):
        out = build_augmentation(scenario, seed=42)(images, training=True).numpy()
        for row in range(len(picks)):
            ax = axes[row, col]
            ax.imshow(np.clip(out[row], 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(LABELS[scenario], fontsize=8)
            if col == 0:
                ax.set_ylabel(class_names[y[picks[row]]], fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "aug_strip.png", dpi=220)
    plt.close(fig)


def curves_and_bars():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.35, 1]})
    for s in SCENARIOS:
        h = json.loads((ROOT / "results" / "runs" / f"{s}_seed42" / "history.json").read_text())["history"]
        ep = np.arange(1, len(h["val_accuracy"]) + 1)
        a1.plot(ep, h["val_accuracy"], color=COLORS[s], lw=1.4 if s in ("reference", "full") else 1.0,
                label=LABELS[s])
    a1.set_xlabel("Época"); a1.set_ylabel("Exactitud de validación"); a1.set_ylim(0.1, 0.9)
    a1.set_title("Validación por época (todas las condiciones)")
    a1.grid(alpha=0.3); a1.legend(fontsize=6.5, ncol=2, loc="lower right")
    m = pd.read_csv(ROOT / "results" / "scenario_comparison.csv").set_index("scenario").loc[list(SCENARIOS)]
    bars = a2.barh([LABELS[s] for s in SCENARIOS], m["macro_f1"] * 100, color=[COLORS[s] for s in SCENARIOS])
    a2.axvline(m.loc["reference", "macro_f1"] * 100, color="black", ls="--", lw=0.8)
    for b, v in zip(bars, m["macro_f1"] * 100):
        a2.text(v - 1.5, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", ha="right", fontsize=7, color="white", fontweight="bold")
    a2.invert_yaxis(); a2.set_xlim(0, 100); a2.set_xlabel("F1 macro en prueba (%)")
    a2.set_title("F1 macro por condición")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "curves_bars.png", dpi=220)
    plt.close(fig)


def confusion_and_delta():
    names = pd.read_csv(ROOT / "results" / "per_class_reference_vs_full.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, s, title in ((axes[0], "reference", "Referencia"), (axes[1], "full", "full (M3)")):
        cm = np.array(json.loads((ROOT / "results" / "runs" / f"{s}_seed42" / "metrics.json").read_text())["confusion_matrix"], float)
        cm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f"{cm[i, j]:.2f}"[1:] if cm[i, j] < 1 else "1", ha="center", va="center",
                        fontsize=4.2, color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(names["class"], rotation=90, fontsize=5.5)
        ax.set_yticklabels(names["class"] if s == "reference" else [], fontsize=5.5)
        ax.set_title(f"{title} (normalizada)")
        ax.set_xlabel("Predicha", fontsize=6.5)
    axes[0].set_ylabel("Real", fontsize=6.5)
    d = names["delta_f1_pp"]
    axes[2].barh(names["class"], d, color=["#c0504d" if v < 0 else "#4f81bd" for v in d])
    axes[2].invert_yaxis(); axes[2].axvline(0, color="black", lw=0.6)
    axes[2].set_title("Δ F1: full − ref. (pp)")
    axes[2].tick_params(axis="y", labelsize=6)
    for i, v in enumerate(d):
        axes[2].text(v - 0.4, i, f"{v:.1f}", ha="right", va="center", fontsize=5.5)
    axes[2].set_xlim(-19, 1)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "confusion_delta.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    augmentation_strip()
    curves_and_bars()
    confusion_and_delta()
    print("figures written to", OUT)
