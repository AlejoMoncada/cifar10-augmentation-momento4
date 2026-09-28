"""Run every scenario outside Jupyter; the notebook later reuses the saved runs."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

from src.data import load_cifar10, make_splits, normalize
from src.experiment import run_experiments

SCENARIOS = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else (
    "reference", "full", "flip_h", "rotation", "zoom", "flip_v", "blur")

train_raw, train_labels, test_raw, test_labels, _ = load_cifar10(ROOT / "data" / "cifar-10-batches-py")
splits = make_splits(train_labels, ROOT / "results" / "splits.npz", test_count=len(test_raw))
tr, va = splits["train"], splits["validation"]
summary, gaps, _ = run_experiments(
    normalize(train_raw[tr]), train_labels[tr], normalize(train_raw[va]), train_labels[va],
    normalize(test_raw), test_labels, scenarios=SCENARIOS, seeds=(42,), epochs=30,
    output_root=ROOT / "results",
)
print(summary.to_string())
print(gaps.to_string())
