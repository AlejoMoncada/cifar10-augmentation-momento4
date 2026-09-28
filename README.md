# CIFAR-10 controlled augmentation study

This Momento 4 notebook compares an unaugmented reference CNN against the identical Momento 3 CNN with in-model augmentation, single-transform ablations, and two deliberately questionable probes. Analysis and conclusion cells are prompts for students, not prewritten conclusions.

## Setup

Use Python 3.12. Install dependencies with:

```bash
uv sync
```

Place the **Python version** of CIFAR-10 at `data/cifar-10-batches-py/` so it contains `data_batch_1` through `data_batch_5`, `test_batch`, and `batches.meta`. The project does not download data at runtime. Dataset provenance and caveats are in [`docs/DATASET.md`](docs/DATASET.md).

## Run

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/cifar10_augmentation.ipynb
```

The notebook's first code cell controls `SCENARIOS`, `SEEDS`, `EPOCHS`, and `SUBSET_PER_CLASS`. The optional balanced subset applies only to training; validation and official test remain complete. The default is the complete training set, one seed, and up to 30 epochs. Full execution can take considerable time on CPU.

## Outputs

- `results/splits.npz`: original training-row indices for the fixed stratified train/validation split and separate official-test row indices.
- `results/dataset_summary.csv`: observed per-class split counts.
- `results/runs/<scenario>_seed<seed>/`: JSON history, test metrics, per-class precision/recall/F1, confusion matrix, test predictions and probabilities.
- `results/metrics.csv`, `results/run_metrics.csv`, `results/per_class_metrics.csv`, `results/generalization_gaps.csv`: aggregate and run-level tables.
- `results/figures/*.png`: 200-dpi EDA, augmentation, learning-curve, confusion, class-F1, and prediction-example figures.

Completed runs are reused when all three run artifacts exist. pHash hits are screening candidates, not confirmed duplicates. Augmentation does not replace representative data.
