# Feature: CIFAR-10 controlled data augmentation study (Momento 4)

## Objective
Reproducible notebook comparing a reference CNN (no augmentation) vs the same CNN trained with label-preserving augmentation applied only to training data, on CIFAR-10. Deliverable: PDF report <= 5 pages + notebook/repo. Due 2026-09-27 23:59.

## Why CIFAR-10 (scope change from LFW)
Teammates built Momento 3 (`Act3_ClasificacionVisualCNN.ipynb`) on CIFAR-10. Momento 4 must compare against a reference scenario with constant architecture, so it reuses their exact CNN and protocol for group coherence.

## Reference protocol (copied from teammates' Momento 3 notebook)
- Data: CIFAR-10 python version, loaded from local `data/cifar-10-batches-py` (same data as `tf.keras.datasets.cifar10.load_data()`).
- Split: `train_test_split(X_train_full, test_size=0.20, stratify=y, random_state=42)` → 40k train / 10k val; official 10k test.
- Normalization: `/255.0` float32.
- Model: `build_cnn_model` from Momento 3 (3 conv blocks with BN + dropout 0.25/0.3/0.4, Dense 128 + BN + dropout 0.5, softmax), 290,090 params.
- Training: Adam lr 1e-3, sparse categorical cross-entropy, batch 128, max 30 epochs, EarlyStopping(val_loss, patience=6, restore_best_weights), ReduceLROnPlateau(factor 0.5, patience 2, min_lr 1e-6), SEED 42.
- IMPORTANT: the Momento 3 model already embeds augmentation (RandomFlip horizontal, RandomRotation 0.1, RandomZoom 0.1). For Momento 4 the reference scenario removes that block; the augmented scenario restores it.
- Mixed precision FP16 was used on Colab GPU; disabled here (CPU). Same setting in every scenario.

## Constraints
- Augmentation only on train (Keras preprocessing layers are inactive at inference; verify explicitly).
- Everything except augmentation identical across scenarios.
- Notebook provides code, tables and figures; analysis/conclusion cells are placeholders for the students (course policy).
- TDD: not configured; functional check = notebook executes end-to-end and writes artifacts.

## Tasks
- [x] T1 Env: uv + Python 3.12 + TF 2.21 (`.venv` already synced). Route: inline (done by parent, re-checked with `.venv/bin/python`).
- [ ] T2 Data + EDA: loader from local batches, class balance, resolution, pixel stats per class, near-duplicate check train↔test (perceptual hash on a sample or full), sample grid, `results/dataset_summary.csv`, `docs/DATASET.md`. Route: delegated (Codex).
- [ ] T3 Protocol: reproduce Momento 3 split and config; save split indices. Route: delegated (Codex). Implementation and synthetic check complete; real `results/splits.npz` pending local data.
- [ ] T4 Augmentation: reference (none), full (flip+rot+zoom as Momento 3), ablations (each alone), and deliberately label-questionable/harmful probes (vertical flip, Gaussian blur). Example grid. Route: delegated (Codex). All seven scenarios and synthetic grid checked; real CIFAR-10 figure pending local data.
- [ ] T5 Training: all scenarios, configurable seeds/epochs; histories + metrics saved. Route: delegated (Codex).
- [ ] T6 Evaluation: test (no aug) accuracy, macro-F1, per-class P/R/F1, side-by-side confusion matrices, learning curves, generalization gap table, correct/incorrect examples with confidence. Route: delegated (Codex).
- [x] T7 Student placeholders + README. Route: delegated (Codex).

## Progress / evidence
- The earlier LFW scaffold was retargeted. `notebooks/lfw_augmentation.ipynb` was removed; `notebooks/cifar10_augmentation.ipynb` and `docs/DATASET.md` were created.
- The earlier parent-reported `uv sync` succeeded. This implementation did not run `uv` (sandbox restriction). `.venv/bin/python` reported Python `3.12.11` and TensorFlow `2.21.0`.
- `src/model.py` reproduces the Momento 3 CNN; `.venv/bin/python -c "from src.model import build_cnn_model; from src.augmentation import build_augmentation; m=build_cnn_model(build_augmentation('full')); print(m.count_params())"` printed `290090` (exit 0).
- `.venv/bin/python -m py_compile src/*.py` exited 0. `pyproject.toml` and the single root package entry in `uv.lock` now use `cifar10-augmentation`; both parse as TOML and the package names match. No `uv` command was run.
- Train-only assertions passed on random float32 inputs for all seven scenarios. Every scenario's model has 290,090 parameters. An identity layer is used for `reference` because Keras cannot call an empty `Sequential` inside a Functional model; it has no stochastic or trainable behavior.
- Synthetic local-pickle fixture (not CIFAR-10) verified uint8 `(N,32,32,3)` loader, stratified 80/20 split, split CSV, full-set pHash candidate screening, and EDA/augmentation figure generation. A separate one-epoch synthetic reference/full run wrote run JSON/NPZ, aggregate metrics, and five comparison figures; epoch times were 1.225 s and 1.295 s. Temporary synthetic artifacts were removed.
- All seven notebook code cells compiled. The actual CIFAR-10 notebook and requested one-epoch, 100-per-class reference/full smoke were not executed because `data/cifar-10-batches-py/test_batch` is absent; only `data/cifar-10-python.tar.gz` is present. No download or extraction was attempted. Therefore T2, T5, and T6 remain unchecked pending the real-data smoke run and observed EDA/metrics/figures; `docs/DATASET.md` retains a clearly marked split-table TODO.
