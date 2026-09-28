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
- [x] T2 Data + EDA: loader from local batches, class balance, resolution, pixel stats per class, near-duplicate check train↔test (perceptual hash on a sample or full), sample grid, `results/dataset_summary.csv`, `docs/DATASET.md`. Route: delegated (Codex).
- [x] T3 Protocol: reproduce Momento 3 split and config; save split indices. Route: delegated (Codex).
- [x] T4 Augmentation: reference (none), full (flip+rot+zoom as Momento 3), ablations (each alone), and deliberately label-questionable/harmful probes (vertical flip, Gaussian blur). Example grid. Route: delegated (Codex).
- [x] T5 Training: all scenarios, configurable seeds/epochs; histories + metrics saved. Route: delegated (Codex).
- [x] T6 Evaluation: test (no aug) accuracy, macro-F1, per-class P/R/F1, side-by-side confusion matrices, learning curves, generalization gap table, correct/incorrect examples with confidence. Route: delegated (Codex).
- [x] T7 Student placeholders + README. Route: delegated (Codex).

## Progress / evidence
- The earlier LFW scaffold was retargeted. `notebooks/lfw_augmentation.ipynb` was removed; `notebooks/cifar10_augmentation.ipynb` and `docs/DATASET.md` were created.
- The earlier parent-reported `uv sync` succeeded. This implementation did not run `uv` (sandbox restriction). `.venv/bin/python` reported Python `3.12.11` and TensorFlow `2.21.0`.
- `src/model.py` reproduces the Momento 3 CNN; `.venv/bin/python -c "from src.model import build_cnn_model; from src.augmentation import build_augmentation; m=build_cnn_model(build_augmentation('full')); print(m.count_params())"` printed `290090` (exit 0).
- `.venv/bin/python -m py_compile src/*.py` exited 0. `pyproject.toml` and the single root package entry in `uv.lock` now use `cifar10-augmentation`; both parse as TOML and the package names match. No `uv` command was run.
- Train-only assertions passed on random float32 inputs for all seven scenarios. Every scenario's model has 290,090 parameters. An identity layer is used for `reference` because Keras cannot call an empty `Sequential` inside a Functional model; it has no stochastic or trainable behavior.
- Synthetic local-pickle fixture (not CIFAR-10) verified uint8 `(N,32,32,3)` loader, stratified 80/20 split, split CSV, full-set pHash candidate screening, and EDA/augmentation figure generation. A separate one-epoch synthetic reference/full run wrote run JSON/NPZ, aggregate metrics, and five comparison figures; epoch times were 1.225 s and 1.295 s. Temporary synthetic artifacts were removed.
- Resumability now writes `run_config.json` only after complete artifacts and fingerprints the ordered pixels and labels of train, validation, and test. A matching synthetic run skipped; changed epochs, subset-per-class, and data fingerprint each raised a clear configuration mismatch; deleting one artifact caused a successful rerun. Synthetic test exit 0; final `py_compile` exit 0. Completed legacy runs without a manifest are rejected rather than silently reused.
- All seven notebook code cells compiled. The unpacked CIFAR-10 data became available later; no download or extraction was attempted. Local loader observed `(50000,32,32,3)` and `(10000,32,32,3)` `uint8` arrays. `results/splits.npz` has 40,000 train, 10,000 validation, and 10,000 separate official-test indices. `results/dataset_summary.csv` has 4,000/1,000/1,000 images per class. `docs/DATASET.md` now contains these observed counts.
- Full-set pHash screening compared all 50,000 official training and 10,000 test images at Hamming distance ≤ 4, finding 117 **candidate pairs** (not confirmed duplicates). Five real-data EDA/augmentation PNGs were written under `results/figures/`: `class_balance.png`, `class_samples.png`, `brightness_contrast.png`, `augmentation_grid.png`, and `near_duplicate_candidates.png`.
- Required real-data smoke ran under isolated `results/smoke/` with exactly 100 training images per class, full 10,000-image validation and test partitions, one epoch, and scenarios `reference` and `full`. Both completed and wrote `history.json`, `metrics.json`, `predictions.npz`, `run_config.json`, four aggregate CSVs, and six comparison PNGs. Observed epoch times: reference `10.641463750012917` s; full `10.951815457985504` s. Test accuracy / macro-F1: reference `0.1000` / `0.01864875864875865`; full `0.1017` / `0.02148725995322227`. These are smoke-only metrics, not final findings.
- T5 and T6 remain unchecked because the full seven-scenario, default 30-epoch experiment and complete notebook were intentionally not run. The real-data smoke proves the pipeline but is not a substitute for the full study.

- Full run (scripts/run_all.py, 40k/10k/10k, seed 42, 30 epochs): test accuracy reference 0.8375, full 0.7406, flip_h 0.8389, rotation 0.7706, zoom 0.8070, flip_v 0.7606, blur 0.5339 (early stop at epoch 8). Manifests backfilled for runs trained before the manifest feature.
- Deliverable notebook executed in place with NB_REUSE_ONLY=1: 17/17 code cells, 0 errors, 12 figures. Fixed hidden row labels in augmentation grid.
- Pending (students): analysis/conclusion cells and PDF report.
