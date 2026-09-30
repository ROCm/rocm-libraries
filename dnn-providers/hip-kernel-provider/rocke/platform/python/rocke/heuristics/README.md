# rocke Heuristics: ML-Based Kernel Selection

Fast, accurate kernel selection for `rocke` operations using LightGBM
regression with Origami-augmented feature engineering.

## What This Does

Instead of running all 4608+ kernel configurations on the GPU to find the best
one (exhaustive search taking ~46 seconds per shape), this system trains an ML
model that predicts TFLOPS for any (problem, kernel) pair in microseconds. It
scores all candidates instantly and picks the best kernel -- achieving 98.28%
of oracle-best TFLOPS efficiency across 108 tested shapes.

## Overview (read first)

This is the **training + prediction pipeline** for the LightGBM kernel-selection
models used with `rocke` -- a self-contained toolkit to **train, retrain,
evaluate, predict, and search** heuristics over the kernel candidate space.

- **Trained models are NOT stored in this folder** (it is code-only). Train your
  own into any `--out_dir` (see Quick Start), or point the tools at a model
  directory you already have.
- **Model directory layout** (what training emits and prediction reads): each
  model dir holds `model_tflops.lgbm` (LightGBM regressor), `feature_spec.json`
  (the exact feature layout), and `cv_metrics_*.json` / `feature_importances_*.json`.
- **Consumers:** any runtime that loads a LightGBM booster and extracts the
  feature layout described in `feature_spec.json` can use these models -- the
  feature layout is the contract. Models are arch- and op-specific.
- **Run scripts from this directory** (flat intra-package imports); it is a
  standalone tool dir, not imported as a deep `rocke.heuristics` sub-package.

## Quick Start

### 1. Generate benchmark data with the rocke sweep

Generate training data natively via `rocke` -- no CK Tile / CMake / ninja
build. The generator enumerates a shape corpus and the validity-filtered
`UniversalGemmSpec` variant grid, builds every variant on the fly
(`build_all_instances`), runs each `(variant, shape)` pair (`sweep_run`), and
writes the canonical training parquet directly:

```bash
python3 -m rocke.heuristics.gen_gemm_sweep_data \
    --out data/gemm_universal_gfx950.parquet \
    --cache-dir /tmp/rocke_sweep_cache \
    --arch gfx950 \
    --shape-set wide
```

No intermediate JSON or log files and no separate conversion step -- the parquet
is already in the canonical schema `train.py` consumes. See
[DATA_GENERATION.md](DATA_GENERATION.md) for all flags, shape corpora, and the
schema.

### 2. Train a model

```bash
python3 train.py \
    --data_dir data/ \
    --out_dir models/gemm_universal_fp8_gfx950 \
    --op gemm_universal --dtype fp8 --arch gfx950
```

**Note**: Trained models are automatically compressed to `.lgbm.gz` format to save space (~67% reduction). The Python tools automatically decompress them on first use and cache the decompressed version. For warm-start training, decompression happens automatically.

### 3. Evaluate

```bash
python3 evaluate.py \
    --model_dir models/gemm_universal_fp8_gfx950 \
    --data_dir data/ --op gemm_universal --dtype fp8
```

### 4. Predict the best kernel for a problem

```bash
python3 predict.py \
    --model_dir models/gemm_universal_fp8_gfx950 \
    --m 128 --n 1536 --k 7168 --layout rcr
```

### 5. Search for optimal configs (optional)

```bash
python3 search.py \
    --model_dir models/gemm_universal_fp8_gfx950 \
    --m 128 --n 1536 --k 7168 \
    --strategy random --budget 500 --top_k 10
```

### 6. Using a trained model from C++

C++ consumers load the model through the LightGBM C API, which requires an
**uncompressed** `.lgbm` file (decompress `.lgbm.gz` with `gunzip`; the Python
tools auto-decompress on first use). The consumer reads `feature_spec.json` to
build the feature vector in the exact order the model was trained on, then calls
the booster to score and rank candidates. The `feature_spec.json` emitted here
is the contract -- keep the C++ feature extraction in lock-step with it.

## Architecture

```
Problem (M, N, K, dtype, layout)
    |
    v
FeatureEngine.extract_batch()    <-- 55 features: problem, kernel, interaction, hardware
    |
    v
LGBMRegressor.predict()          <-- predicts TFLOPS for each candidate kernel
    |
    v
Sort by predicted TFLOPS          <-- rank all candidates
    |
    v
Select Top-1 kernel               <-- 98.28% mean efficiency, <1ms inference
```

Three models are trained per (op, dtype, arch):
- **TFLOPS model** (primary): used for kernel ranking
- **Latency model** (auxiliary): for latency-sensitive workloads
- **Bandwidth model** (auxiliary): for memory-bound analysis

## File Inventory

| File | Purpose |
|---|---|
| `gen_gemm_sweep_data.py` | rocke-native generator: enumerate shapes + `UniversalGemmSpec` variants, build (`build_all_instances`) + run (`sweep_run`), write training parquet |
| `data_pipeline.py` | Parse raw benchmark logs into canonical parquet datasets (legacy/import path) |
| `feature_engine.py` | 55-feature extraction: problem, kernel, interaction, hardware profile |
| `train.py` | Multi-target LGBMRegressor training with GroupKFold CV, IHEM, warm-start |
| `predict.py` | Predictor class: predict TFLOPS/latency/bandwidth, rank kernels |
| `evaluate.py` | Full evaluation: global metrics, per-shape/layout/pipeline slices |
| `search.py` | Surrogate search: discrete DE, random top-K |
| `DATA_GENERATION.md` | Detailed guide for generating training data via the rocke sweep |
| `plan.md` | Full design plan with architecture, milestones, and rationale |

## Features Used (55 total)

### Problem features (13)
`M, N, K, split_k, log2(M), log2(N), log2(K), log2(MNK),
arithmetic_intensity, aspect_ratio_mn, aspect_ratio_mk, aspect_ratio_nk, layout`

### Kernel features (17)
`tile_m, tile_n, tile_k, warp_m, warp_n, warp_k, warp_tile_m, warp_tile_n,
warp_tile_k, pipeline, scheduler, epilogue, pad_m, pad_n, pad_k, persistent,
num_warps, tile_volume, tile_mn, lds_usage_estimate, lds_usage_ratio`

### Interaction features (9)
`num_tiles_m, num_tiles_n, num_tiles_k, total_output_tiles,
tile_eff_m, tile_eff_n, tile_eff_k, overall_tile_efficiency, cu_utilization`

### Hardware profile features (12)
`hw_num_cus, hw_simds_per_cu, hw_total_simds, hw_shader_engines,
hw_max_clock_mhz, hw_max_waves_per_cu, hw_wavefront_size, hw_lds_capacity,
hw_l1_cache_kb, hw_l2_cache_kb, hw_l3_cache_kb, hw_num_xcd`

## Model Performance

### fp8 RCR, gfx950

| Metric | 108 shapes (original) | 168 shapes (wide coverage) |
|---|---|---|
| Mean TFLOPS Efficiency | 98.28% | 97.51% |
| P10 TFLOPS Efficiency | 94.64% | 93.89% |
| tiny_m (M=1) Efficiency | 95.57% | 96.04% |
| R2 (TFLOPS) | 0.997 | 0.993 |

### fp16 RCR, gfx950

Trained on 25 shapes, 1,024 kernels, 21,920 valid benchmarks.

| Metric | Value |
|---|---|
| Mean TFLOPS Efficiency | 99.36% |
| P10 TFLOPS Efficiency | 98.05% |
| P50 TFLOPS Efficiency | 100.00% |
| Min Efficiency | 95.45% |
| NDCG@1 | 64.00% |
| Top-5 Hit Rate | 88.00% |

**Shape Family Breakdown:**

| Shape Family | Mean Eff | P10 Eff | Shapes |
|---|---|---|---|
| Large M (M≥1024) | 99.54% | 99.07% | 4 |
| Medium M (128≤M<1024) | 99.62% | 98.74% | 7 |
| Small M (8≤M<128) | 98.82% | 96.22% | 8 |
| Tiny M (M<8) | 99.65% | 98.96% | 6 |

**Pipeline Breakdown:**

| Pipeline | Mean Eff | P10 Eff |
|---|---|---|
| compv3 | 99.75% | 99.09% |
| compv4 | 99.40% | 98.54% |
| mem | 99.08% | 96.59% |

Training uses `log1p(TFLOPS)` as the target by default, which normalizes the
scale across shapes spanning 0.02 to 2230 TFLOPS. This was the key finding
that improved tiny-M shapes from 84% to 96% efficiency. See
[LEARNINGS.md](LEARNINGS.md) for details.

## Validation

Training uses `GroupKFold(n_splits=5)` with group key `(M, N, K)` to ensure
the model is evaluated on shapes it has never seen during training. Layout is
excluded from the group key to force cross-layout generalization.

## Incremental Training (Warm Start)

When new benchmark data arrives, update the model without retraining from scratch:

```bash
python3 train.py \
    --data_dir data/ \
    --out_dir models/v2 \
    --warm_start models/gemm_universal_fp8_gfx950 \
    --warm_start_n_estimators 200
```

This adds 200 new trees on top of the existing model. Feature schemas must
match exactly (automatically enforced).

## Extending to New Ops

Adding support for a new operation (e.g., `gemm_streamk`, `grouped_conv`):

1. **Enumerate variants**: add the op's spec/variant grid (mirror
   `gen_gemm_sweep_data.py`, which drives `all_dispatcher_configs` +
   `build_all_instances` + `sweep_run`)
2. **Subclass `FeatureEngine`**: add op-specific features (e.g., StreamK split factor)
3. **Generate data**: sweep the variant x shape grid into a training parquet
4. **Train**: `python3 train.py --op gemm_streamk --dtype fp8 --data_dir data/ --out_dir models/`

The training, evaluation, prediction, and search infrastructure is fully
op-agnostic -- only the feature engine needs a new subclass.

## Tests

102 tests covering all modules:

```bash
python3 -m pytest tests/ -v
```

Test coverage includes:
- Log parsing with malformed JSON, empty logs, single-kernel shapes
- Feature formula correctness (tile efficiency, LDS usage, arithmetic intensity)
- Corner-case shapes: M=1, N=1, K=1, prime dimensions, 20480x7168x256
- Batch vs single extraction parity
- Parameter space validation and projection
- Predictor: single/batch prediction, ranking, missing models, empty inputs
- Training: group keys, efficiency computation, warm-start, feature compatibility
- Search: random, DE, config validity, determinism

## Documentation

- **[README.md](README.md)**: This file -- quick start, architecture, performance
- **[DATA_GENERATION.md](DATA_GENERATION.md)**: Complete guide for generating training
  data via the rocke sweep (`gen_gemm_sweep_data.py`), managing datasets, and troubleshooting
- **[LEARNINGS.md](LEARNINGS.md)**: Empirical findings and design decisions (log-transform,
  IHEM results, tiny-M analysis, feature importance, N=1/K=1 edge cases)
