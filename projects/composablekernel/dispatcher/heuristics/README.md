# CK Tile Heuristics: ML-Based Kernel Selection

Fast, accurate kernel selection for CK Tile operations using LightGBM regression
with Origami-augmented feature engineering.

## What This Does

Instead of running all 4608+ kernel configurations on the GPU to find the best
one (exhaustive search taking ~46 seconds per shape), this system trains an ML
model that predicts TFLOPS for any (problem, kernel) pair in microseconds. It
scores all candidates instantly and picks the best kernel -- achieving 98.28%
of oracle-best TFLOPS efficiency across 108 tested shapes.

## Quick Start

### 1. Generate and convert benchmark data

**Step 1: Generate benchmark data**

```bash
python3 generate_benchmark_data.py \
    --build_dir /path/to/build \
    --output_dir data/fp16_original \
    --dtype fp16 \
    --layout rcr \
    --num_build_jobs 4 \
    --warmup 10 \
    --repeat 50
```

This outputs JSON with all benchmark results.

**Step 2: Convert JSON to parquet training format**

```bash
python3 convert_json_to_parquet.py \
    --input data/fp16_original/benchmark_results_fp16_rcr.json \
    --output data/fp16_original/fp16_training_data.parquet \
    --arch gfx950
```

The converter automatically fixes pad flags for `_mem` kernels and validates data.

**Alternative: Parse existing logs**

If you have raw benchmark logs from CK Tile:

```bash
python3 data_pipeline.py ck_tile_testrun_2.log \
    -o data/gemm_universal_fp8_rcr_gfx950.parquet \
    --arch gfx950 --capture_hw
```

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

### 6. Using models in C++ (requires decompression)

C++ code uses the LightGBM C API which requires uncompressed `.lgbm` files. If you have compressed models (`.lgbm.gz`), decompress them first:

```bash
cd models/gemm_universal_fp16_gfx950
gunzip model_tflops.lgbm.gz
```

Then use in C++ examples:

```bash
cd dispatcher/build
./gemm_09_ml_heuristic --model ../heuristics/models/gemm_universal_fp16_gfx950/model_tflops.lgbm
```

**Note**: Python tools automatically decompress `.lgbm.gz` files on first use, so you can run Python scripts first to trigger decompression, then use the same models in C++.

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
| `generate_benchmark_data.py` | Build and run benchmarks across ~25 diverse problem sizes, output JSON |
| `convert_json_to_parquet.py` | Convert benchmark JSON to parquet training format, fix `_mem` pad flags |
| `data_pipeline.py` | Parse raw benchmark logs into canonical parquet datasets |
| `feature_engine.py` | 55-feature extraction: problem, kernel, interaction, hardware profile |
| `train.py` | Multi-target LGBMRegressor training with GroupKFold CV, IHEM, warm-start |
| `predict.py` | Predictor class: predict TFLOPS/latency/bandwidth, rank kernels |
| `evaluate.py` | Full evaluation: global metrics, per-shape/layout/pipeline slices |
| `search.py` | Surrogate search: discrete DE, random top-K |
| `generate_wide_coverage.py` | Generate benchmark data across 706 diverse shapes |
| `generate_edge_dims.py` | Generate N=1, K=1, and other edge-case shapes |
| `DATA_GENERATION.md` | Detailed guide for building binaries and generating data |
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

1. **Build binaries**: `ninja -C build benchmark_gemm_streamk_fp8_rcr`
2. **Subclass `FeatureEngine`**: add op-specific features (e.g., StreamK split factor)
3. **Generate data**: run benchmarks across diverse shapes
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
- **[DATA_GENERATION.md](DATA_GENERATION.md)**: Complete guide for building tile engine
  binaries, running benchmarks, managing datasets, and troubleshooting
- **[LEARNINGS.md](LEARNINGS.md)**: Empirical findings and design decisions (log-transform,
  IHEM results, tiny-M analysis, feature importance, N=1/K=1 edge cases)

## Grouped Convolution ML Heuristics

### Overview

ML-based kernel selection for grouped convolution operations (forward, bwd_data, bwd_weight) on gfx950 with bf16 precision.

### Results

#### Forward Pass Model
- **Training Data**: 187 MIOpen problems × 30 kernels = 5,640 measurements  
- **Cross-validation Performance**:
  - Mean Efficiency: **96.1%**
  - P10 Efficiency: **87.2%**
  - R²: **0.9559**

#### BWD_DATA Model
- **Training Data**: 1,242 synthetic problems × 20 kernels = 14,562 measurements
- **Validation**: 10 unseen problems
- **Performance**:
  - Mean Efficiency: **100.0%**
  - P10 Efficiency: **99.9%**
  - P50 Efficiency: **100.0%**
  - Cross-validation R²: **0.9924**

#### BWD_WEIGHT Model
- **Training Data**: 2,009 synthetic problems × 20 kernels = 18,150 measurements
- **Validation**: 10 unseen problems
- **Performance**:
  - Mean Efficiency: **100.0%**
  - P10 Efficiency: **100.0%**
  - P50 Efficiency: **100.0%**
  - Cross-validation R²: **0.9948**

### Training Data Generation

Extended synthetic problem sets for backward passes cover diverse scenarios:
- Small spatial (7×7, 14×14) + various channels (64-1024)
- Medium spatial (28×28, 32×32, 56×56) + various channels (32-512)
- Large spatial (112×112) + small/medium channels (16-256)
- Asymmetric C/K combinations
- Small and large batch sizes (N=1 to 128)
- Grouped convolutions (G=2, 4, 8)
- Depthwise convolutions (G=C=K)
- Stride-2 downsampling

### Model Files

Trained models stored in:
- `models/grouped_conv_forward_bf16_gfx950/`
- `models/grouped_conv_bwd_data_bf16_gfx950/`
- `models/grouped_conv_bwd_weight_bf16_gfx950/`

Each contains:
- `model_tflops.lgbm` - LightGBM model (compressed with gzip)
- `feature_spec.json` - Feature configuration
- `cv_metrics_tflops.json` - Cross-validation metrics
- `feature_importances_tflops.json` - Feature importance rankings

Models are automatically decompressed on first use.

### Usage

```python
from predict_grouped_conv import predict_best_kernel

# Define problem
problem = {
    'N': 16, 'C': 256, 'K': 128, 'G': 1,
    'Hi': 28, 'Wi': 28, 'Y': 3, 'X': 3,
    'stride_h': 1, 'stride_w': 1,
    'pad_h': 1, 'pad_w': 1,
    'dtype': 'bf16'
}

# Get best kernel for backward data gradient
best_kernel = predict_best_kernel(
    problem=problem,
    variant='bwd_data',
    arch='gfx950'
)

print(f"Best kernel: {best_kernel['name']}")
print(f"Predicted TFLOPS: {best_kernel['tflops']:.2f}")
```

### Validation

Run validation against oracle benchmarks:

```bash
cd projects/composablekernel/tile_engine/ops/grouped_conv
python3 validate_ml_vs_oracle.py --variant bwd_data
python3 validate_ml_vs_oracle.py --variant bwd_weight
```

