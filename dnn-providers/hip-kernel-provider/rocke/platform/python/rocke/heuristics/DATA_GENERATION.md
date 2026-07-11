# Data Generation Guide

This document explains how to generate training data for the ML kernel
performance prediction system using the **rocke sweep** -- entirely within
`rocke`, with no CK Tile / CMake / ninja build step.

## Overview

The ML heuristic needs benchmark data: measured TFLOPS, latency, and bandwidth
for every (problem shape, kernel config) pair. `gen_gemm_sweep_data.py` produces
this end-to-end inside `rocke`:

1. **Enumerate a shape corpus** `(M, N, K)` covering inference / training / edge
   cases.
2. **Enumerate kernel-config variants** as `UniversalGemmSpec` objects via
   `rocke.instances.all_dispatcher_configs` (already validity-filtered by
   `is_valid_spec`).
3. **Build every variant on the fly** with `rocke.sweep.build_all_instances`
   (LLVM IR -> HSACO, cached on disk).
4. **Run each `(variant, shape)`** through `rocke.sweep_bench.sweep_run` to
   record per-shape TFLOPS and correctness.
5. **Write a training parquet** with exactly the columns
   `feature_engine.GemmUniversalFeatureEngine` / `train.py` consume.

```
shape corpus  (M,N,K)  ---\
                           >--  build_all_instances  -->  cached HSACOs
UniversalGemmSpec variants /         (LLVM IR -> HSACO)
                                           |
                                           v
                                     sweep_run        -->  TFLOPS + correctness
                                  (per (variant, shape))         per shape
                                           |
                                           v
                                  training parquet  (canonical schema)
```

Because the config columns are read directly off each `UniversalGemmSpec`
(authoritative), the parquet is always self-consistent with the kernels that
were actually built -- no kernel-name parsing.

## Prerequisites

- **ROCm**: HIP >= 6.0.3 (for gfx950: HIP >= 6.0.4)
- **Python**: 3.10+ with `pandas`, `pyarrow`
- **GPU**: ROCm-capable AMD GPU (MI250X, MI300X, MI355X, etc.)
- **rocke**: importable on `PYTHONPATH` (the generator drives `rocke.sweep`
  and `rocke.sweep_bench` directly; no separate binary build is required)

---

## Quick Start

### 1. Generate training data with the rocke sweep

Run the generator as a module from the `python/` directory (so `rocke` is
importable):

```bash
python3 -m rocke.heuristics.gen_gemm_sweep_data \
    --out data/gemm_universal_gfx950.parquet \
    --cache-dir /tmp/rocke_sweep_cache \
    --arch gfx950 \
    --shape-set wide
```

This builds the variant grid (cached under `--cache-dir`), runs every
`(variant, shape)` pair, and writes the training parquet directly -- no
intermediate JSON or log files, and no `data_pipeline.py` conversion step.

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--out` | (required) | Output training parquet path. |
| `--cache-dir` | `/tmp/rocke_sweep_cache` | Cached HSACO binaries + sweep manifest/results. Reused across runs. |
| `--arch` | `gfx950` | Target GPU architecture (e.g. `gfx942`, `gfx950`). |
| `--shape-set` | `wide` | Shape corpus: `wide`, `edge`, or `all`. |
| `--max-shapes` | (all) | Cap the number of shapes (smoke tests / quick iteration). |
| `--pipelines` | `compv3,compv4` | Comma-separated pipeline families to enumerate. |
| `--epilogues` | `default,cshuffle` | Comma-separated epilogue families to enumerate. |
| `--parallel` | `os.cpu_count()` | Build parallelism (`1` = serial). |
| `--attempts` | `3` | Fresh-process perf attempts per `(variant, shape)`. |
| `--launcher` | (python) | Optional C++ launcher; omit to use `python -m rocke.run_manifest`. |

**Smoke test (a handful of shapes, fast):**
```bash
python3 -m rocke.heuristics.gen_gemm_sweep_data \
    --out /tmp/smoke.parquet --shape-set wide --max-shapes 16
```

**Edge-case corpus (N=1, K=1, tiny dims):**
```bash
python3 -m rocke.heuristics.gen_gemm_sweep_data \
    --out data/gemm_universal_gfx950_edge.parquet --shape-set edge
```

### Shape corpora

The shape corpus is generated in-process (no external scripts):

- **`wide`** (`generate_wide_shapes()`): comprehensive coverage -- M=1
  single-token inference, tiny/small/medium/large M, square powers-of-two,
  skinny/tall, deep/shallow K, prime dimensions, and LLM-specific shapes
  (DeepSeek MoE, LLaMA-7B/70B, GPT-style attention).
- **`edge`** (`generate_edge_shapes()`): degenerate / edge cases -- N=1
  (vector-matrix), K=1 (rank-1 update), M=1, all-ones, and small N/K (2-16).
- **`all`**: the union of `wide` and `edge`.

These corpora were folded in verbatim from the retired CK-Tile
`generate_wide_coverage.py` / `generate_edge_dims.py`, so coverage is unchanged.

### 2. Train a model

The parquet is already in the canonical schema, so training is a direct call:

```bash
python3 train.py \
    --data_dir data/ \
    --out_dir models/gemm_universal_gfx950 \
    --op gemm_universal --arch gfx950
```

The downstream pipeline (`train.py` / `predict.py` / `evaluate.py` /
`search.py` / `feature_engine.py`) is unchanged -- it is data-source-agnostic
and consumes the same canonical schema the CK-Tile path used to emit, so
existing models keep working.

## Canonical Data Schema

Every parquet file follows this schema (each row is one `(variant, shape)`
pair; the config columns are recovered directly from the `UniversalGemmSpec`):

| Column | Type | Description |
|---|---|---|
| `op_type` | str | `gemm_universal` |
| `dtype` | str | `fp8`, `fp16`, `bf16`, `bf8` |
| `layout` | str | `rcr`, `rrr`, `crr`, `ccr` |
| `arch` | str | `gfx942`, `gfx950`, etc. |
| `kernel_name` | str | Full kernel identifier (`spec.kernel_name()`) |
| `m`, `n`, `k` | int | Problem dimensions |
| `split_k` | int | Split-K factor (1 = standard) |
| `measured_tflops` | float | Ground-truth TFLOPS (0 if incorrect/failed) |
| `latency_ms` | float | Latency derived from TFLOPS |
| `bandwidth_gb_s` | float | Bandwidth derived from latency |
| `is_valid` | bool | True if the variant ran correctly with tflops > 0 |
| `tile_m`, `tile_n`, `tile_k` | int | Tile dimensions |
| `warp_m`, `warp_n`, `warp_k` | int | Warp config |
| `warp_tile_m/n/k` | int | Warp tile dimensions |
| `pipeline` | str | `compv3`, `compv4`, etc. |
| `scheduler` | str | `intrawave`, `interwave` |
| `epilogue` | str | `cshuffle`, `default` |
| `pad_m`, `pad_n`, `pad_k` | bool | Padding flags |
| `persistent` | bool | Persistent kernel flag |
| `run_id` | int | Collection run identifier |

Rows that fail build, verification, or perf are emitted with `is_valid=False`
and zero targets, so the model can learn the failure surface (same convention
the CK-Tile path used).

## Shape Selection Guidelines

Good training data requires diverse shapes. The `wide` and `edge` corpora
already cover the regimes below; extend `generate_wide_shapes()` /
`generate_edge_shapes()` in `gen_gemm_sweep_data.py` if you need more.

### By M dimension (batch size / output rows)
- **M=1**: single-token inference (hardest case for tiling)
- **Tiny M (2-16)**: small batch inference
- **Small M (32-128)**: medium batch
- **Medium M (256-2048)**: large batch / training
- **Large M (4096-20480)**: very large batch

### By N and K dimension
- **N=1**: vector-matrix multiply (degenerate)
- **K=1**: rank-1 update / outer product (degenerate)
- **Small N or K (2-16)**: stress tile efficiency
- **Deep K (K > 4096)**: compute-bound regime
- **Shallow K (K < 256)**: memory-bound regime

### By shape family
- **Square**: M ~ N ~ K (powers of 2)
- **Tall**: M >> N (tall output matrix)
- **Wide**: N >> M (wide output matrix)
- **Deep-K**: K >> M and K >> N

### Special cases
- **Prime dimensions**: 17, 31, 127, 251, 509, 1021, 2039, 4093
  (worst-case for tile alignment, tests padding logic)
- **Non-power-of-2**: 48, 96, 192, 384, 576, 768, 1536, 3072, 4608
  (common in LLM architectures)
- **LLM inference shapes**: DeepSeek, LLaMA-7B, LLaMA-70B MLP/attention dims

### Minimum recommended coverage

For a production-quality model, aim for:
- At least 200 unique (M, N, K) shapes (the `wide` corpus is well above this)
- At least 10 shapes per shape family
- The full variant grid run against every shape
- Multiple layouts if training a cross-layout model

## Benchmark Quality Guidelines

### Attempts
- The generator runs `--attempts` fresh-process perf passes per
  `(variant, shape)` and keeps the median TFLOPS. Default `3`; raise it for
  more stable measurements.

### Noise handling
- TFLOPS is taken as the median across attempts.
- Avoid benchmarking under thermal throttling (check GPU temperature).
- Lock GPU clocks if possible for reproducibility.

### Environment metadata
Store with every dataset:
- GPU model and architecture (`--arch`)
- ROCm driver version
- Clock mode (default / locked)
- Git hash of the rocke checkout
- Timestamp

## Extending to a New Layout or Dtype

The same generator covers new layouts and dtypes via the variant grid:

- The enumerated `UniversalGemmSpec` variants already span the dtype/layout
  families supported by `all_dispatcher_configs`.
- Train a cross-layout model by putting all layouts in the same `data_dir`; the
  feature engine includes `layout` as a categorical feature, so one model can
  handle all layouts.

```bash
# Train a cross-layout model over all generated parquets
python3 train.py --data_dir data/ \
    --out_dir models/gemm_universal_gfx950_all_layouts
```

## Incremental Data Collection

When you want to add more data:

1. Generate new data (new shapes, new arch, etc.) into a new parquet in the
   same `data_dir`. The HSACO cache under `--cache-dir` is reused, so only
   new variants are rebuilt.
2. Warm-start from the previous model:
   ```bash
   python3 train.py --data_dir data/ --out_dir models/v2 \
       --warm_start models/v1 \
       --warm_start_n_estimators 200
   ```

This adds 200 new trees on top of the existing model. The feature schema must
match exactly (enforced automatically).

## File Organization

Recommended directory structure:

```
heuristics/
  data/
    gemm_universal_gfx950.parquet        # wide corpus
    gemm_universal_gfx950_edge.parquet   # edge corpus
  models/
    gemm_universal_gfx950/               # trained model artifacts
      model_tflops.lgbm
      model_latency.lgbm
      model_bandwidth.lgbm
      feature_spec.json
      train_manifest.json
      cv_metrics_tflops.json
      eval_report.json
      ...
/tmp/rocke_sweep_cache/                 # cached HSACOs + sweep manifest/results
```

## Troubleshooting

### Some variants are marked `is_valid=False`
Some kernel configs are invalid or fail to build/verify for certain problem
sizes (e.g., tile_m=256 with M=16). These rows are emitted with `is_valid=False`
and zero targets, and are filtered out during training. This is expected and
gives the model a failure surface to learn from.

### Edge dims produce very few valid results
N=1 and K=1 shapes are degenerate -- most kernel configurations have minimum
dimension requirements and will fail or produce zero TFLOPS. The small number
of valid results is still useful (it tells the model which configs work for
these shapes).

### Generation is slow
Each shape runs the full variant grid. Tips:
- Run on a dedicated GPU (no other workloads).
- Use `--max-shapes` to iterate on a subset first.
- Keep `--cache-dir` stable so HSACOs are reused across runs (only new
  variants rebuild).
- Lower `--attempts` for quick iteration; raise it for production quality.

### Data from different GPUs / driver versions
Store `run_id`, `arch`, and hardware metadata with each dataset. Training on
mixed data is allowed but not recommended for production models. Filter to a
single `arch` for clean experiments.
