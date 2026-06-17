# CK DSL Provider — Conv Heuristics

ML heuristic infrastructure for the CK DSL provider's implicit-GEMM
convolution-forward path.

## Directory layout

```
heuristics/
  models/
    grouped_conv_forward_fp16_gfx942/
      model_tflops.lgbm.gz    — compressed LightGBM model (in-repo, ~14 MB)
      feature_spec.json       — feature schema used at training time
      cv_metrics_tflops.json  — cross-validation efficiency metrics
      feature_importances_tflops.json
      train_manifest.json     — data provenance (row count, timestamp)
  scripts/
    convert_dsl_csv_to_parquet.py        — converts sweep CSV output to parquet
  sweep/
    ConvCandidateSweep.cpp  — enumerates all DSL candidates per shape,
                              compiles + times each, writes training CSV rows
    main.cpp                — CLI entry point (--shapes / --out)
    CMakeLists.txt          — build definition (plugs into rocm-libraries superbuild)
    build.sh                — self-contained build driver; run inside a ROCm container
```

Training and shape-generation scripts live in `projects/composablekernel/dispatcher/heuristics/`
and are called directly from there:

```
projects/composablekernel/dispatcher/heuristics/
  train.py                        — LightGBM training (GroupKFold CV, IHEM, warm-start)
  data_pipeline.py                — parquet loader / builder used by train.py
  feature_engine_grouped_conv.py  — 101-feature extractor for grouped conv (see Features)
  feature_engine.py               — base class imported by feature_engine_grouped_conv.py
  generate_wide_coverage_conv.py      — wide-coverage training shapes
  generate_edge_dims_conv.py          — edge-case training shapes
  generate_targeted_shapes_conv.py    — OOF-driven targeted top-up shape generation
  sample_shapes_conv.py               — stratified merge + shard
```

## How the model is used at runtime

`DslMlHeuristic` loads models from the directory pointed to by
`CK_DSL_ML_MODEL_DIR`. Expected layout:

```
$CK_DSL_ML_MODEL_DIR/
  conv/model_tflops.lgbm    ← conv-forward model (decompressed)
  gemm/model_tflops.lgbm   ← GEMM model (optional)
  fmha/model_tflops.lgbm   ← FMHA model (optional)
```

The in-repo model is stored compressed. Decompress before use:

```bash
gunzip -k dnn-providers/ck-dsl-provider/heuristics/models/grouped_conv_forward_fp16_gfx942/model_tflops.lgbm.gz
# Point the runtime at a directory that has conv/model_tflops.lgbm:
mkdir -p /tmp/ckdsl_models/conv
cp dnn-providers/ck-dsl-provider/heuristics/models/grouped_conv_forward_fp16_gfx942/model_tflops.lgbm \
   /tmp/ckdsl_models/conv/
export CK_DSL_ML_MODEL_DIR=/tmp/ckdsl_models
```

The decompressed `.lgbm` is excluded from git via `models/.gitignore`.

---

## Validation

After updating a model, spot-check heuristic quality against a held-out oracle
sweep using `validate_ml_vs_oracle_conv.py`:

```bash
python3 $CK_HEURISTICS/validation/grouped_conv/validate_ml_vs_oracle_conv.py \
    --model          $HEURISTICS/models/grouped_conv_forward_fp16_gfx942 \
    --oracle-parquet $WORK/results/validation_sweep.parquet
```

The script compares the model's top-1 kernel choice against the empirically
fastest kernel for each shape and reports mean/P10 efficiency.

---

## Full retraining workflow (single machine)

The steps below run entirely on one machine with a ROCm GPU. No cluster
or Slurm required. All paths are relative to the repo root unless noted.

Replace `gfx942` with `gfx90a` or `gfx950` throughout to retrain for a
different architecture. The `--arch` flag controls hardware feature injection
in `convert_dsl_csv_to_parquet.py` and the output directory name convention.

Variables used throughout:

| Variable | Example | Description |
|---|---|---|
| `REPO` | `<path/to>/rocm-libraries` | Absolute path to this repo |
| `HEURISTICS` | `$REPO/dnn-providers/ck-dsl-provider/heuristics` | This directory |
| `CK_HEURISTICS` | `$REPO/projects/composablekernel/dispatcher/heuristics` | Shared shape generators |
| `WORK` | `/tmp/ckdsl_retrain` | Writable scratch directory |

```bash
REPO="${REPO:-$(git rev-parse --show-toplevel)}"   # or set to your checkout
HEURISTICS=$REPO/dnn-providers/ck-dsl-provider/heuristics
CK_HEURISTICS=$REPO/projects/composablekernel/dispatcher/heuristics
WORK=/tmp/ckdsl_retrain
mkdir -p $WORK/{shapes,results,data,models}
```

---

### Step 1 — Install Python dependencies

Training requires LightGBM, pandas, pyarrow, and scikit-learn. The sweep
build additionally needs pybind11 (handled automatically by `build.sh`).

```bash
python3 -m venv $WORK/venv
source $WORK/venv/bin/activate
pip install lightgbm pandas pyarrow scikit-learn
```

---

### Step 2 — Generate training shapes

Two generators produce complementary sets; `sample_shapes_conv.py` merges,
deduplicates, and shards the result for parallel sweep runs.

```bash
python3 $CK_HEURISTICS/generate_wide_coverage_conv.py \
    --out $WORK/shapes/wide_coverage_conv.csv

python3 $CK_HEURISTICS/generate_edge_dims_conv.py \
    --out $WORK/shapes/edge_dims_conv.csv

python3 $CK_HEURISTICS/sample_shapes_conv.py \
    --inputs    $WORK/shapes/wide_coverage_conv.csv \
                $WORK/shapes/edge_dims_conv.csv \
    --out       $WORK/shapes/all_shapes.csv \
    --shards    8 \
    --shard_dir $WORK/shapes

# Produces: $WORK/shapes/all_shapes.csv
#           $WORK/shapes/shard_00.csv .. shard_07.csv
```

---

### Step 3 — Build the sweep binary

`build.sh` builds `conv_candidate_sweep` via CMake. The sweep uses the
pure-C ck_dsl engine (`libckc_core`) to JIT-compile and time every candidate
— no Python, no pybind11, no hipdnn SDK. Run this on a machine with ROCm
installed.

```bash
export BUILD_DIR=$WORK/sweep_build   # default: $HOME/ckdsl_sweep_build

bash $HEURISTICS/sweep/build.sh
# Binary: $WORK/sweep_build/conv_candidate_sweep
```

If `libckc_core.a` is not under `/opt/rocm`, set `CKC_CORE_LIB` before calling
`build.sh`:

```bash
export CKC_CORE_LIB=/path/to/libckc_core.a
bash $HEURISTICS/sweep/build.sh
```

---

### Step 4 — Run the sweep

Each shard is independent. Run shards sequentially or in parallel across
terminals. The sweep appends to the output file, so interrupted runs are
safely resumed by re-invoking with the same `--out` path.

```bash
BINARY=$WORK/sweep_build/conv_candidate_sweep
mkdir -p $WORK/results

for shard in $WORK/shapes/shard_*.csv; do
    name=$(basename $shard .csv)
    $BINARY --shapes $shard --out $WORK/results/${name}.csv
done
```

Each output row records one (shape, candidate) timing measurement:
`N,G,C,K,Hi,Wi,Y,X,stride_h,stride_w,pad_h,pad_w,tile_m,tile_n,tile_k,pipeline,tflops,latency_us`

---

### Step 5 — Convert sweep output to parquet

Merge all shard CSVs and convert to the parquet format `train.py` expects.
`convert_dsl_csv_to_parquet.py` skips duplicate header rows automatically.

```bash
cat $WORK/results/shard_*.csv > $WORK/all_shards.csv

python3 $HEURISTICS/scripts/convert_dsl_csv_to_parquet.py \
    --input  $WORK/all_shards.csv \
    --output $WORK/data/conv_fp16_gfx942_dsl.parquet \
    --arch   gfx942 \
    --run-id 1
```

---

### Step 6 — Train

```bash
source $WORK/venv/bin/activate

python3 $CK_HEURISTICS/train.py \
    --data_dir  $WORK/data \
    --out_dir   $WORK/models/grouped_conv_forward_fp16_gfx942 \
    --operation grouped_conv \
    --dtype     fp16 \
    --arch      gfx942 \
    --targets   tflops
# Model: $WORK/models/grouped_conv_forward_fp16_gfx942/model_tflops.lgbm
```

`train.py` runs 5-fold GroupKFold cross-validation and prints per-fold
TFLOPS efficiency before training the final model on all data.

To warm-start from the current in-repo model (adds trees on top rather than
retraining from scratch):

```bash
gunzip -k $HEURISTICS/models/grouped_conv_forward_fp16_gfx942/model_tflops.lgbm.gz

python3 $CK_HEURISTICS/train.py \
    --data_dir   $WORK/data \
    --out_dir    $WORK/models/grouped_conv_forward_fp16_gfx942 \
    --operation  grouped_conv \
    --dtype      fp16 \
    --arch       gfx942 \
    --targets    tflops \
    --warm_start $HEURISTICS/models/grouped_conv_forward_fp16_gfx942
```

---

### Step 7 — Evaluate OOF efficiency and decide whether to iterate

`train.py` writes `oof_predictions.parquet` alongside the model. This contains
one row per (shape, candidate) with the model's out-of-fold prediction — the
same prediction the model would make on data it has never seen. Use
`generate_targeted_shapes_conv.py` to turn these into a per-subset efficiency
report:

```bash
python3 $CK_HEURISTICS/generate_targeted_shapes_conv.py \
    --oof     $WORK/models/grouped_conv_forward_fp16_gfx942/oof_predictions.parquet \
    --train   $WORK/data/conv_fp16_gfx942_dsl.parquet \
    --analytics --dry-run
```

This prints:
- Global mean/P10/P50/P90 top-1 efficiency
- Per-subset breakdown sorted worst-first (N × group_type × spatial × channel × filter)
- Worst 20 individual shapes; `actual_tflops_of_pred_best` is the measured tflops of the model's top-1 kernel choice (not the model's predicted tflops)

**Target: mean ≥ 0.90 and P10 ≥ 0.75 across all subsets.**

If satisfied, proceed to Step 8. If subsets are below threshold, run a targeted
top-up sweep (Step 7a) before updating the model.

---

### Step 7a — Targeted top-up (if OOF reveals hard subsets)

`generate_targeted_shapes_conv.py` identifies the subset buckets where the
model's top-1 pick is worst, then generates a dense grid of new shapes covering
only those buckets. The grid spans `_N_VALUES × _C_VALUES × _K_VALUES ×
_HW_VALUES × _FILTER_PADS × _STRIDES × _G_VALUES`, filtered to shapes that
(a) fall in a targeted subset and (b) are not already in the training set.
`--density 2` expands the grid with intermediate values; `--target N` applies
stratified sampling to cap the output.

```bash
# Generate 500 targeted shapes, single shard (no array needed for top-up)
python3 $CK_HEURISTICS/generate_targeted_shapes_conv.py \
    --oof       $WORK/models/grouped_conv_forward_fp16_gfx942/oof_predictions.parquet \
    --train     $WORK/data/conv_fp16_gfx942_dsl.parquet \
    --out       $WORK/shapes/topup/shard_00.csv \
    --shards    1 \
    --target    500 \
    --density   2 \
    --threshold 0.90
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--oof` | required | `oof_predictions.parquet` from `train.py` |
| `--train` | required | Training parquet; generated shapes are guaranteed non-overlapping |
| `--out` | required | Output CSV path |
| `--target` | all | Cap on output shapes; stratified sampling preserves bucket diversity |
| `--density` | 1 | `2` = denser grid (adds intermediate N/spatial/channel values) |
| `--threshold` | 0.90 | Mean efficiency below which a subset is targeted |
| `--analytics` | off | Print global stats and worst 20 shapes |
| `--dry-run` | off | Analyse only; do not write shape files |

Sweep the generated shapes (Steps 3–5 above), place the resulting parquet in
the same data directory as the original training parquet (or pass both to
`train.py --data_dir`), then warm-start retrain:

```bash
python3 $CK_HEURISTICS/train.py \
    --data_dir   $WORK/data/topup \
    --out_dir    $WORK/models/grouped_conv_forward_fp16_gfx942_v2 \
    --operation  grouped_conv \
    --dtype      fp16 \
    --arch       gfx942 \
    --targets    tflops \
    --warm_start $WORK/models/grouped_conv_forward_fp16_gfx942
```

Warm-start adds trees on top of the existing model without disturbing prior
trees. It requires an identical feature schema — if `feature_engine_grouped_conv.py`
has changed (features added/removed), a full retrain from scratch is necessary.

Repeat Steps 6–7a until OOF targets are met.

---

### Step 8 — Update the in-repo model

When CV efficiency is satisfactory, compress and commit:

```bash
MODEL_SRC=$WORK/models/grouped_conv_forward_fp16_gfx942
MODEL_DST=$HEURISTICS/models/grouped_conv_forward_fp16_gfx942

gzip -9 -c $MODEL_SRC/model_tflops.lgbm > $MODEL_DST/model_tflops.lgbm.gz

cp $MODEL_SRC/feature_spec.json               $MODEL_DST/
cp $MODEL_SRC/cv_metrics_tflops.json          $MODEL_DST/
cp $MODEL_SRC/feature_importances_tflops.json $MODEL_DST/
cp $MODEL_SRC/train_manifest.json             $MODEL_DST/

git add $MODEL_DST
git commit -m "[CK DSL] conv model: retrain fp16/gfx942 ($(date +%Y-%m-%d))"
```

Validate heuristic efficiency using the OOF predictions produced during training:

```bash
# Inspect per-subset efficiency from the last training run.
python3 $CK_HEURISTICS/generate_targeted_shapes_conv.py \
    --oof      oof_predictions.parquet \
    --train    conv_fp16_<arch>_dsl.parquet \
    --analytics --dry-run
# Target: mean efficiency >= 0.90 across all subsets.
```

If subsets are below threshold, generate a targeted top-up shape set and re-sweep:

```bash
# Generate shapes covering hard subsets (zero overlap with existing training data).
python3 $CK_HEURISTICS/generate_targeted_shapes_conv.py \
    --oof   oof_predictions.parquet \
    --train conv_fp16_<arch>_dsl.parquet \
    --out   all_shapes.csv \
    --shards 32

# Sweep the targeted shapes, convert, and warm-start retrain.
# See sweep/build.sh and the full retraining workflow above.
```

---

## Features

`feature_engine_grouped_conv.py` extracts per-(shape, candidate) features fed
to the LightGBM model. Features fall into four tiers:

| Tier | Description | Count |
|------|-------------|-------|
| Shape | N, G, C, K, Hi, Wi, filter, stride, pad, derived spatial/channel dims | ~30 |
| Candidate tile | gemm_m/n/k per block, pipeline, wave_mode, block_size, has_dsb/si | ~15 |
| Hardware | CU count, SIMD/CU, shader engines, clock, wavefront size, cache sizes | ~15 |
| Interaction | K_per_C (K/C), GEMM M/N/K, occupancy estimates, bucket indicators | ~39 |

`K_per_C = K / C` is the directional channel ratio. It allows the model to
distinguish `C=64, K=256` (K/C=4: more outputs than inputs) from `C=256, K=64`
(K/C=0.25: more inputs than outputs) in a single split, rather than requiring a
two-condition conjunction from raw C and K values. This is important because the
GEMM mapping is asymmetric: C maps to GEMM K_gemm (contraction dimension) and K
maps to GEMM N (output dimension), so the optimal tile differs between the two cases.

The feature schema version is recorded in `feature_spec.json` alongside each
trained model. Warm-start retraining requires an identical schema — any change
to feature count or order forces a full retrain from scratch.
