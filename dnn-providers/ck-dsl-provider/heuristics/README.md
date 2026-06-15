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
    generate_validation_shapes_conv.py — held-out validation shape set
    convert_dsl_csv_to_parquet.py      — converts sweep CSV output to parquet
  sweep/
    ConvCandidateSweep.cpp  — enumerates all DSL candidates per shape,
                              compiles + times each, writes training CSV rows
    main.cpp                — CLI entry point (--shapes / --out)
    CMakeLists.txt          — build definition (plugs into rocm-libraries superbuild)
    build.sh                — self-contained build driver; run inside a ROCm container
  training/
    train.py                — LightGBM training (GroupKFold CV, IHEM, warm-start)
    data_pipeline.py        — parquet loader / builder used by train.py
    feature_engine_grouped_conv.py — 97-feature extractor for grouped conv
```

Shape generators and the shared `feature_engine.py` base class live in:

```
projects/composablekernel/dispatcher/heuristics/
  generate_wide_coverage_conv.py  — wide-coverage training shapes
  generate_edge_dims_conv.py      — edge-case training shapes
  sample_conv_shapes.py           — stratified merge + shard
  feature_engine.py               — base class imported by feature_engine_grouped_conv.py
```

`training/` scripts are copies of their canonical sources in
`projects/composablekernel/dispatcher/heuristics/`. Apply changes to both
locations until a shared package is established.

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

## Full retraining workflow (single machine)

The steps below run entirely on one machine with a ROCm GPU. No cluster
or Slurm required. All paths are relative to the repo root unless noted.

Variables used throughout:

| Variable | Example | Description |
|---|---|---|
| `REPO` | `/home/AMD/cerb/rocm-libraries` | Absolute path to this repo |
| `HEURISTICS` | `$REPO/dnn-providers/ck-dsl-provider/heuristics` | This directory |
| `CK_HEURISTICS` | `$REPO/projects/composablekernel/dispatcher/heuristics` | Shared shape generators |
| `WORK` | `/tmp/ckdsl_retrain` | Writable scratch directory |

```bash
REPO=/home/AMD/cerb/rocm-libraries      # adjust to your checkout
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

Two generators produce complementary sets; `sample_conv_shapes.py` merges,
deduplicates, and stratified-samples them to a target count, then optionally
shards the result for parallel sweep runs.

```bash
python3 $CK_HEURISTICS/generate_wide_coverage_conv.py \
    --out $WORK/shapes/wide_coverage_conv.csv

python3 $CK_HEURISTICS/generate_edge_dims_conv.py \
    --out $WORK/shapes/edge_dims_conv.csv

python3 $CK_HEURISTICS/sample_conv_shapes.py \
    --inputs    $WORK/shapes/wide_coverage_conv.csv \
                $WORK/shapes/edge_dims_conv.csv \
    --out       $WORK/shapes/all_shapes.csv \
    --target    2000 \
    --shards    8 \
    --shard_dir $WORK/shapes

# Produces: $WORK/shapes/all_shapes.csv
#           $WORK/shapes/shard_00.csv .. shard_07.csv
```

---

### Step 3 — Build the sweep binary

`build.sh` wraps the rocm-libraries superbuild. It creates a Python venv
for pybind11 if one does not already exist, then builds `conv_candidate_sweep`
via CMake. Run this on a machine with ROCm installed.

```bash
export BUILD_DIR=$WORK/sweep_build   # default: $HOME/ckdsl_sweep_build

bash $HEURISTICS/sweep/build.sh
# Binary: $WORK/sweep_build/oracle_sweep/conv_candidate_sweep
```

The build decompresses the in-repo model automatically so the CMake resolver
can find it.

---

### Step 4 — Run the sweep

Each shard is independent. Run shards sequentially or in parallel across
terminals. The sweep appends to the output file, so interrupted runs are
safely resumed by re-invoking with the same `--out` path.

```bash
BINARY=$WORK/sweep_build/oracle_sweep/conv_candidate_sweep
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

`train.py` imports `feature_engine.py` (the shared base class) from
`CK_HEURISTICS`. Set `PYTHONPATH` so it is importable alongside the copies
in `training/`.

```bash
source $WORK/venv/bin/activate

export PYTHONPATH=$CK_HEURISTICS:${PYTHONPATH:-}

python3 $HEURISTICS/training/train.py \
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

python3 $HEURISTICS/training/train.py \
    --data_dir   $WORK/data \
    --out_dir    $WORK/models/grouped_conv_forward_fp16_gfx942 \
    --operation  grouped_conv \
    --dtype      fp16 \
    --arch       gfx942 \
    --targets    tflops \
    --warm_start $HEURISTICS/models/grouped_conv_forward_fp16_gfx942
```

---

### Step 7 — Update the in-repo model

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

Validate heuristic efficiency on the held-out validation set before merging:

```bash
python3 $HEURISTICS/scripts/generate_validation_shapes_conv.py \
    --parquet $WORK/data/conv_fp16_gfx942_dsl.parquet \
    --out     $WORK/shapes/validation/all_shapes.csv \
    --shards  4

# Run the sweep binary against the validation shards and compare
# oracle vs heuristic pick to confirm mean efficiency >= 90%.
```
