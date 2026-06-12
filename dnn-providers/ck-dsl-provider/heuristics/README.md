# CK DSL Provider — Conv Heuristics

ML heuristic infrastructure for the CK DSL provider's implicit-GEMM
convolution-forward path.

## Directory layout

```
heuristics/
  models/
    grouped_conv_forward_fp16_gfx942/
      model_tflops.lgbm.gz    — compressed model (in-repo, ~14 MB)
      feature_spec.json       — feature schema used at training time
      train_manifest.json     — data provenance (rows, shapes, timestamp)
  scripts/
    convert_dsl_csv_to_parquet.py  — DSL-specific CSV→parquet converter
  sweep/
    ConvCandidateSweep.cpp  — candidate sweep: enumerates all DSL candidates,
                           compiles + times each on-device, writes training CSV
    main.cpp             — entry point (--shapes / --out CLI args)
    CMakeLists.txt       — build definition (wraps the rocm-libraries superbuild)
    build.sh             — build driver; run inside a ROCm container on the target arch
```

Shape generation, sampling, and LightGBM training are shared with the CK
dispatcher heuristics pipeline and live in:

```
projects/composablekernel/dispatcher/heuristics/
  generate_wide_coverage_conv.py
  generate_edge_dims_conv.py
  sample_conv_shapes.py
  train.py
  data_pipeline.py
  feature_engine.py
  feature_engine_grouped_conv.py
```

CSV-to-parquet conversion uses the DSL-specific script in `scripts/` because
the sweep CSV format (explicit tile columns) differs from the ckProfiler
format expected by the shared `convert_csv_to_parquet.py`.

## How the model is used at runtime

The CMake resolver `ck_dsl_provider_resolve_grouped_conv_fwd_fp16_gfx942_model()`
in `cmake/CkDslProviderPaths.cmake` walks up from the provider source directory
until it finds:

```
dnn-providers/ck-dsl-provider/heuristics/models/
  grouped_conv_forward_fp16_gfx942/model_tflops.lgbm
```

The model is stored compressed (`.lgbm.gz`). Decompress before building:

```bash
gunzip -k dnn-providers/ck-dsl-provider/heuristics/models/grouped_conv_forward_fp16_gfx942/model_tflops.lgbm.gz
```

The decompressed `.lgbm` is excluded from git via `models/.gitignore`.

## How the candidate sweep works

`ConvCandidateSweep.cpp` calls `enumerateCandidates(problem, arch)` — the same
function the production dispatcher uses — to get the complete candidate set
for a given shape. For each candidate it invokes `CompileServiceBridge` to
compile the kernel via the DSL Python codegen, loads the HIP module, and
times it on-device with `PerfMeasurement`. The measured result for every
successful candidate is written to the training CSV. There is no manually
curated config file; the sweep and the dispatcher share the same enumeration
code, so training data coverage is correct by construction.

Arguments:

| Argument | Purpose |
|---|---|
| `--shapes <path>` | CSV of conv shapes to sweep (required) |
| `--out <path>` | Path to write training rows; appended if file exists (required) |

## Retraining workflow

Variables used below:

| Variable | Description |
|---|---|
| `HEURISTICS` | path to this directory in the repo |
| `CK_HEURISTICS` | `projects/composablekernel/dispatcher/heuristics/` |
| `WORK_DIR` | writable scratch directory outside the repo |
| `N_SHARDS` | number of parallel shards (32 is typical) |

### 1. Build the oracle sweep binary

`build.sh` wraps the rocm-libraries superbuild. It uses container-internal
mount paths (`/rocm-libraries` for the repo, `/work` for scratch), so
`WORK_DIR` inside the container corresponds to wherever the scratch directory
is mounted as `/work`.

```bash
# Run inside a ROCm container on the target GPU architecture.
bash $HEURISTICS/sweep/build.sh
# Binary: <container /work>/sweep_build/oracle_sweep/conv_candidate_sweep
```

### 2. Generate and shard shapes

```bash
SHAPES=$WORK_DIR/shapes/dsl
mkdir -p $SHAPES

python3 $CK_HEURISTICS/generate_wide_coverage_conv.py --out $SHAPES/wide_coverage_conv.csv
python3 $CK_HEURISTICS/generate_edge_dims_conv.py     --out $SHAPES/edge_dims_conv.csv

python3 $CK_HEURISTICS/sample_conv_shapes.py \
    --inputs $SHAPES/wide_coverage_conv.csv $SHAPES/edge_dims_conv.csv \
    --out    $SHAPES/all_shapes.csv \
    --target 2000 \
    --shards $N_SHARDS \
    --shard_dir $SHAPES
# Produces $SHAPES/shard_00.csv .. shard_$(N_SHARDS-1).csv
```

### 3. Run the sweep

Run once per shard. Each invocation is independent and can be parallelized
across GPU nodes.

```bash
RESULTS=$WORK_DIR/results/dsl_sweep_run1
mkdir -p $RESULTS

# For each shard index NN in 00..$(N_SHARDS-1):
$WORK_DIR/sweep_build/oracle_sweep/conv_candidate_sweep \
    --shapes $SHAPES/shard_NN.csv \
    --out    $RESULTS/shard_NN.csv
```

### 4. Convert CSV output to parquet

Merge all shards before converting. Each shard CSV has a header row;
`convert_dsl_csv_to_parquet.py` skips duplicate headers automatically.

```bash
cat $RESULTS/shard_*.csv > $WORK_DIR/all_shards.csv

mkdir -p $WORK_DIR/data/dsl_run1
python3 $HEURISTICS/scripts/convert_dsl_csv_to_parquet.py \
    --input   $WORK_DIR/all_shards.csv \
    --output  $WORK_DIR/data/dsl_run1/conv_fp16_gfx942_dsl.parquet \
    --arch    gfx942 \
    --run-id  1
```

### 5. Train

```bash
cd $CK_HEURISTICS
python3 train.py \
    --data_dir  $WORK_DIR/data/dsl_run1 \
    --out_dir   $WORK_DIR/models/grouped_conv_forward_fp16_gfx942 \
    --operation grouped_conv \
    --dtype     fp16 \
    --arch      gfx942 \
    --targets   tflops
# Model: $WORK_DIR/models/grouped_conv_forward_fp16_gfx942/model_tflops.lgbm
```

### 6. Update the in-repo model

```bash
MODEL_DIR=$HEURISTICS/models/grouped_conv_forward_fp16_gfx942

gzip -9 -c $WORK_DIR/models/grouped_conv_forward_fp16_gfx942/model_tflops.lgbm \
    > $MODEL_DIR/model_tflops.lgbm.gz

cp $WORK_DIR/models/grouped_conv_forward_fp16_gfx942/{feature_spec,train_manifest}.json \
   $MODEL_DIR/
```

Commit the updated files, then run the `ConvOracleSweepGfx942` GTest suite
to validate heuristic efficiency ≥90% on the canonical shape set before merging.
