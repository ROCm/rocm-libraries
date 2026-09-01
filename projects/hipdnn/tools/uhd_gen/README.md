# UHD Generation Tool

Train and export heuristic models for hipDNN's Universal Heuristic Descriptor (UHD) system.

## Overview

This tool takes benchmark timing data and produces:
1. A trained LightGBM model
2. A FlatBuffer model artifact (`model.bin`) for `TreeDataAdapter`
3. A UHD descriptor (`<stem>.uhd.json`) for `DescriptorLoader` (RFC 0019 §4)

…and then installs that pair into a descriptor tree, pointing an engine's UED at it.

## Installation

```bash
cd projects/hipdnn/tools/uhd_gen
pip install -e .
```

## The pipeline

```
  sweep  ->  export-benchmarks  ->  train  ->  promote
   |              |                   |          |
   |              |                   |          `- writes the UED's "heuristic" id
   |              |                   `- <stem>.uhd.json + model.bin
   |              `- §8.3 training CSV
   `- ingestor benchmark log
```

```bash
# 1. sweep: run the graphs you care about with benchmark logging on
HIPDNN_LOG_LEVEL=info HIPDNN_LOG_FILE=sweep.log <run your graphs>

# 2. export-benchmarks: log -> the §8.3 training CSV
python -m uhd_gen export-benchmarks sweep.log -o bench.csv

# 3. train: CSV -> descriptor + model artifact
python -m uhd_gen train \
    --input bench.csv \
    --features q.M q.N q.K kernel.tile_m kernel.tile_n kernel.tile_k device.cu_count \
    --target tflops \
    --group-by q.M q.N q.K \
    --output-dir ./uhd_output \
    --descriptor-name gemm \
    --name "GEMM UHD"

# 4. promote: install the pair beside the engine's UED and write its heuristic id
python -m uhd_gen promote \
    --model-dir ./uhd_output \
    --descriptor-tree ./descriptors \
    --engine hipkernel:gemm
```

**Step 4 is not optional.** Until a UED's `heuristic` field names the new UHD's id,
`DescriptorLoader` resolves nothing and the engine ranks its kernels by priority then
descriptor id — a perfectly valid state that logs no error. The only symptom of a
skipped promote is that the model "did nothing".

### Feature columns must be namespace-qualified

Every `--features` column has to start with `q.`, `kernel.`, or `device.` — the three
namespaces the runtime binds (RFC 0019 §7.1):

| Namespace | Source | Example |
|-----------|--------|---------|
| `q.` | Problem / query shape | `q.M`, `q.seqlen_q` |
| `kernel.` | Per-candidate UKD metadata | `kernel.tile_m`, `kernel.split_k` |
| `device.` | Device properties | `device.cu_count` |

The tool rejects unqualified names, and has to: a bare `cu_count` becomes `$cu_count`
in the signature, which the runtime cannot resolve. Every selection then throws
`Undefined variable` and quietly degrades to static ordering. Nothing downstream
catches it — descriptor registration only inspects `$kernel.`-prefixed references — so
an unqualified descriptor loads, validates, and never once uses the model.

Rename the columns in your CSV to match.

### Constant feature columns are dropped, loudly

A column with one value across the whole input cannot separate one candidate from
another. `train` detects those before fitting and drops them from
`features_signature`, naming each one and its value:

```
WARNING - DROPPED constant feature column kernel.tile_m: every row is 128. It cannot
separate one candidate from another, so it is NOT in the trained features_signature
and features_hash is over the smaller set. ...
```

`features_hash` is computed over the signature that was actually trained, so the
descriptor loads; `train_manifest.json` records `requested_features`,
`constant_features` (with values) and `dropped_constant_features`, so the provenance
says the emitted signature is not the one you typed.

This is the ordinary case, not an edge case: rocKE's attention kernels bake their
geometry in, so the kernel matcher pins 8 of their 14 fields before ranking begins and
those 8 can never vary among the candidates the model ranks.

But a CSV cannot tell that apart from the opposite situation — a column that *does*
vary in the world, sampled at one value because the corpus is thin. Dropping there
produces a model that cannot generalise, and the fix is a wider corpus, not a smaller
signature. So:

- pass **`--keep-constant-features`** when you know the corpus is thin. Every requested
  column stays in the signature, so its hash matches the richer corpus you will retrain
  on, and the run says which columns it kept and why they inform nothing today;
- when **two thirds or more** of the requested columns are constant, `train` warns that
  the proportion looks like a thin corpus and points at the input file. The threshold
  sits above the 8-of-14 rocKE shape (57%) on purpose: a warning that fires on every
  normal run is one people learn to ignore;
- when **every** requested column is constant, `train` fails and names each column with
  its value. `--keep-constant-features` does not override this — it changes the
  signature, not the fact that nothing varies. A model over zero varying features scores
  every candidate identically, and shipping one is worse than shipping none: the engine
  ranks by a model that cannot discriminate instead of falling back to its declared
  order.

### `train` arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--input` | Yes | Path to benchmark CSV/JSON |
| `--features` | Yes | Namespace-qualified feature column names (space-separated) |
| `--target` | No | Target column name (default: `tflops`) |
| `--objective` | No | `max` or `min` (default: `max`). Pass `min` for a cost target such as `latency_ms`, or the runtime will prefer the *worst* kernel. |
| `--score-units` | No | Units the score is expressed in (default: the `--target` column name) |
| `--calibrated` | No | Declare the score cross-engine comparable (RFC 0019 §12.3). Off by default; nothing here verifies the claim. |
| `--group-by` | No | Columns for GroupKFold CV |
| `--output-dir` | Yes | Output directory |
| `--name` | No | UHD display name |
| `--descriptor-name` | No | Stem for the emitted descriptor (default: `heuristic`), producing `<stem>.uhd.json` |
| `--uhd-id` | No | Reuse this UUID as the descriptor's id instead of minting a fresh one |
| `--num-boost-round` | No | Max boosting rounds (default: 500) |
| `--early-stopping` | No | Early stopping patience (default: 50) |
| `--keep-lgbm` | No | Keep intermediate .lgbm file |
| `--keep-constant-features` | No | Keep feature columns that never vary in the input (default: drop them from the signature). For a thin corpus, so the signature matches the richer one you will retrain on. |
| `--training-arches` | No | Architectures the model was trained on, for §9.2 OOD detection |
| `--model-version` | No | Semantic version embedded in the model metadata |

`--uhd-id` makes retraining a no-edit operation: pass the id the engine's UED already
names and the pair is simply overwritten in place. A value that is not a UUID is
rejected before training starts — a typo'd id becomes the descriptor's *identity*, so
the UED would point at an id nothing defines and the engine would load with no
heuristic and no error.

### `promote` arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--model-dir` | Yes | The `train --output-dir` result: one `<stem>.uhd.json` plus its artifact |
| `--descriptor-tree` | Yes | Tree holding the engine's `<name>.ued.json`; searched recursively |
| `--engine` | If ambiguous | The UED's `name` (e.g. `hipkernel:pointwise_model`) |
| `--dry-run` | No | Print the plan, write nothing |

`promote` copies the descriptor and its artifact into the UED's own directory and sets
that UED's `heuristic` to the new UHD's id, rewriting only that one line.

It validates everything before writing anything, and refuses rather than half-succeed:

- the descriptor's `id` must parse as a UUID, and its `tree_data.artifact` must exist
  next to it — installing a descriptor without its model makes the loader drop the
  engine outright, which is worse than the stale model it replaced;
- with more than one UED in the tree, `--engine` is **required**. Promoting into the
  wrong engine fails twice over: the engine you retrained keeps its old model, and one
  you never touched starts ranking with a model trained for a different kernel set.
  Both load cleanly and report nothing, so this is never guessed;
- overwriting a *different* UHD that happens to share the destination filename warns
  loudly, and is refused outright when another UED still names the id being replaced.
  Same for an artifact another installed descriptor points at — give the model a
  distinct `--descriptor-name` instead.

## Input Format

The input file should be a CSV or JSON with:
- Feature columns (problem dimensions, kernel config, device properties)
- Target column (typically TFLOPS or time)

Example CSV:
```csv
M,N,K,tile_m,tile_n,tile_k,cu_count,tflops
1024,1024,1024,128,128,32,120,50.5
2048,2048,2048,256,128,32,120,75.2
...
```

### Derived Features

The tool trains on raw columns from the input. If you need derived features
(log2, arithmetic intensity, tile efficiency), pre-compute them in your input:

```python
df["log2_M"] = np.log2(df["M"])
df["arith_intensity"] = 2 * df["M"] * df["N"] * df["K"] / (
    df["bytes_per_elem"] * (df["M"]*df["K"] + df["K"]*df["N"] + df["M"]*df["N"])
)
```

## Output

`train` generates:

```
output_dir/
├── <stem>.uhd.json     # the UHD: features, objective, score units, artifact
├── model.bin           # FlatBuffer GbdtModel for TreeDataAdapter
└── train_manifest.json # training provenance
```

`<stem>` comes from `--descriptor-name` (default `heuristic`).

`DescriptorLoader` globs `<stem>.uhd.json` and reads `tree_data.artifact` as a
path relative to that file, so the directory relocates as a unit.

The engine's UED has to name the heuristic by id. `train` prints the id it
generated and records it in `train_manifest.json`; `promote` writes it into the UED
for you, and installs the pair next to it:

```
descriptor_tree/
├── <engine>.ued.json   # "heuristic": "<the new UHD's id>"   <- the one line promote edits
├── <stem>.uhd.json     # copied from output_dir/
└── model.bin           # copied from output_dir/
```

## Generated FlatBuffers bindings

`model.bin` is written through flatc-generated Python bindings that
live in `_generated/`, committed alongside the tool the same way the C++
`*_generated.h` headers are committed alongside the SDK.

```
_generated/hipdnn_flatbuffers_sdk/data_objects/
└── GbdtModel.py, GbdtTree.py                                        # gbdt_model.fbs
```

`uhd_gen/__init__.py` prepends that directory to `sys.path`, so `import
hipdnn_flatbuffers_sdk.data_objects.GbdtModel` resolves to the bindings that
match the schema shipping beside this tool rather than to any other copy
installed on the system.

Regenerate after editing the schema — a build with `HIPDNN_GENERATE_SDK_HEADERS=ON`
does it automatically, and so does the `flatc-hipdnn` pre-commit hook:

```bash
python projects/hipdnn/scripts/run_flatc.py \
    projects/hipdnn/flatbuffers_sdk/schemas/gbdt_model.fbs
```

That command emits both the C++ header and these bindings from one invocation, so
the two cannot drift. Requires flatc 25.9.23 on PATH (see
`projects/hipdnn/CONTRIBUTING.md`).

**Do not hand-write FlatBuffers vtables here.** The writer this tool replaced did,
declaring `StartObject(11)` against a 13-field table: every field from
`features_signature` on landed one slot low, so every descriptor it produced failed
verification. Nothing caught it, because the only structural assertion in the test
suite was the four-byte file identifier — which sits before the root table and
survives any vtable error. The descriptor is JSON now; `model.bin` is the only
FlatBuffer this tool writes, and it goes through generated bindings.

### `<stem>.uhd.json`

```json
{
  "version": "1.0",
  "id": "...",
  "name": "GEMM UHD",
  "adapter": "tree_data",
  "features_signature": ["$q.M", "$q.N", "$q.K", "$kernel.tile_m", ...],
  "features_hash": "sha256:...",
  "objective": "max",
  "score": {"units": "tflops", "calibrated": false, "transform": "log1p"},
  "tree_data": {"artifact": "model.bin"}
}
```

## Training Details

- **Target transform**: `log1p(tflops)` for scale-invariant training
- **Cross-validation**: GroupKFold when `--group-by` specified
- **Early stopping**: Prevents overfitting
- **Model format**: LightGBM → FlatBuffer GbdtModel

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Integration

The output files are loaded by hipDNN's UHD system:

`DescriptorLoader` parses `<stem>.uhd.json` while walking a descriptor tree, and
`makeKernelHeuristic` builds the scorer from it: `TreeDataAdapter` loads
`tree_data.artifact` relative to the descriptor, and `FeatureExtractor`
recomputes `features_hash` from `features_signature` and refuses the pair if the
two disagree.
