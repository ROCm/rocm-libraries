# UHD Generation Tool

Train and export heuristic models for hipDNN's Universal Heuristic Descriptor (UHD) system.

## Overview

This tool takes benchmark timing data and produces:
1. A trained LightGBM model
2. A FlatBuffer model artifact (`model.bin`) for `TreeDataAdapter`
3. A UHD descriptor (`<stem>.uhd.json`) for `DescriptorLoader` (RFC 0019 §4)
4. A human-readable UHD JSON (`uhd.json`)

## Installation

```bash
cd projects/hipdnn/tools/uhd_gen
pip install -e .
```

## Usage

```bash
# Input CSV must have feature columns and a target column (default: tflops)
python -m uhd_gen \
    --input benchmark_results.csv \
    --features q.M q.N q.K kernel.tile_m kernel.tile_n kernel.tile_k device.cu_count \
    --target tflops \
    --group-by q.M q.N q.K \
    --output-dir ./uhd_output \
    --name "GEMM UHD"
```

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

### Arguments

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
| `--num-boost-round` | No | Max boosting rounds (default: 500) |
| `--early-stopping` | No | Early stopping patience (default: 50) |
| `--keep-lgbm` | No | Keep intermediate .lgbm file |

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

The tool generates:

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
generated and records it in `train_manifest.json`.

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
