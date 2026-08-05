# UHD Generation Tool

Train and export heuristic models for hipDNN's Universal Heuristic Descriptor (UHD) system.

## Overview

This tool takes benchmark timing data and produces:
1. A trained LightGBM model
2. A FlatBuffer model artifact (`model.bin`) for `TreeDataAdapter`
3. A UHD descriptor JSON (`uhd.json`)

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
    --features M N K tile_m tile_n tile_k cu_count \
    --target tflops \
    --group-by M N K \
    --output-dir ./uhd_output \
    --name "GEMM UHD"
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--input` | Yes | Path to benchmark CSV/JSON |
| `--features` | Yes | Feature column names (space-separated) |
| `--target` | No | Target column name (default: `tflops`) |
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
├── model.bin          # FlatBuffer GbdtModel for TreeDataAdapter
├── uhd.json           # UHD descriptor
└── train_manifest.json # Training metadata
```

### uhd.json

```json
{
  "schema": "hipdnn.uhd/v1",
  "id": "...",
  "name": "GEMM UHD",
  "adapter": "tree_data",
  "features_signature": ["$M", "$N", "$K", ...],
  "features_hash": "sha256:...",
  "objective": "max",
  "score": {"units": "tflops", "calibrated": true, "transform": "log1p"},
  "model": {"artifact": "model.bin"}
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

The output `model.bin` is loaded by `TreeDataAdapter` in hipDNN backend:

```cpp
auto adapter = TreeDataAdapter::load("model.bin", expectedFeaturesHash);
double score = adapter->score(featureVector);
```
