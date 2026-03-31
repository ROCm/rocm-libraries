# Grouped Convolution ML Heuristics - Data Collection Plan

## Overview

This plan details how to collect benchmark data for training ML heuristics that automatically select optimal kernels for grouped convolution operations. We'll train **3 separate models** (forward, bwd_data, bwd_weight) following the dispatcher/heuristics framework.

## Data Collection Strategy

### 1. Problem Space Coverage - **COMPLETED**

✅ **Problem sets generated from ALL_CONFIGS_FULL.txt** (125,325 MIOpenDriver configs):
- Filtered for dispatcher compatibility (2D, reasonable sizes)
- Deduplicated: 23,180 unique shapes
- Sampled for diversity: **200 forward + 150 bwd_data + 150 bwd_weight**

Generated files in `problems/`:
- `forward_training.py` - 200 forward problems
- `bwd_data_training.py` - 150 backward data problems
- `bwd_weight_training.py` - 150 backward weight problems

#### **Forward Convolution Problems** (200 shapes)
Diversity coverage achieved:

**Degenerate cases** (10-15 problems):
- Tiny spatial: H,W ∈ {1, 2, 3}
- Tiny channels: C,K ∈ {1, 3, 8, 16}
- Large groups: G = C (depthwise)
- Small filters: Y,X = 1 (pointwise)

**Small convolutions** (40-50 problems):
- Batch: N ∈ {1, 2, 4, 8}
- Channels: C,K ∈ {32, 64, 128, 256}
- Spatial: H,W ∈ {7, 14, 28, 56}
- Groups: G ∈ {1, 2, 4, 8, C}
- Filters: Y,X ∈ {1, 3, 5, 7}
- Strides: S ∈ {1, 2}
- Padding: Valid + same
- Dilation: D ∈ {1, 2}

**Medium convolutions** (50-70 problems):
- ResNet-like: stages 2-5 (C:64→2048, H,W:56→7)
- MobileNet-like: depthwise (G=C) + pointwise (1x1)
- VGG-like: repeated 3x3, increasing channels
- Batch sizes: {1, 8, 16, 32, 64}

**Large convolutions** (40-50 problems):
- Batch: N ∈ {64, 128, 256}
- Channels: C,K ∈ {256, 512, 1024, 2048}
- Spatial: H,W ∈ {14, 28, 56, 112, 224}
- Depthwise separable variants

#### **Backward Data Problems** (100-120 shapes)
Similar diversity but focused on:
- Gradient flow patterns (reverse of forward)
- Larger batch sizes (training scenarios)
- Common architectures during backprop

#### **Backward Weight Problems** (100-120 shapes)
Weight gradient accumulation patterns:
- Batch size impact (critical for weight gradients)
- Channel ratio C:K variations
- Filter size importance

### 2. Kernel Configuration Coverage

For each problem, we'll benchmark **all kernels** from JSON configs:

**Forward**: 20 kernels (bf16/fp16 × compv3/compv4 × 5 tiles)
**Backward Data**: 20 kernels (bf16/fp16 × compv3/mem × 5 tiles)
**Backward Weight**: 20 kernels (bf16/fp16 × compv3/mem × 5 tiles)

**Total measurements per variant**: ~150 problems × 20 kernels = **3,000 measurements**
**Total across all 3 variants**: **~9,000 measurements**

### 3. Data Type Coverage

- **fp16** (primary)
- **bf16** (primary)
- **fp32** (optional, for receipt0 sweep)

## Execution Workflow

### Phase 1: Problem Definition

Create problem definition files:

```bash
# tile_engine/ops/grouped_conv/problems/
├── forward_training_set.py      # 150-200 forward problems
├── bwd_data_training_set.py     # 100-120 bwd_data problems
└── bwd_weight_training_set.py   # 100-120 bwd_weight problems
```

Each file defines:
```python
TRAINING_PROBLEMS = [
    TestProblem(
        name="resnet_stage2_bottleneck",
        N=64, C=64, K=256, G=1,
        Hi=56, Wi=56, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
        dtype="fp16",
    ),
    # ... 150-200 more
]
```

### Phase 2: Benchmark Execution

Use existing `grouped_conv_full_benchmark.py` with modifications:

```bash
# Forward convolution sweep
python grouped_conv_full_benchmark.py \
  --variant forward \
  --problems-file problems/forward_training_set.py \
  --config configs/forward.json \
  --workers 64 \
  --csv data/forward_raw.csv \
  --json data/forward_raw.json

# Backward data sweep
python grouped_conv_full_benchmark.py \
  --variant bwd_data \
  --problems-file problems/bwd_data_training_set.py \
  --config configs/bwd_data.json \
  --workers 64 \
  --csv data/bwd_data_raw.csv \
  --json data/bwd_data_raw.json

# Backward weight sweep
python grouped_conv_full_benchmark.py \
  --variant bwd_weight \
  --problems-file problems/bwd_weight_training_set.py \
  --config configs/bwd_weight.json \
  --workers 64 \
  --csv data/bwd_weight_raw.csv \
  --json data/bwd_weight_raw.json
```

**Estimated runtime**:
- Single measurement: ~50-100ms (JIT compile amortized)
- 3,000 measurements per variant: ~5-10 minutes
- Total for all 3 variants: **~30 minutes** on gfx950

### Phase 3: Data Format Conversion

Convert CSV/JSON to canonical Parquet format for ML training.

#### Required Schema (Parquet)

Following `dispatcher/heuristics/DATA_GENERATION.md` schema, adapted for grouped_conv:

```python
REQUIRED_COLUMNS = [
    # Problem identification
    "op_type",          # str: "forward", "bwd_data", "bwd_weight"
    "dtype",            # str: "fp16", "bf16", "fp32"
    "layout",           # str: "nhwgc"
    "arch",             # str: "gfx950"

    # Problem dimensions (M, N, K analogs for convolution)
    # For forward: M=N*Ho*Wo, N=K, K=C*Y*X
    "m",                # int: effective M dimension
    "n",                # int: effective N dimension
    "k",                # int: effective K dimension

    # Convolution-specific problem dimensions
    "batch_size",       # int: N
    "in_channels",      # int: C
    "out_channels",     # int: K
    "groups",           # int: G
    "input_h",          # int: Hi
    "input_w",          # int: Wi
    "output_h",         # int: Ho
    "output_w",         # int: Wo
    "filter_h",         # int: Y
    "filter_x",         # int: X
    "stride_h",         # int
    "stride_w",         # int
    "dilation_h",       # int
    "dilation_w",       # int
    "pad_h",            # int
    "pad_w",            # int

    # Kernel configuration
    "kernel_name",      # str: unique kernel identifier
    "tile_m",           # int
    "tile_n",           # int
    "tile_k",           # int
    "wave_m",           # int
    "wave_n",           # int
    "wave_k",           # int
    "warp_tile_m",      # int
    "warp_tile_n",      # int
    "warp_tile_k",      # int
    "pipeline",         # str: "compv3", "compv4", "mem"
    "scheduler",        # str: "intrawave", "interwave"
    "epilogue",         # str: "cshuffle"
    "vector_size_a",    # int
    "vector_size_b",    # int
    "vector_size_c",    # int

    # Performance metrics (target variable)
    "measured_tflops",  # float: actual hardware measurement
    "latency_ms",       # float: HIP event timing
    "bandwidth_gb_s",   # float: memory bandwidth utilization
]
```

#### Conversion Script

Create `convert_csv_to_parquet.py`:

```python
#!/usr/bin/env python3
"""Convert grouped_conv CSV benchmarks to canonical Parquet format."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def compute_effective_mnk(row):
    """Compute M, N, K dimensions for convolution."""
    # Forward: M=N*Ho*Wo, N=K, K=C*Y*X
    m = row['N'] * row['Ho'] * row['Wo']
    n = row['K']
    k = row['C'] * row['Y'] * row['X']
    return m, n, k

def convert_csv_to_parquet(csv_path, parquet_path, variant):
    df = pd.read_csv(csv_path)

    # Compute effective M, N, K
    df[['m', 'n', 'k']] = df.apply(
        lambda row: pd.Series(compute_effective_mnk(row)),
        axis=1
    )

    # Add op_type
    df['op_type'] = variant

    # Rename columns to canonical schema
    df = df.rename(columns={
        'N': 'batch_size',
        'C': 'in_channels',
        'K': 'out_channels',
        'G': 'groups',
        'Hi': 'input_h',
        'Wi': 'input_w',
        'Ho': 'output_h',
        'Wo': 'output_w',
        'Y': 'filter_h',
        'X': 'filter_w',
        'tflops': 'measured_tflops',
    })

    # Write Parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path, compression='snappy')

if __name__ == "__main__":
    convert_csv_to_parquet("data/forward_raw.csv", "data/forward.parquet", "forward")
    convert_csv_to_parquet("data/bwd_data_raw.csv", "data/bwd_data.parquet", "bwd_data")
    convert_csv_to_parquet("data/bwd_weight_raw.csv", "data/bwd_weight.parquet", "bwd_weight")
```

### Phase 4: Feature Engineering

Create `GroupedConvFeatureEngine` subclass in `dispatcher/heuristics/`:

```python
class GroupedConvFeatureEngine(FeatureEngine):
    """Feature extraction for grouped convolution ML heuristics."""

    def extract_features(self, row: pd.Series) -> np.ndarray:
        """Extract 55+ features from problem + kernel config."""

        # Base FeatureEngine: 55 features (problem/kernel/interaction/hardware)
        base_features = super().extract_features(row)

        # Grouped conv-specific features (additional 10-15):
        conv_features = [
            # Spatial features
            row['input_h'] * row['input_w'],              # input spatial size
            row['output_h'] * row['output_w'],            # output spatial size
            row['filter_h'] * row['filter_w'],            # filter spatial size

            # Group structure
            row['groups'],                                # number of groups
            row['in_channels'] / row['groups'],           # channels per group
            row['out_channels'] / row['groups'],          # output channels per group
            int(row['groups'] == row['in_channels']),     # is_depthwise flag

            # Stride/dilation impact
            row['stride_h'] * row['stride_w'],            # stride factor
            row['dilation_h'] * row['dilation_w'],        # dilation factor

            # Channel ratios
            row['out_channels'] / max(row['in_channels'], 1),

            # Padding impact
            row['pad_h'] + row['pad_w'],

            # Pointwise flag
            int(row['filter_h'] == 1 and row['filter_w'] == 1),
        ]

        return np.concatenate([base_features, conv_features])
```

## Training Workflow

Following `dispatcher/heuristics/train.py`:

### 1. Data Preparation

```bash
# Combine all 3 variants into single training set (optional)
# OR train 3 separate models (recommended for specialization)

python -m dispatcher.heuristics.train \
  --data data/forward.parquet \
  --output models/grouped_conv_forward.lgbm.gz \
  --variant forward

python -m dispatcher.heuristics.train \
  --data data/bwd_data.parquet \
  --output models/grouped_conv_bwd_data.lgbm.gz \
  --variant bwd_data

python -m dispatcher.heuristics.train \
  --data data/bwd_weight.parquet \
  --output models/grouped_conv_bwd_weight.lgbm.gz \
  --variant bwd_weight
```

### 2. Training Configuration

LightGBM hyperparameters (from LEARNINGS.md):
- **num_leaves**: 31
- **n_estimators**: 2000 trees
- **max_depth**: -1 (unlimited)
- **learning_rate**: 0.05
- **boosting_type**: gbdt
- **objective**: regression
- **metric**: rmse
- **min_child_samples**: 20

**Critical**: Use `log1p(measured_tflops)` as target (essential for scale normalization)

### 3. Cross-Validation Strategy

**GroupKFold with (M, N, K) grouping** to prevent data leakage:
- Same effective shape appears in same fold
- Tests generalization to unseen shapes
- 5-fold CV recommended

### 4. Evaluation Metrics

From `dispatcher/heuristics/evaluate.py`:
- **Top-1 Efficiency**: `measured_tflops / best_tflops` (target: >95%)
- **Top-3 Efficiency**: Model picks top-3, best of 3 used
- **RMSE**: log-space error
- **Per-shape analysis**: Identify failure modes

## Expected Outcomes

### Training Data Size
- **Forward**: ~3,000 measurements (150 problems × 20 kernels)
- **Backward Data**: ~2,400 measurements (120 problems × 20 kernels)
- **Backward Weight**: ~2,400 measurements (120 problems × 20 kernels)
- **Total**: ~7,800 measurements

### Model Performance Targets (from FMHA learnings)
- **Top-1 Efficiency**: 97%+ (following FMHA: 97.51%)
- **Top-3 Efficiency**: 99%+
- **Inference Time**: <1ms per prediction
- **Model Size**: <500KB compressed (.lgbm.gz)

### Failure Mode Analysis
Expect challenges with:
- **Tiny spatial sizes**: H,W ∈ {1,2,3} (harder to predict)
- **Extreme group counts**: G = C (depthwise) vs G = 1
- **Large batch + small spatial**: Memory-bound vs compute-bound transition

## Timeline Estimate

1. **Problem definition**: 4-6 hours (write 350-400 test problems)
2. **Benchmark execution**: 30-60 minutes (hardware time on gfx950)
3. **Data conversion**: 30 minutes (CSV → Parquet, schema validation)
4. **Feature engineering**: 2-3 hours (GroupedConvFeatureEngine implementation)
5. **Model training**: 1-2 hours (3 models with CV)
6. **Evaluation**: 1-2 hours (metrics, failure analysis)

**Total**: 1-2 days for complete ML heuristics pipeline

## Integration with Dispatcher

Final models will be loaded by dispatcher heuristics system:

```python
# dispatcher/heuristics/grouped_conv_heuristic.py
class GroupedConvHeuristic:
    def __init__(self):
        self.forward_model = load_compressed_model("models/grouped_conv_forward.lgbm.gz")
        self.bwd_data_model = load_compressed_model("models/grouped_conv_bwd_data.lgbm.gz")
        self.bwd_weight_model = load_compressed_model("models/grouped_conv_bwd_weight.lgbm.gz")
        self.feature_engine = GroupedConvFeatureEngine()

    def select_kernel(self, problem, variant):
        model = {
            "forward": self.forward_model,
            "bwd_data": self.bwd_data_model,
            "bwd_weight": self.bwd_weight_model,
        }[variant]

        features = self.feature_engine.extract_features(problem)
        predicted_tflops = model.predict([features])[0]
        return best_kernel
```

## Next Steps

1. **Create problem definition files** (forward_training_set.py, etc.)
2. **Modify grouped_conv_full_benchmark.py** to accept --problems-file
3. **Run benchmark sweeps** (3 variants × 30 minutes)
4. **Implement conversion script** (CSV → Parquet)
5. **Subclass FeatureEngine** for conv-specific features
6. **Train 3 models** using dispatcher/heuristics/train.py
7. **Evaluate** using dispatcher/heuristics/evaluate.py
8. **Integrate** into dispatcher runtime selection
