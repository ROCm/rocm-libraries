# ML Heuristics Training Plan for Grouped Convolution

## Overview

Train a LightGBM regression model to predict TFLOPS for grouped convolution kernels, enabling fast kernel selection without exhaustive GPU benchmarking.

**Current Status**: ✅ Data collected (3,760 measurements)
**Next Steps**: Convert data → Train model → Evaluate → Integrate

---

## Step-by-Step Plan

### Phase 1: Data Preparation

#### Step 1.1: Convert CSV to Parquet Format

**Why**: Parquet is the canonical format for heuristics training (efficient, typed, supports chunking)

**Actions**:
```bash
cd /workspace/rocm-libraries/projects/composablekernel/dispatcher/heuristics

# Create grouped_conv data directory
mkdir -p data/grouped_conv_forward_bf16_gfx950

# Convert CSV to parquet
python3 convert_csv_to_parquet_grouped_conv.py \
    --input ../../tile_engine/ops/grouped_conv/training_data_forward_bf16_20.csv \
    --output data/grouped_conv_forward_bf16_gfx950/training_data.parquet \
    --arch gfx950 \
    --dtype bf16 \
    --variant forward
```

**What this script needs to do**:
1. Read CSV with columns: `kernel, problem_idx, N, C, K, G, Hi, Wi, Y, X, stride_h, stride_w, pad_h, pad_w, latency_ms, tflops, non_zero`
2. Parse kernel name to extract kernel features:
   - `grouped_conv_forward_bf16_2d_16x64x64_compv3` → Extract:
     - `block_size`: 16
     - `gemm_m_per_block`: 64
     - `gemm_n_per_block`: 64
     - `pipeline`: compv3
     - `ndim_spatial`: 2
3. Add metadata columns: `arch`, `dtype`, `variant`
4. Write parquet with schema matching heuristics expectations

**Schema for parquet**:
```python
{
    # Problem features
    'N': int, 'C': int, 'K': int, 'G': int,
    'Hi': int, 'Wi': int, 'Y': int, 'X': int,
    'stride_h': int, 'stride_w': int,
    'pad_h': int, 'pad_w': int,

    # Kernel features (parsed from name)
    'block_size': int,
    'gemm_m_per_block': int,
    'gemm_n_per_block': int,
    'gemm_k_per_block': int,
    'wave_m': int, 'wave_n': int, 'wave_k': int,
    'warp_tile_m': int, 'warp_tile_n': int, 'warp_tile_k': int,
    'pipeline': str,
    'scheduler': str,
    'epilogue': str,

    # Metadata
    'arch': str,  # 'gfx950'
    'dtype': str,  # 'bf16'
    'variant': str,  # 'forward'
    'ndim_spatial': int,  # 2

    # Targets
    'latency_ms': float,
    'tflops': float,
    'kernel_name': str  # Full kernel name for reference
}
```

#### Step 1.2: Validate Parquet Data

**Actions**:
```bash
# Quick inspection
python3 -c "
import pandas as pd
df = pd.read_parquet('data/grouped_conv_forward_bf16_gfx950/training_data.parquet')
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'TFLOPS range: {df.tflops.min():.2f} - {df.tflops.max():.2f}')
print(f'Unique kernels: {df.kernel_name.nunique()}')
print(f'Unique problems: {df.groupby([\"N\",\"C\",\"K\",\"Hi\",\"Wi\"]).ngroups}')
"
```

**Expected output**:
```
Rows: 3760
Columns: ['N', 'C', 'K', 'G', 'Hi', 'Wi', 'Y', 'X', 'stride_h', 'stride_w',
          'pad_h', 'pad_w', 'block_size', 'gemm_m_per_block', ..., 'tflops']
TFLOPS range: 4.19 - 166.31
Unique kernels: 20
Unique problems: ~188 (some problems may have failed for all kernels)
```

---

### Phase 2: Feature Engineering

#### Step 2.1: Create GroupedConvFeatureEngine

**File**: `dispatcher/heuristics/feature_engine_grouped_conv.py`

**Actions**: Subclass `FeatureEngine` to add grouped-conv-specific features

**Features to implement** (following GEMM pattern):

##### Problem Features (~15)
```python
- N, C, K, G  # Batch, channels, kernels, groups
- Hi, Wi, Ho, Wo  # Input/output spatial dims
- Y, X  # Filter size
- stride_h, stride_w, pad_h, pad_w
- log2(N), log2(C), log2(K)
- log2(input_volume)  # N * Hi * Wi * C
- log2(output_volume)  # N * Ho * Wo * K
- arithmetic_intensity  # FLOPs / bytes_transferred
- aspect_ratio_hw  # Hi / Wi
- channels_per_group  # C / G
```

##### Kernel Features (~20)
```python
- block_size, gemm_m_per_block, gemm_n_per_block, gemm_k_per_block
- wave_m, wave_n, wave_k
- warp_tile_m, warp_tile_n, warp_tile_k
- pipeline  # compv3, compv4 (categorical)
- scheduler  # intrawave, interwave (categorical)
- epilogue  # cshuffle, default (categorical)
- ndim_spatial  # 2 or 3
- num_warps  # wave_m * wave_n * wave_k
- tile_volume  # block_size * gemm_m * gemm_n * gemm_k
- tile_mn  # gemm_m * gemm_n
- lds_usage_estimate  # Approx LDS bytes for tiles
- lds_usage_ratio  # lds_usage / hw_lds_capacity
```

##### Interaction Features (~12)
```python
- num_tiles_spatial_h  # ceil(Ho / gemm_m)
- num_tiles_spatial_w  # ceil(Wo / gemm_n)
- num_tiles_channels  # ceil(K / gemm_k)
- total_output_tiles  # num_tiles_h * num_tiles_w * num_tiles_k
- tile_eff_h  # Ho / (num_tiles_h * gemm_m)
- tile_eff_w  # Wo / (num_tiles_w * gemm_n)
- tile_eff_k  # K / (num_tiles_k * gemm_k)
- overall_tile_efficiency  # product of efficiencies
- cu_utilization  # total_tiles / (hw_num_cus * hw_max_waves_per_cu)
- filter_area  # Y * X
- is_1x1_conv  # Y == 1 and X == 1
- is_3x3_conv  # Y == 3 and X == 3
```

##### Hardware Features (12) - Same as GEMM
```python
- hw_num_cus, hw_simds_per_cu, hw_total_simds
- hw_shader_engines, hw_max_clock_mhz
- hw_max_waves_per_cu, hw_wavefront_size
- hw_lds_capacity, hw_l1_cache_kb, hw_l2_cache_kb
- hw_l3_cache_kb, hw_num_xcd
```

**Total Features**: ~59 (vs 55 for GEMM)

**Implementation**:
```python
class GroupedConvFeatureEngine(FeatureEngine):
    """Feature extraction for grouped convolution kernels."""

    def __init__(self, arch: str = "gfx950"):
        super().__init__(arch)
        self.op = "grouped_conv"

    def extract_problem_features(self, row: dict) -> dict:
        """Extract problem-specific features for grouped conv."""
        # Calculate output dimensions
        Ho = (row['Hi'] + 2*row['pad_h'] - row['Y']) // row['stride_h'] + 1
        Wo = (row['Wi'] + 2*row['pad_w'] - row['X']) // row['stride_w'] + 1

        # FLOPs = 2 * N * Ho * Wo * K * C/G * Y * X
        flops = 2 * row['N'] * Ho * Wo * row['K'] * (row['C']//row['G']) * row['Y'] * row['X']

        # Bytes = input + weight + output (assuming BF16 = 2 bytes)
        input_bytes = row['N'] * row['Hi'] * row['Wi'] * row['C'] * 2
        weight_bytes = row['K'] * row['Y'] * row['X'] * (row['C']//row['G']) * 2
        output_bytes = row['N'] * Ho * Wo * row['K'] * 2
        total_bytes = input_bytes + weight_bytes + output_bytes

        return {
            'N': row['N'],
            'C': row['C'],
            'K': row['K'],
            'G': row['G'],
            'Hi': row['Hi'],
            'Wi': row['Wi'],
            'Ho': Ho,
            'Wo': Wo,
            'Y': row['Y'],
            'X': row['X'],
            'stride_h': row['stride_h'],
            'stride_w': row['stride_w'],
            'pad_h': row['pad_h'],
            'pad_w': row['pad_w'],
            'log2_N': np.log2(row['N'] + 1),
            'log2_C': np.log2(row['C']),
            'log2_K': np.log2(row['K']),
            'log2_input_volume': np.log2(row['N'] * row['Hi'] * row['Wi'] * row['C']),
            'log2_output_volume': np.log2(row['N'] * Ho * Wo * row['K']),
            'arithmetic_intensity': flops / total_bytes if total_bytes > 0 else 0,
            'aspect_ratio_hw': row['Hi'] / row['Wi'] if row['Wi'] > 0 else 1.0,
            'channels_per_group': row['C'] / row['G'] if row['G'] > 0 else row['C'],
        }
```

#### Step 2.2: Test Feature Extraction

**Actions**:
```bash
cd /workspace/rocm-libraries/projects/composablekernel/dispatcher/heuristics

# Test on first few rows
python3 -c "
import pandas as pd
from feature_engine_grouped_conv import GroupedConvFeatureEngine

df = pd.read_parquet('data/grouped_conv_forward_bf16_gfx950/training_data.parquet')
engine = GroupedConvFeatureEngine('gfx950')

# Extract features for first row
features = engine.extract_batch(df.head(10))
print(f'Feature matrix shape: {features.shape}')
print(f'Feature names ({len(features.columns)}):')
print(list(features.columns))
"
```

**Expected output**:
```
Feature matrix shape: (10, 59)
Feature names (59):
['N', 'C', 'K', 'G', 'Hi', 'Wi', 'Ho', 'Wo', 'Y', 'X', 'stride_h', 'stride_w', ...]
```

---

### Phase 3: Model Training

#### Step 3.1: Train TFLOPS Model

**Actions**:
```bash
cd /workspace/rocm-libraries/projects/composablekernel/dispatcher/heuristics

# Train model (5-fold cross-validation)
python3 train.py \
    --data_dir data/ \
    --out_dir models/grouped_conv_forward_bf16_gfx950 \
    --op grouped_conv \
    --dtype bf16 \
    --arch gfx950 \
    --variant forward
```

**What train.py does**:
1. Load parquet from `data/grouped_conv_forward_bf16_gfx950/training_data.parquet`
2. Extract features using `GroupedConvFeatureEngine`
3. Group by `(N, C, K, Hi, Wi)` for cross-validation (ensure no data leakage)
4. Train 3 models:
   - **TFLOPS model** (primary): `log1p(tflops)` as target
   - **Latency model**: `log1p(latency_ms)` as target
   - **Bandwidth model**: `latency_ms * problem_bytes` as target
5. Use LightGBM with default hyperparameters:
   - `n_estimators`: 1000
   - `learning_rate`: 0.05
   - `max_depth`: 8
   - `num_leaves`: 64
   - `objective`: 'regression'
   - `metric`: 'rmse'
6. Save models to `models/grouped_conv_forward_bf16_gfx950/`:
   - `model_tflops.lgbm.gz`
   - `model_latency.lgbm.gz`
   - `model_bandwidth.lgbm.gz`
   - `feature_names.json`
   - `training_metadata.json`

**Expected output**:
```
Loading data from data/grouped_conv_forward_bf16_gfx950/training_data.parquet
Found 3760 measurements
Extracting features using GroupedConvFeatureEngine...
Feature matrix shape: (3760, 59)

Training TFLOPS model (log1p target)...
  Fold 1/5: RMSE=0.052, R2=0.995
  Fold 2/5: RMSE=0.048, R2=0.996
  Fold 3/5: RMSE=0.051, R2=0.995
  Fold 4/5: RMSE=0.049, R2=0.996
  Fold 5/5: RMSE=0.050, R2=0.995
  Mean CV RMSE: 0.050 ± 0.002
  Mean CV R2: 0.995 ± 0.001

Training latency model...
Training bandwidth model...

Saving models to models/grouped_conv_forward_bf16_gfx950/
  ✓ model_tflops.lgbm.gz
  ✓ model_latency.lgbm.gz
  ✓ model_bandwidth.lgbm.gz
  ✓ feature_names.json
  ✓ training_metadata.json
```

#### Step 3.2: Feature Importance Analysis

**Actions**:
```bash
# View feature importance
python3 -c "
import lightgbm as lgb
import gzip
import json

# Decompress and load model
with gzip.open('models/grouped_conv_forward_bf16_gfx950/model_tflops.lgbm.gz', 'rb') as f:
    model = lgb.Booster(model_file=f.read())

# Get feature importance
importance = model.feature_importance(importance_type='gain')
feature_names = json.load(open('models/grouped_conv_forward_bf16_gfx950/feature_names.json'))

# Sort by importance
sorted_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)

print('Top 20 Most Important Features:')
for i, (name, score) in enumerate(sorted_features[:20]):
    print(f'{i+1:2d}. {name:30s} {score:10.0f}')
"
```

**Expected top features** (similar to GEMM):
```
Top 20 Most Important Features:
 1. log2_output_volume          450000
 2. K                           320000
 3. C                           280000
 4. arithmetic_intensity        210000
 5. tile_eff_k                  180000
 6. Ho                          160000
 7. Wo                          150000
 8. gemm_k_per_block           140000
 9. block_size                  130000
10. pipeline                    120000
...
```

---

### Phase 4: Model Evaluation

#### Step 4.1: Comprehensive Evaluation

**Actions**:
```bash
cd /workspace/rocm-libraries/projects/composablekernel/dispatcher/heuristics

# Run full evaluation
python3 evaluate.py \
    --model_dir models/grouped_conv_forward_bf16_gfx950 \
    --data_dir data/ \
    --op grouped_conv \
    --dtype bf16 \
    --arch gfx950
```

**What evaluate.py reports**:
1. **Global Metrics**:
   - Mean TFLOPS Efficiency (target: >98%)
   - P10/P50/P90 Efficiency
   - NDCG@1, NDCG@5
   - Top-5 Hit Rate
   - R² scores for all models

2. **Per-Shape Family Breakdown**:
   - Small spatial (H,W < 14)
   - Medium spatial (14 ≤ H,W < 56)
   - Large spatial (H,W ≥ 56)
   - Tiny batch (N=1-2)
   - Large batch (N≥4)

3. **Per-Pipeline Breakdown**:
   - compv3 efficiency
   - compv4 efficiency

4. **Per-Filter Size Breakdown**:
   - 1×1 convolutions
   - 3×3 convolutions
   - 2×2 convolutions

**Expected output**:
```
================================================================================
EVALUATION: grouped_conv_forward_bf16_gfx950
================================================================================

Global Metrics:
  Mean TFLOPS Efficiency: 98.5%
  P10 Efficiency: 95.2%
  P50 Efficiency: 99.1%
  Min Efficiency: 91.3%

  TFLOPS Model R²: 0.995
  Latency Model R²: 0.993

  NDCG@1: 67.2%
  Top-5 Hit Rate: 89.4%

Shape Family Breakdown:
  Small Spatial (H,W<14):    Mean Eff=97.2%, P10=93.1%, Shapes=42
  Medium Spatial (14≤H<56):  Mean Eff=98.9%, P10=96.5%, Shapes=86
  Large Spatial (H≥56):      Mean Eff=99.2%, P10=97.8%, Shapes=60

  Tiny Batch (N=1-2):        Mean Eff=98.1%, P10=94.7%, Shapes=152
  Large Batch (N≥4):         Mean Eff=99.3%, P10=97.2%, Shapes=36

Pipeline Breakdown:
  compv3: Mean Eff=98.7%, P10=95.8%
  compv4: Mean Eff=98.3%, P10=94.6%

Filter Size Breakdown:
  1×1 Conv: Mean Eff=99.1%, P10=97.3%, Shapes=72
  3×3 Conv: Mean Eff=98.2%, P10=94.1%, Shapes=98
  2×2 Conv: Mean Eff=97.9%, P10=93.5%, Shapes=18
```

#### Step 4.2: Error Analysis

**Actions**:
```bash
# Find worst-performing shapes
python3 -c "
import pandas as pd
import numpy as np
from predictor import Predictor

# Load model and data
pred = Predictor.load('models/grouped_conv_forward_bf16_gfx950')
df = pd.read_parquet('data/grouped_conv_forward_bf16_gfx950/training_data.parquet')

# Group by problem, find oracle best vs predicted best
problems = df.groupby(['N', 'C', 'K', 'Hi', 'Wi'])
worst_cases = []

for prob_key, prob_df in problems:
    oracle_best = prob_df.loc[prob_df.tflops.idxmax()]

    # Predict TFLOPS for all kernels in this problem
    predictions = pred.predict_tflops(prob_df)
    predicted_best_idx = predictions.argmax()
    predicted_best = prob_df.iloc[predicted_best_idx]

    efficiency = predicted_best.tflops / oracle_best.tflops
    if efficiency < 0.95:  # Less than 95% efficiency
        worst_cases.append({
            'N': prob_key[0], 'C': prob_key[1], 'K': prob_key[2],
            'H': prob_key[3], 'W': prob_key[4],
            'efficiency': efficiency,
            'oracle_tflops': oracle_best.tflops,
            'predicted_tflops': predicted_best.tflops,
            'oracle_kernel': oracle_best.kernel_name,
            'predicted_kernel': predicted_best.kernel_name,
        })

worst_df = pd.DataFrame(worst_cases).sort_values('efficiency')
print(f'Shapes with <95% efficiency: {len(worst_df)}')
print(worst_df.head(10))
"
```

---

### Phase 5: Integration & Deployment

#### Step 5.1: Python API Integration

**Create predictor wrapper**:
```bash
cd /workspace/rocm-libraries/projects/composablekernel/tile_engine/ops/grouped_conv

# Create prediction script
cat > predict_best_kernel.py << 'EOF'
#!/usr/bin/env python3
"""Predict best kernel for a grouped conv problem using ML heuristics."""

import sys
from pathlib import Path

# Add heuristics to path
_HEURISTICS = Path(__file__).parents[3] / "dispatcher" / "heuristics"
sys.path.insert(0, str(_HEURISTICS))

from predictor import Predictor

def predict_best_kernel(N, C, K, G, Hi, Wi, Y, X, stride_h, stride_w, pad_h, pad_w):
    """Predict best kernel for given problem."""

    # Load model
    model_dir = _HEURISTICS / "models" / "grouped_conv_forward_bf16_gfx950"
    predictor = Predictor.load(str(model_dir))

    # Create problem dict
    problem = {
        'N': N, 'C': C, 'K': K, 'G': G,
        'Hi': Hi, 'Wi': Wi,
        'Y': Y, 'X': X,
        'stride_h': stride_h, 'stride_w': stride_w,
        'pad_h': pad_h, 'pad_w': pad_w,
    }

    # Rank all 20 kernels
    ranked_kernels = predictor.rank_kernels(problem, top_k=5)

    return ranked_kernels

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--C", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--G", type=int, default=1)
    parser.add_argument("--H", type=int, required=True)
    parser.add_argument("--W", type=int, required=True)
    parser.add_argument("--Y", type=int, default=3)
    parser.add_argument("--X", type=int, default=3)
    parser.add_argument("--stride_h", type=int, default=1)
    parser.add_argument("--stride_w", type=int, default=1)
    parser.add_argument("--pad_h", type=int, default=1)
    parser.add_argument("--pad_w", type=int, default=1)
    args = parser.parse_args()

    results = predict_best_kernel(
        args.N, args.C, args.K, args.G, args.H, args.W,
        args.Y, args.X, args.stride_h, args.stride_w, args.pad_h, args.pad_w
    )

    print(f"Top-5 Kernels for N={args.N} C={args.C} K={args.K} H={args.H} W={args.W}:")
    for i, (kernel, tflops) in enumerate(results):
        print(f"  {i+1}. {kernel:50s} {tflops:8.2f} TFLOPS")
EOF

chmod +x predict_best_kernel.py
```

**Test prediction**:
```bash
python3 predict_best_kernel.py --N 1 --C 128 --K 256 --H 32 --W 32 --Y 3 --X 3
```

**Expected output**:
```
Top-5 Kernels for N=1 C=128 K=256 H=32 W=32:
  1. grouped_conv_forward_bf16_2d_16x64x64_compv4         9.86 TFLOPS
  2. grouped_conv_forward_bf16_2d_16x64x64_compv3         8.45 TFLOPS
  3. grouped_conv_forward_bf16_2d_32x64x64_compv4         7.98 TFLOPS
  4. grouped_conv_forward_bf16_2d_32x64x64_compv3         6.99 TFLOPS
  5. grouped_conv_forward_bf16_2d_64x64x64_compv3         6.51 TFLOPS
```

#### Step 5.2: C++ Integration (Optional)

Follow heuristics README Section 6 for C++ integration using LightGBM C API.

---

## File Checklist

### Scripts to Create

- [ ] `convert_csv_to_parquet_grouped_conv.py` - CSV → Parquet converter
- [ ] `feature_engine_grouped_conv.py` - Feature extraction
- [ ] Modify `train.py` to support `--op grouped_conv`
- [ ] Modify `evaluate.py` to support `--op grouped_conv`
- [ ] Modify `predictor.py` to support `grouped_conv`
- [ ] `predict_best_kernel.py` - Standalone prediction script

### Data Files

- [ ] `data/grouped_conv_forward_bf16_gfx950/training_data.parquet`
- [ ] `models/grouped_conv_forward_bf16_gfx950/model_tflops.lgbm.gz`
- [ ] `models/grouped_conv_forward_bf16_gfx950/model_latency.lgbm.gz`
- [ ] `models/grouped_conv_forward_bf16_gfx950/model_bandwidth.lgbm.gz`
- [ ] `models/grouped_conv_forward_bf16_gfx950/feature_names.json`
- [ ] `models/grouped_conv_forward_bf16_gfx950/training_metadata.json`

---

## Success Criteria

- [  ] Data conversion: 3,760 rows → parquet with 59 features
- [ ] Training: Mean CV R² > 0.99 for TFLOPS model
- [ ] Evaluation: Mean TFLOPS Efficiency > 98%
- [ ] Evaluation: P10 Efficiency > 94%
- [ ] Prediction: <1ms inference time per problem
- [ ] Integration: Python API working end-to-end

---

## Next Actions

1. **Create `convert_csv_to_parquet_grouped_conv.py`** (parse kernel names, extract features)
2. **Create `feature_engine_grouped_conv.py`** (59 features total)
3. **Test feature extraction** on sample data
4. **Run training** with existing `train.py` (may need minor modifications)
5. **Evaluate model** with existing `evaluate.py`
6. **Create prediction API** for easy integration

---

## Timeline Estimate

- Phase 1 (Data Prep): 2-3 hours
- Phase 2 (Feature Engineering): 3-4 hours
- Phase 3 (Training): 5 minutes (GPU) + validation
- Phase 4 (Evaluation): 1 hour
- Phase 5 (Integration): 2 hours

**Total**: 1-2 days for complete pipeline

---

## References

- Heuristics README: `/workspace/rocm-libraries/projects/composablekernel/dispatcher/heuristics/README.md`
- GEMM Feature Engine: `/workspace/rocm-libraries/projects/composablekernel/dispatcher/heuristics/feature_engine.py`
- Training Script: `/workspace/rocm-libraries/projects/composablekernel/dispatcher/heuristics/train.py`
- Data collected: `/workspace/rocm-libraries/projects/composablekernel/tile_engine/ops/grouped_conv/training_data_forward_bf16_20.csv`
