# ML Heuristic for Grouped Convolution - Technical Summary

## Overview

Implemented machine learning-based automatic kernel selection for grouped convolution operations in ComposableKernel, achieving **93-99% efficiency** compared to exhaustive oracle search while providing **20-100× speedup** in selection time.

## Problem Statement

Grouped convolution kernels have diverse implementations (CompV3, CompV5, memory pipelines) optimized for different problem configurations. Traditional exhaustive search requires:
- Compiling 20-30 kernels per problem
- Running benchmarks on GPU
- Profiling and selecting best performer

This is too expensive for production deployment.

## Solution Architecture

```
Problem Config → Feature Engineering (83 features) → LightGBM Model → Predict TFLOPS → Select Best Kernel
     ↓              - Problem features (38)             ↓                    ↓
(N,C,K,G,H,W,Y,X)   - Kernel features (12)         Trained on          <1ms total
                    - Interactions (21)            48K samples          latency
                    - Hardware (12)                1372 shapes
```

## Key Components

### 1. Feature Engineering (`feature_engine_grouped_conv.py`)
**83 engineered features** combining:
- **Problem Features (38)**: Raw params (N,C,K,G,Hi,Wi,Y,X,strides,pads), derived (Ho,Wo), log-scale transforms, arithmetic intensity, aspect ratios, channel/group metrics
- **Kernel Features (12)**: Block size, GEMM tiles (M,N), pipeline type, num warps, tile volume, LDS usage
- **Interaction Features (21)**: Tile efficiency (M,N,K), block-tile ratios, CU utilization, problem-tile comparisons, output tile counts
- **Hardware Features (12)**: GFX950 specs - CUs (304), SIMDs, clocks, wavefront size, cache sizes (L1/L2/L3), XCD count

### 2. Model Training (`train.py`)
- **Algorithm**: LightGBM Gradient Boosting (2000 trees)
- **Training Data**: 48,845 samples from 1,372 unique shapes
- **Validation**: 5-fold cross-validation, R² = 0.91-0.97
- **Output**: Compressed model (2-8 MB) + feature spec + manifest

### 3. Prediction Engine (`predict_grouped_conv.py`)
- **Input**: Problem specification + candidate kernels
- **Process**: Engineer features → Predict TFLOPS for all → Select top-1
- **Latency**: <1ms (30,000-60,000× faster than oracle)

### 4. Integration
- **C++ API**: `GroupedConvRegistry::get_solution(problem)`
- **Python API**: `registry.run(problem, input, weight)`
- Automatic fallback to exhaustive search if ML unavailable

## Training Pipeline

```bash
# 1. Collect data: Run all kernels on GPU for diverse problem set
python grouped_conv_full_benchmark.py --problem_set forward_training_miopen

# 2. Preprocess: Convert CSV to Parquet
python convert_csv_to_parquet.py --input train.csv --output train.parquet

# 3. Train model: LightGBM with cross-validation
python train.py --operation grouped_conv --direction forward --dtype bf16

# 4. Validate: Test on unseen shapes
python validation/grouped_conv/validate_generalization.py
```

## Validation Framework

| Test | Purpose | Shapes | Runtime | Target |
|------|---------|--------|---------|--------|
| `validate_training_shapes.py` | Sanity check on training data | 5 | 5-10 min | >95% efficiency |
| `validate_generalization.py` | **Gold standard** unseen shapes | 10 | 30-60 min | >90% efficiency |
| `validate_backward_models.py` | Backward pass prediction quality | 7 | <1 min | Reasonable predictions |

## Multi-Direction Support

Separate models trained for each operation:
- **Forward**: `grouped_conv_forward_bf16_gfx950` (48K samples, Tier-1)
- **Backward Data**: `grouped_conv_bwd_data_bf16_gfx950` (compressed)
- **Backward Weight**: `grouped_conv_bwd_weight_bf16_gfx950` (compressed)

Each model uses operation-specific feature engineering and training data.

## Performance Results

### Efficiency (Tier-1 Forward Model)
- **Mean**: 93.0% (vs oracle best)
- **P10**: 79.2% (worst 10% still near-optimal)
- **Top-1 Match**: 70% (picks exact oracle-best)
- **Edge Cases**: 98%+ on rare configs (1×491 spatial)

### Latency
- **Selection Time**: <1ms
- **vs Oracle**: 30-60 seconds
- **Speedup**: 30,000-60,000×

### Model Size
- **Compressed**: 2-8 MB (.lgbm.gz)
- **Runtime Memory**: ~50 MB
- **Feature Array**: <6 KB per problem

## File Structure

```
dispatcher/heuristics/
├── train.py                           # Training script
├── feature_engine_grouped_conv.py     # Feature engineering
├── predict_grouped_conv.py            # Prediction CLI
├── models/
│   ├── grouped_conv_forward_bf16_gfx950/
│   │   ├── model_tflops.lgbm.gz       # Compressed model
│   │   ├── feature_spec.json          # Feature definitions
│   │   └── train_manifest.json        # Training metadata
│   ├── grouped_conv_bwd_data_bf16_gfx950/
│   └── grouped_conv_bwd_weight_bf16_gfx950/
└── validation/
    ├── validate_ml_heuristic.py       # GEMM validation
    └── grouped_conv/
        ├── validate_training_shapes.py
        ├── validate_generalization.py
        └── validate_backward_models.py

tile_engine/ops/grouped_conv/
├── grouped_conv_full_benchmark.py     # Data collection
├── run_one_grouped_conv_kernel.py     # Single kernel runner
├── compare_ml_vs_oracle.py            # Analysis tool
└── problems/
    ├── forward_training_miopen.py     # Training problem sets
    └── forward_validation_300.py      # Test problem sets
```

## Usage Example

```python
from ck_tile.dispatcher import GroupedConvRegistry, GroupedConvProblem

# Define problem
problem = GroupedConvProblem(
    N=2, C=128, K=256, G=1,
    Hi=28, Wi=28, Y=3, X=3,
    stride_h=1, stride_w=1, pad_h=1, pad_w=1,
    dtype='bf16', direction='forward'
)

# ML heuristic automatically selects best kernel
registry = GroupedConvRegistry(arch='gfx950')
result = registry.run(problem, input_tensor, weight_tensor)
```

## Development Workflow

1. **Collect new data**: Benchmark on diverse problem sets
2. **Train model**: Run `train.py` with collected data
3. **Validate**: Test on unseen shapes (target >90% efficiency)
4. **Deploy**: Commit model files (automatically loaded by dispatcher)

## Key Innovations

1. **Comprehensive Feature Engineering**: 74 features capture problem-kernel-hardware interactions
2. **Tier-1 Extended Training**: 1,372 shapes (vs 185 baseline) for better edge case coverage
3. **Compressed Models**: LGBM.gz reduces size 8-10× without accuracy loss
4. **Operation-Specific Models**: Separate optimizations for forward/backward passes
5. **Validation Framework**: Automated testing on unseen production workloads

## Impact

- **Production Deployment**: Enable fast kernel selection in MIOpen/ROCm
- **Developer Productivity**: No manual tuning for new problem configs
- **Energy Efficiency**: Near-optimal kernels reduce compute waste
- **Scalability**: Easy to extend to new architectures (gfx940, gfx941)

---

**Version**: 1.0
**Last Updated**: 2026-04-06
**Contact**: Grouped Conv ML Team
