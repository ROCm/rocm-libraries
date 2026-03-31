# Grouped Convolution Tile Engine

GPU benchmark and kernel generation utilities for grouped convolution operations with JIT compilation.

## Files

- **`grouped_conv_benchmark.py`** - Quick GPU benchmark for specific kernel configs
- **`grouped_conv_full_benchmark.py`** - Production-scale systematic sweep (all kernels × all problems)
- **`grouped_conv_instance_builder.py`** - Kernel configuration generator from JSON sweeps
- **`configs/*.json`** - Kernel sweep configuration files

## Features

- **JIT compilation** via dispatcher's `setup_multiple_grouped_conv_dispatchers()`
- **Actual GPU execution** with HIP event timing
- Tests forward, backward data, and backward weight convolutions
- Multiple problem presets (ResNet, MobileNet, custom)
- Trait-based kernel filtering via JSON configs
- CSV and JSON output with streaming results
- Resume support for long-running sweeps

## Quick Benchmark Usage

Fast benchmarking for specific configs and problems:

```bash
# Forward with ResNet problems
python grouped_conv_benchmark.py configs/forward_ci.json --problems resnet --best

# Backward data with filter
python grouped_conv_benchmark.py configs/bwd_data.json --problems bwd_data \
  --filter "c.tile_n >= 128" --csv results.csv

# Backward weight
python grouped_conv_benchmark.py configs/bwd_weight.json --problems all \
  --workers 8 --json results.json --best

# Compile only (no GPU execution)
python grouped_conv_benchmark.py configs/forward.json --compile-only

# Clean build
python grouped_conv_benchmark.py configs/forward.json --clean --problems resnet
```

## Full Benchmark Usage

Systematic sweeps for all kernels × all problems (production testing):

```bash
# Quick test (2 problems × all kernels from config)
python grouped_conv_full_benchmark.py --variant forward --category quick

# Full sweep (8 problems × all kernels)
python grouped_conv_full_benchmark.py --variant forward --category full --workers 256

# Backward data with filtering
python grouped_conv_full_benchmark.py --variant bwd_data --category full \
  --filter "c.tile_n >= 128" --csv sweep_bwd_data.csv

# Resume from previous run (skip completed measurements)
python grouped_conv_full_benchmark.py --variant forward --category full
# (automatically resumes if CSV exists)
```

## Problem Presets

### Quick Benchmark
- `resnet` - ResNet stage configurations (6 problems)
- `mobilenet` / `grouped` - MobileNet depthwise convolutions (4 problems)
- `bwd_data` - Backward data problems (3 problems)
- `bwd_weight` - Backward weight problems (3 problems)
- `all` - All ResNet + MobileNet problems
- Custom: `"N,C,K,G,H,W,Y,X,..."` format

### Full Benchmark
- `quick` - 2 problems for rapid testing
- `full` - 8+ problems for comprehensive coverage

## Output

- **Console**: Live results with best kernel highlighted
- **CSV**: Detailed measurements (problem, kernel, latency, TFLOPS, non-zero elements)
- **JSON**: Full metadata + results

Example CSV columns:
```
problem_name,N,C,K,G,Hi,Wi,Y,X,dtype,kernel,variant,pipeline,scheduler,tile,latency_ms,tflops,non_zero
```

## Instance Builder

Generate kernel configurations from JSON sweep files:

```bash
# List all kernels from a config
python grouped_conv_instance_builder.py configs/forward.json --arch gfx950 --list

# Count kernels only
python grouped_conv_instance_builder.py configs/receipt0_forward.json --count-only

# Apply custom filter
python grouped_conv_instance_builder.py configs/forward.json \
  --filter "c.tile_n >= 128 and c.pipeline == 'compv4'" --list

# Export to JSON
python grouped_conv_instance_builder.py configs/forward.json \
  --export-json kernels.json
```

### Config Files

- **`forward.json`** - General forward convolution (fp16/bf16, compv3/compv4) - 20 kernels
- **`bwd_data.json`** - Backward data (fp16/bf16, compv3/mem) - 20 kernels
- **`bwd_weight.json`** - Backward weight (fp16/bf16, compv3/mem) - 20 kernels
- **`forward_ci.json`** - CI subset (fp16 only, compv3 only) - 5 kernels
- **`receipt0_forward.json`** - Full sweep (fp16/bf16/fp32, all pipelines) - 30 kernels

Config structure (trait-based filtering):
```json
{
  "variant": "forward",
  "trait_config": {
    "data_type": {"values": ["fp16", "bf16"]},
    "pipeline": {"values": ["compv3", "compv4"]},
    "scheduler": {"values": ["intrawave"]},
    "ndim_spatial": {"values": [2]}
  }
}
```

**Filtering logic:**
- **Absent trait** → all values generated (e.g., no `pipeline` filter → generates compv3 + compv4)
- **Present trait** → only listed values kept (e.g., `"pipeline": {"values": ["compv3"]}` → compv3 only)

## Architecture

Based on FMHA tile engine design:

1. **`grouped_conv_instance_builder.py`** - Expands JSON configs → GroupedConvKernelConfig lists
2. **Dispatcher integration** - `setup_multiple_grouped_conv_dispatchers()` for parallel JIT
3. **GPU runners** - `GpuGroupedConvRunner` for actual hardware execution
4. **Streaming results** - CSV/JSON output written incrementally

## Example Workflow

```bash
# 1. Quick validation (5 kernels × 6 problems)
python grouped_conv_benchmark.py configs/forward_ci.json --problems resnet --best

# 2. Full sweep (20 kernels × 8 problems)
python grouped_conv_full_benchmark.py --variant forward --category full --workers 16

# 3. Analyze results
head -20 grouped_conv_sweep_results.csv

# 4. Resume if interrupted
python grouped_conv_full_benchmark.py --variant forward --category full
# Skips already-completed measurements from CSV
```

## Hardware Validation

All benchmarks run actual GPU kernels on hardware (tested on gfx950):
- Forward: 30.94 TFLOPS peak (ResNet-stage2, 1x64x64 tile)
- Backward Data: 4.75 TFLOPS peak
- Backward Weight: 2.00 TFLOPS peak

Pipelines: compv3, compv4 (forward), compv3/mem (backward)
Schedulers: intrawave, interwave
Tile sizes: 64x64, 128x128, 256x256
