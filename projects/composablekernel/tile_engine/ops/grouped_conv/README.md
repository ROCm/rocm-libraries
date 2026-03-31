# Grouped Convolution Tile Engine

Benchmark and kernel generation utilities for grouped convolution operations.

## Files

- **`grouped_conv_benchmark.py`** - Quick benchmark utility for testing specific kernels
- **`grouped_conv_instance_builder.py`** - Kernel configuration generator from JSON sweeps
- **`configs/*.json`** - Kernel sweep configuration files

## Features

- Tests forward, backward data, and backward weight convolutions
- Multiple problem presets (ResNet, MobileNet, custom)
- Multiple kernel configurations (pipelines, schedulers, tile sizes)
- Trait-based kernel filtering via JSON configs
- CSV and JSON output support
- Based on actual CK tile problem sizes

## Usage

```bash
# Forward convolution with ResNet problems
python grouped_conv_benchmark.py --variant forward --problems resnet

# Backward data
python grouped_conv_benchmark.py --variant bwd_data --problems bwd_data

# All variants
for variant in forward bwd_data bwd_weight; do
  python grouped_conv_benchmark.py --variant $variant --problems resnet --csv results_$variant.csv
done

# Show best kernel per problem
python grouped_conv_benchmark.py --variant forward --problems all --best
```

## Problem Presets

- `resnet` - ResNet stage configurations (from dispatcher example 07)
- `grouped` - MobileNet-style grouped convolutions
- `bwd_data` - Backward data problem sizes (from dispatcher example 05)
- `bwd_weight` - Backward weight problem sizes (from dispatcher example 06)
- `all` - All ResNet + grouped problems
- Custom: `"N,C,K,G,H,W,Y,X;..."` format

## Kernel Configurations

Based on dispatcher/examples/grouped_conv/cpp/07_multi_tile_benchmark.cpp:

- **Small tile**: 1x64x64, wave 1x4x1, warp 16x16x32
- **Medium tile**: 1x128x128, wave 2x2x1, warp 32x32x16  
- **Large tile**: 1x256x256, wave 2x2x1, warp 32x32x16

Pipelines: compv3, compv4 (forward), compv3/mem (backward)
Schedulers: intrawave, interwave

## Output

- Console: Problem/kernel matrix with FLOPs
- CSV: Detailed measurements per problem/kernel
- JSON: Full benchmark metadata + results

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

- **`forward.json`** - General forward convolution (fp16/bf16, compv3/compv4)
- **`bwd_data.json`** - Backward data (fp16/bf16, compv3/mem)
- **`bwd_weight.json`** - Backward weight (fp16/bf16, compv3/mem)
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

If a trait is **absent** from `trait_config`, all values are generated.
If a trait is **present**, only listed values are kept.
