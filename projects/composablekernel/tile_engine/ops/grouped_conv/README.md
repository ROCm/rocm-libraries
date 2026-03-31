# Grouped Convolution Tile Engine Benchmark

Benchmark utility for testing grouped convolution kernels with various problem dimensions, algorithms, and pipelines.

## Features

- Tests forward, backward data, and backward weight convolutions
- Multiple problem presets (ResNet, MobileNet, custom)
- Multiple kernel configurations (pipelines, schedulers, tile sizes)
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
