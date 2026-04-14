## CK Tile direct convolutions

CK Tile convolutions uses warp level GEMM pipelines to compute to forward, backward data, backward weigth convolutions.

The initial implementations port the MIOpen HIP convolutions (`projects/miopen/src/hipconv`) to use CK Tile abstractions.

For the original README, see [`projects/miopen/src/hipconv/README.md`](../../../../../miopen/src/hipconv/README.md).

## Supported configurations

- **Data type**: fp16
- **Layout**: NHWC input, KYXC weights, NHWK output
- **Direction**: Fprop (Forward), Dgrad (Backward Data), Wgrad (Backward Weigth)
- **Filter**: 3x3, stride 1, dilation 1
- **Group size**: 4, 8, and 16 channels per group
- **Constraint**: input channels == output channels (`c == k`)

The kernels use MFMA 16x16x16 (4c, 16c) and MFMA 16x16x32 (8c) instructions with buffer-load-to-LDS for input staging.

## Supported device architectures

CK Tile direct convolutions support `gfx942` and `gfx950` device architectures.

## Project structure

```
kernel/                  — kernel implementations
  grouped_4c_fp16.h      — 4-channel kernel (MFMA 16x16x16)
utils/                   — utilities for kernel implementations

```

## Testing

The unit and integration tests are located in directory `projects/composablekernel/test/ck_tile/direct_conv`.


## Documentation

The direct convolution documentation is located at `projects/composablekernel/docs/direct_convolution`.