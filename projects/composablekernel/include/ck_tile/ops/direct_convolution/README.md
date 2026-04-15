## CK Tile direct convolutions

CK Tile convolutions use warp-level GEMM pipelines to compute forward, backward data, and backward weight convolutions.

The initial implementations port the MIOpen HIP convolutions (`projects/miopen/src/hipconv`) to use CK Tile abstractions.

For the original README, see [`projects/miopen/src/hipconv/README.md`](../../../../../miopen/src/hipconv/README.md).

## Supported configurations

- **Data type**: fp16
- **Layout**: NHWC input, KYXC weights, NHWK output
- **Direction**: Fprop (Forward), Dgrad (Backward Data)
- **Filter**: 3x3, stride 1, dilation 1
- **Group size**: 4 channels per group
- **Constraint**: input channels == output channels (`c == k`)

The kernel uses MFMA 4x4x4 batch-16 instructions with buffer-load-to-LDS for input staging.

## Supported device architectures

CK Tile direct convolutions support `gfx942` and `gfx950` device architectures.

## Project structure

```
kernel/                               — kernel implementations
  grouped_4c_fp16_hip_conv_impl.hpp   — 4-channel grouped fp16 kernel (MFMA 4x4x4 batch-16) using pure HIP.
  grouped_4c_fp16_tile_conv_impl.hpp  — 4-channel grouped fp16 kernel (MFMA 4x4x4 batch-16) using pure CK Tile abstractions.
utils/                                — utilities for kernel implementations
  conv_params.hpp                     — Conv2dParams, SizeView, Direction/DataType enums
  types.hpp                           — fp16/bf16/fp32/fp8 type aliases and mapping
  mathutil.hpp                        — maximum(), divup() constexpr helpers
  launch_params.hpp                   — LaunchParams (grid, block, shared memory)
  kernel_variant.hpp                  — KernelVariant dispatch interface
  matrix_layout.hpp                   — MFMA register layout mapping
  swizzle.hpp                         — LDS swizzle for bank-conflict-free access
  transpose_lds_layout.hpp            — DS_READ_TR_B16 layout for CDNA4 Dgrad
  memory.hpp                          — vmcnt wait intrinsics
  detail.hpp                          — static_for / dispatch compile-time helpers
```

## Testing

The unit and integration tests are located in directory `projects/composablekernel/test/ck_tile/direct_conv`.

## Documentation

Detailed documentation is available at `projects/composablekernel/docs/direct_convolution`:

- [Utility Components](../../../../../composablekernel/docs/direct_convolution/utils.md) — Conv2dParams, SizeView, MatrixLayout, Swizzle, and other utilities
- [4-Channel FP16 Kernel](../../../../../composablekernel/docs/direct_convolution/kernel_4c_fp16.md) — Kernel architecture, MFMA usage, streaming pipeline, Dgrad specifics

## Basic CK Tile utilities

We want to port the `grouped_4c_fp16_hip_conv_impl.hpp` from pure HIP to use the CK Tile abstractions

- BufferView - `projects/composablekernel/include/ck_tile/core/tensor/buffer_view.hpp` (raw linear memory access abstraction)
- TensorDescriptor - `projects/composablekernel/include/ck_tile/core/tensor/tensor_descriptor.hpp` (multi-dimensional strided access to linear memory) 
- TensorView - `projects/composablekernel/include/ck_tile/core/tensor/tensor_view.hpp` (multi-dimensional tensor over linear memory usinf BufferView and TensorDescriptor)
- TileDistributionEncoding - `projects/composablekernel/include/ck_tile/core/tensor/tile_distribution.hpp` (distribution of a tensor over threads)
- TileDistribution - `projects/composablekernel/include/ck_tile/core/tensor/tile_distribution.hpp` (mapping between tensor coordinates and data based on TileDistribution encoding).
- TileWindow - `projects/composablekernel/include/ck_tile/core/tensor/tile_window.hpp` (abstaraction of data load/store using tile distribution).
- StaticDistributedTensor - `projects/composablekernel/include/ck_tile/core/tensor/static_distributed_tensor.hpp` (thread local data container defined by the TileDistribution). 
