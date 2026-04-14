# 4-Channel FP16 Grouped Convolution Kernel

Source: `projects/composablekernel/include/ck_tile/ops/direct_convolution/kernel/grouped_4c_fp16_kernel.hpp`

---

## Supported Configurations

| Property        | Value                                 |
|-----------------|---------------------------------------|
| Data type       | fp16 (input, weight, output)          |
| Input layout    | NHWC                                  |
| Weight layout   | KYXC                                  |
| Output layout   | NHWK                                  |
| Filter size     | 3x3 only                             |
| Stride          | 1                                     |
| Dilation        | 1                                     |
| Channels/group  | 4                                     |
| Constraint      | `c == k` (input channels == output channels) |
| Directions      | Fprop (Forward), Dgrad (Backward Data)|
| Architecture    | gfx950 (CDNA4)                        |

---

## Architecture Overview

The kernel uses a **streaming row-by-row architecture**: it processes one input row at a time rather than tiling the full 2D input.

### Circular Accumulator Buffer

A circular buffer of size `kh` (3 for a 3x3 filter) holds partial output accumulators. As each input row is loaded and convolved against `kh` filter rows, the results accumulate into `kh` output row accumulators. When an accumulator has received contributions from all `kh` filter rows, its output row is complete and is written out, freeing the slot for the next output row.

### Wave-Level Parallelism

Each wave computes **4 output columns x 16 groups** = 64 output elements. Multiple waves tile across the output width and group dimensions.

---

## Memory Hierarchy

### 1. Buffer-Load-to-LDS

Input data is transferred from global memory directly into LDS using `__builtin_amdgcn_raw_ptr_buffer_load_lds`. This avoids register pressure from staging data through VGPRs.

### 2. Double-Buffered Input

Two LDS buffers are used for input data, allowing overlap of:
- Loading the next input row into one buffer
- Computing on the current input row from the other buffer

### 3. Weight Prologue

All filter weights are loaded into LDS and then into registers **before** the main loop begins. Since the filter is small (3x3x4 = 36 elements per group), this fits entirely in registers and avoids repeated weight loads during the row loop.

### 4. Output Through LDS

Output results follow the path: fp32 accumulators -> fp16 conversion -> LDS -> global memory. A swizzled write pattern (`SwizzleT`) ensures bank-conflict-free LDS access.

---

## MFMA Usage

The kernel uses the `__builtin_amdgcn_mfma_f32_4x4x4f16` instruction (MFMA 4x4x4 with batch-16).

- **Accumulation** is performed in fp32 for numerical precision.
- **Output conversion** from fp32 to fp16 happens before writing through LDS.
- Each wave processes **16 groups** in the MFMA batch dimension.

---

## Config Struct

```cpp
struct Config {
    int waves_c64;   // waves tiling across groups (in units of 64)
    int waves_q4;    // waves tiling across output width (in units of 4)
    int kh;          // filter height
    int kw;          // filter width
    int n_fold;      // batch folding factor
    int direction;   // Fprop or Dgrad
};
```

### Available Configurations

There are 10 configurations total: 5 for Fprop and 5 for Dgrad. They vary in:

- `waves_c64`: 1 or 2 (how many groups of 64 channels per workgroup)
- `waves_q4`: 1, 2, 4, or 8 (how many groups of 4 output columns per workgroup)

### Configuration Selection

`is_valid_config()` selects a configuration based on:
- Direction (Fprop vs Dgrad)
- Whether the number of groups is divisible by the config's group tiling
- Whether the output width is compatible with the config's column tiling

---

## Dgrad Specifics

### Reversed Filter Indexing

For backward data (Dgrad), the filter is applied in reverse order:

```
weights_reg[(kh - 1 - R) * kw + (kw - 1 - S)]
```

This implements the 180-degree filter rotation required by the transposed convolution.

### Transposed Weight Read

Dgrad uses the `__builtin_amdgcn_ds_read_tr16_b64_v4i16` instruction via `TransposeLDSLayout` for loading weights from LDS. This CDNA4 instruction performs a transposed read that maps naturally to the Dgrad data flow.

### SizeView Remapping

`SizeView<Dgrad>` remaps convolution dimensions so the same kernel loop structure works for both directions:

- Swaps input spatial dimensions (`h`, `w`) with output spatial dimensions (`p`, `q`)
- Swaps stride with dilation
- Inverts padding: `pad = (kh - 1) * dilation - pad`

---

## API

### `make_variant()`

Returns a `KernelVariant` populated with function pointers for applicability checking, launch parameter computation, and kernel dispatch. The variant checks:
- Data type is fp16
- Filter is 3x3, stride 1, dilation 1
- Channels per group is 4
- `c == k`

### `launch(config_idx, lp, par, in, wei, out, workspace, stream)`

Launches the kernel for a given configuration index. Parameters:
- `config_idx` -- index into the configuration table
- `lp` -- `LaunchParams` (grid, block, shared memory)
- `par` -- `Conv2dParams`
- `in`, `wei`, `out` -- device pointers for input, weights, output
- `workspace` -- device pointer for temporary workspace
- `stream` -- HIP stream

### `get_launch_params(config_idx, par)`

Computes grid dimensions and shared memory requirements for a given configuration and convolution parameters.
