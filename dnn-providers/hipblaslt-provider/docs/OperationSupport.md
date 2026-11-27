# hipBLASLt Provider Plugin - Operation Support

This document provides detailed information about the operations supported by the hipBLASLt Provider Plugin for hipDNN.

For general information about hipDNN's operation support, please see the [hipDNN Operation Support](../../../projects/hipdnn/docs/OperationSupport.md) documentation.

## Layout Convention

hipBLASLt is **column-major**; hipDNN's output is **row-major**. To produce `C = A × B` row-major, every plan computes `Cᵀ = Bᵀ × Aᵀ`: `B` and `A` are passed to hipBLASLt swapped (first operand `B`, second `A`), with `transA = getTrans(B)`, `transB = getTrans(A)`. `getTrans` reads each operand's strides (`OP_N` if row-major, `OP_T` if column-major), so both input layouts work.

A matrix (the last two dimensions of a tensor) is row-major when `dims[-1] == 1` or `stride[-1] == 1`; otherwise it is column-major when `dims[-2] == 1` or `stride[-2] == 1`; otherwise it is not supported. A size-1 dimension therefore accepts any stride. The same rule derives the transpose flag and the hipBLASLt matrix layout.

The leading dimension the layout uses (the row stride of a row-major matrix, the column stride of a column-major one) must not be negative for A and B; the engine declines a matrix whose leading dimension is negative. Zero is allowed for A and B, for example a broadcast operand whose rows or columns repeat. The output must not overlap itself, so its row stride must be at least its column count (a single-row output takes any row stride), and when it has a batch, its batch stride must be at least the span of one output matrix ((rows - 1) × row stride + columns); a padded output satisfies both. A size-1 dimension's stride is never used, so it does not have to be positive.

The output must be row-major under this rule. The tensor written to the output (the activation output, else the bias output, else the matmul output) is checked, and the engine declines a column-major output.

## Current Operation Support

### Stand-alone Matmul (GEMM)

hipBLASLt Provider Plugin supports stand-alone Matmul (GEMM, general matrix multiplication) operations with the following features and constraints:
- Input and output data types: FP32, FP16, BF16. A and B must have the same data type. With a fused activation, the activation output must be FP32, FP16 or BF16
- Compute data type: FP32
- Transposed inputs: supported
- Batched matmuls: only equal batch sizes are supported, or broadcasting when one input has a single batch (batch=1). hipBLASLt takes one flattened batch stride, so the batch axes of each tensor must nest contiguously (each axis with extent above one strides by the inner batch stride times the extents inside it). Batch axes of size 1 are ignored. In a fused graph, the final output must have the same batch dimensions as the matmul output; its batch stride is its own, so a padded final output is served
- Pass-by-value tensors: A, B, the bias input and the output each need a device buffer, so none of them may be pass-by-value
- Tensor roles: the plan writes only the final output, so every fused intermediate (the matmul output when a bias or activation follows, the bias output when an activation follows) must be `virtual`; all other tensors must not be. A graph with any other layout is declined
- Bias: a bias tensor must have packed strides (hipBLASLt reads it as a vector; strides of size-1 dimensions are ignored) and data type FP32 or the final output type (hipBLASLt's documented bias types are the D type and the FP32 scale type); other FP16/BF16 combinations pass the type check and are declined when hipBLASLt has no kernel
- Fused operations: Matmul supports fused bias, forward activation (ReLU, clamp, GELU with tanh approximation, and Swish with unit beta), and fused bias + forward activation (same supported activations).
- Fused intermediate data types: the matmul output when a bias or activation follows, and the bias output when an activation follows, must be FP32. hipBLASLt runs the epilogue on the FP32 accumulator and rounds once to the output type, so a narrower intermediate cannot be honored and the engine declines the graph. Half-precision inputs and outputs with FP32 intermediates (the cuDNN v8 mixed-precision shape) are accepted.
- ReLU parameters (`RELU_FWD`): an unset `relu_lower_clip` is treated as 0, an unset `relu_upper_clip` as no upper bound, and an unset `relu_lower_clip_slope` or a slope of exactly 0 as no leaky slope.
    - Standard ReLU: no upper clip and a lower clip of 0 (or unset)
    - Clamp: `relu_upper_clip` set; the lower bound is `relu_lower_clip`, or 0 if unset (so ReLU6 needs only the upper clip)
    - Not supported: a non-zero `relu_lower_clip_slope` (leaky ReLU; hipBLASLt has no such epilogue), or a non-zero `relu_lower_clip` without an upper clip

### MX BlockScaleDequantize + GEMM (MX GEMM)

This is a hipDNN graph of two `BlockScaleDequantize` nodes feeding a `Matmul` node — an OCP MX block-scaled matrix multiplication. The plugin recognizes the pattern and executes it as a single fused GEMM, mapping the block-scale dequantization onto hipBLASLt's `VEC32_UE8M0` scale mode.

Supported MX input element types are **FP8 OCP** (`FP8_E4M3` / `FP8_E5M2`), **OCP MX FP6** (`FP6_E2M3` / `FP6_E3M2`), and **OCP MX FP4** (`FP4_E2M1`), including **mixed pairs** (e.g. FP8 × FP4, or FP6 E2M3 × FP6 E3M2).

> **Build-time feature flag.** This path is gated behind the CMake option
> `HIPBLASLTPROVIDER_ENABLE_MX_GEMM`, which defaults to **OFF**; when OFF the plugin
> reports these graphs as unsupported. Build with
> `-DHIPBLASLTPROVIDER_ENABLE_MX_GEMM=ON` to enable it. With the flag ON the plugin
> needs HIPRTC (`hiprtc`) to build and links it: it compiles its scale relayout kernel
> at run time, for the device in use, when the first plan that needs swizzled scales is
> built.

**Graph topology:**
```
BlockScaleDequantize(x_a: MX type, scale_a: UE8M0) → a (virtual)
BlockScaleDequantize(x_b: MX type, scale_b: UE8M0) → b (virtual)
Matmul(a, b) → d
```

**Supported types:**
- Inputs: `FP8_E4M3`, `FP8_E5M2` (OCP FP8), `FP6_E2M3`, `FP6_E3M2` (OCP MX FP6), or `FP4_E2M1` (OCP MX FP4); A and B may differ (mixed pairs allowed)
- Scale tensors: `FP8_E8M0` (UE8M0 block scale)
- Output: `FP32`, `FP16`, or `BF16`
- Compute type: `FP32`

> **Known issue (#10811):** FP8 OCP on operand A paired with FP6 OCP on operand
> B returns incorrect results when A is column-major, so that layout is rejected as
> unsupported on every architecture (observed on gfx950). With a row-major
> (K-contiguous) A the pair is supported.

**Hardware support:**
- **gfx950:** every operand layout in the table below.
- **gfx1250:** K-contiguous A and B only (#13715). The type pairs are those the installed hipBLASLt has gfx1250 MX kernels for; graphs it has no kernel for (FP6 inputs among them) are declined.

**Operand layouts:** each MX input is a packed matrix.

| A `[M, K]` | B `[K, N]` | gfx950 | gfx1250 |
|---|---|---|---|
| row-major (K-contiguous) | column-major (K-contiguous) | yes | yes |
| column-major (M-contiguous) | row-major (N-contiguous) | yes | no |
| row-major | row-major | yes | no |
| column-major | column-major | yes | no |

**Graph requirements:** the plugin validates the hipDNN graph
against these before accepting it. They derive from hipBLASLt's `VEC32_UE8M0` scale mode — see the
[hipBLASLt API reference](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/api-reference.html).
- Tensor roles: the MX inputs, the UE8M0 scales, and the output `d` must be non-virtual; the dequantize outputs `a`/`b` must be virtual (fused intermediates)
- Scale tensors: dtype `FP8_E8M0`, natural (unswizzled) values, one scale per 32-element block: `[M, K/32]` for `scale_a`, `[K/32, N]` for `scale_b`
- Layout: A and B are packed matrices in the orders of the table above; each scale is a packed row- or column-major matrix in either order, whatever its operand's order (a scale in the other order is reordered on the device, in the workspace); the output `d` is a packed row-major matrix; padded leading dimensions are rejected
- Dequantize attributes: `is_negative_scale` is declined; the plan scales by 2^(scale) only
- Alignment: `M % 16 == 0`, `N % 16 == 0`, `K % 128 == 0`
- Block size: innermost `block_size == 32` for both scale tensors (VEC32)
- Batch: single batch only (`batch == 1`); hipBLASLt requires `B == 1` for this scale mode
- Epilogue: none — no fused bias or activation permitted
- Output type: limited to the supported output types above (`FP32`/`FP16`/`BF16`); FP8 and other quantized outputs are not produced by this fused path

## Legend

### Datatypes
- **FP16**: Half-precision floating point (16-bit)
- **BF16**: Brain floating point (16-bit)
- **FP32**: Single-precision floating point (32-bit)
- **FP8_E4M3**: OCP FP8, 4 exponent / 3 mantissa bits (8-bit)
- **FP8_E5M2**: OCP FP8, 5 exponent / 2 mantissa bits (8-bit)
- **FP8_E8M0**: UE8M0 block-scale type, 8-bit unsigned exponent (MX scales)
- **FP6_E2M3**: OCP MX FP6, 2 exponent / 3 mantissa bits (6-bit, packed 4 values per 3 bytes)
- **FP6_E3M2**: OCP MX FP6, 3 exponent / 2 mantissa bits (6-bit, packed 4 values per 3 bytes)
- **FP4_E2M1**: OCP MX FP4, 2 exponent / 1 mantissa bits (4-bit, packed 2 values per byte)
