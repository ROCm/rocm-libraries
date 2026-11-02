# hipBLASLt Provider Plugin - Operation Support

This document provides detailed information about the operations supported by the hipBLASLt Provider Plugin for hipDNN.

For general information about hipDNN's operation support, please see the [hipDNN Operation Support](../../../projects/hipdnn/docs/OperationSupport.md) documentation.

## Layout Convention

hipBLASLt is **column-major**; hipDNN's output is **row-major**. To produce `C = A × B` row-major, every plan computes `Cᵀ = Bᵀ × Aᵀ`: `B` and `A` are passed to hipBLASLt swapped (first operand `B`, second `A`), with `transA = getTrans(B)`, `transB = getTrans(A)`. `getTrans` reads each operand's strides (`OP_N` if row-major, `OP_T` if column-major), so both input layouts work.

## Current Operation Support

### Stand-alone Matmul (GEMM)

hipBLASLt Provider Plugin supports stand-alone Matmul (GEMM, general matrix multiplication) operations with the following features and constraints:
- Input and output data types: FP32, FP16, BF16
- Compute data type: FP32
- Transposed inputs: supported
- Batched matmuls: only equal batch sizes are supported, or broadcasting when one input has a single batch (batch=1)
- Fused operations: Matmul supports fused bias, forward activation (ReLU, clamp, GELU with tanh approximation, and Swish with unit beta), and fused bias + forward activation (same supported activations).
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
> `-DHIPBLASLTPROVIDER_ENABLE_MX_GEMM=ON` to enable it.

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
- **gfx1250:** K-contiguous A and B only (#13715); no FP6 inputs.

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
- Layout: A, B and both scales are packed row- or column-major matrices (see the table above); padded leading dimensions are rejected
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
