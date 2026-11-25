# hipBLASLt Provider Plugin - Operation Support

This document provides detailed information about the operations supported by the hipBLASLt Provider Plugin for hipDNN.

For general information about hipDNN's operation support, please see the [hipDNN Operation Support](../../../projects/hipdnn/docs/OperationSupport.md) documentation.

## Layout Convention

hipBLASLt is **column-major**; hipDNN's output is **row-major**. To produce `C = A × B` row-major, every plan computes `Cᵀ = Bᵀ × Aᵀ`: `B` and `A` are passed to hipBLASLt swapped (first operand `B`, second `A`), with `transA = getTrans(B)`, `transB = getTrans(A)`. `getTrans` reads each operand's strides (`OP_N` if row-major, `OP_T` if column-major), so both input layouts work.

## Current Operation Support

### Stand-alone Matmul (GEMM)

hipBLASLt Provider Plugin supports stand-alone Matmul (GEMM, general matrix multiplication) operations with the following features and constraints:
- Input and output data types: FP32, FP16, BF16. A and B must have the same data type. With a fused activation, the activation output must be FP32, FP16 or BF16
- Compute data type: FP32
- Transposed inputs: supported
- Batched matmuls: only equal batch sizes are supported, or broadcasting when one input has a single batch (batch=1). hipBLASLt takes one flattened batch stride, so the batch axes of each tensor must nest contiguously (each axis with extent above one strides by the inner batch stride times the extents inside it). Batch axes of size 1 are ignored. In a fused graph, the final output must have the same batch dimensions and the same flattened batch stride as the matmul output
- Pass-by-value tensors: A, B, the bias input and the output each need a device buffer, so none of them may be pass-by-value
- Tensor roles: the plan writes only the final output, so every fused intermediate (the matmul output when a bias or activation follows, the bias output when an activation follows) must be `virtual`; all other tensors must not be. A graph with any other layout is declined
- Bias: a bias tensor must have packed strides (hipBLASLt reads it as a vector; strides of size-1 dimensions are ignored) and data type FP32 or the final output type (hipBLASLt's documented bias types are the D type and the FP32 scale type); other FP16/BF16 combinations pass the type check and are declined when hipBLASLt has no kernel
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

### rocBLAS Matmul (`ROCBLAS_MATMUL_ENGINE`)

> **Build-time feature flag.** Built with `-DHIPBLASLTPROVIDER_ENABLE_ROCBLAS=ON` (default **OFF**). hipBLASLt itself is
> optional (`HIPBLASLTPROVIDER_ENABLE_HIPBLASLT`, default **ON**), so a rocBLAS-only plugin serves the targets hipBLASLt is not
> built for, such as gfx906 and gfx103X.

A stand-alone Matmul through `rocblas_gemm_strided_batched_ex`, with no fused operations:
- Input and output data types: FP32, FP16, BF16; an FP32 output from FP16 or BF16 inputs
- Compute data type: FP32
- Any row- or column-major operand layout, batches collapsed to one batch stride, and broadcasting of a single-batch operand
- Every size and leading dimension must fit 32 bits

### Convolution as explicit GEMM (`HIPBLASLT_CONV_ENGINE`, `ROCBLAS_CONV_ENGINE`)

> **Build-time feature flag.** Built with `-DHIPBLASLTPROVIDER_ENABLE_CONV=ON` (default **OFF**), five engines per enabled
> backend: the exact engine, its TF32 and reduced-precision-reduction siblings, and deterministic siblings of the exact and
> TF32 engines (table below).

Single-node convolution graphs, forward, backward data and backward weights, and forward convolutions followed by a pointwise
epilogue (below):
- Data types: FP32, FP16, BF16, the same for all three tensors; compute type unset or FP32,
  or FP16 for FP16 tensors (cuDNN's set), accumulated in FP32
- 1-D, 2-D and 3-D (rank 3 to 5), any x and y strides, groups (including depthwise), padding (including asymmetric), stride,
  dilation, cross-correlation and true convolution

How a convolution runs, in order of preference:
1. **Directly on the tensors**, no workspace: a 1x1 filter with unit stride and no padding (a GEMM over pixels or channels), or a
   filter that covers the whole unpadded input (a GEMM over images).
2. **im2col**: every other convolution. A repack kernel writes the column matrix to the workspace and one GEMM reads it; backward data
   runs the GEMM into an FP32 column matrix that col2im sums into dx. The repack workspace is held to 1 GiB: a larger column
   matrix is built a slice of images at a time, and an image whose column matrix alone exceeds the cap in bands of rows along its
   outermost spatial axis (backward data bands dx rows, each computing the output rows that reach them). Backward weights sums
   the slices or bands, and any reduction long enough that rocBLAS would split it, in FP32 before rounding once. The repack
   kernels are compiled at runtime by HIPRTC on first use.

Fused forward graphs (every engine but the RPR siblings): the convolution, then a chain of pointwise nodes: scale and bias
(per-channel or scalar), relu (clipped, clamp, leaky), GELU (exact and tanh), swish/silu, sigmoid, tanh, elu, abs and softplus.
The chain runs in FP32 and y is rounded once:
- **Native:** a leading per-channel bias in the GEMM that writes y (hipBLASLt's bias epilogue where y's channels are the
  GEMM's rows; a broadcast bias matrix added as the GEMM's C, in y's type, where they are its columns, on both backends),
  then hipBLASLt's sigmoid (16-bit y), or a relu / clamp whose bounds y's type holds exactly as an in-place pass over y
- **Fused into the pass:** a channels-first y staged channels-last; the copy back into y applies the chain
- **Extra pass:** otherwise (groups, other activations, a bias of another type): the GEMMs write y in FP32 to the workspace
  and one cast applies the chain

`HIPBLASLT_CONV_LIBRARY_ACTIVATION_ENGINE` (opt-in: `HIPBLASLT_PROVIDER_CONV_LIBRARY_ACTIVATIONS=1`; deterministic,
hipBLASLt offered no workspace) runs relu, clamp, swish (beta 1), tanh GELU and fp32 sigmoid in hipBLASLt's epilogue. It serves only fused
graphs with one of those. Its special values differ from every other engine's: relu maps NaN to 0, clamp maps NaN to its upper
bound, swish and tanh GELU map -inf to NaN.

Engines and their numerical notes ([RFC 0025](../../../projects/hipdnn/docs/rfcs/0025_NumericalNotes.md)). Every engine
below but the opt-in library-activation one is registered by default; a caller excludes one by its notes through the hipDNN API's cuDNN-compatible
`deselect_numeric_notes`.

| Engine | Numerical notes |
|---|---|
| `ROCBLAS_CONV_ENGINE`, `HIPBLASLT_CONV_ENGINE` | `NONDETERMINISTIC` |
| `ROCBLAS_CONV_ENGINE_DETERMINISTIC`, `HIPBLASLT_CONV_ENGINE_DETERMINISTIC` | none |
| `ROCBLAS_CONV_TF32_ENGINE`, `HIPBLASLT_CONV_TF32_ENGINE` | `TENSOR_CORE`, `NONDETERMINISTIC` |
| `ROCBLAS_CONV_TF32_ENGINE_DETERMINISTIC`, `HIPBLASLT_CONV_TF32_ENGINE_DETERMINISTIC` | `TENSOR_CORE` |
| `ROCBLAS_CONV_RPR_ENGINE`, `HIPBLASLT_CONV_RPR_ENGINE` | `REDUCED_PRECISION_REDUCTION`, `NONDETERMINISTIC` |
| `HIPBLASLT_CONV_LIBRARY_ACTIVATION_ENGINE` (opt-in: `HIPBLASLT_PROVIDER_CONV_LIBRARY_ACTIVATIONS=1`) | none |

- **TF32** (`*_CONV_TF32_ENGINE`): they serve only convolutions with an FP32 GEMM and permit TF32 for them: hipBLASLt is
  asked for `HIPBLAS_COMPUTE_32F_FAST_TF32` and falls back to FP32 when it has no algorithm; rocBLAS runs under
  `rocblas_xf32_xdl_math_op`, which it applies only on gfx942. They report `TENSOR_CORE`, the note cuDNN reports for its TF32
  convolution engines and the one PyTorch's cuDNN path excludes for FP32 when TF32 is not allowed.
- **Reduced-precision reduction** (`*_CONV_RPR_ENGINE`): one form, a 1x1, unit-stride, unpadded FP16 backward weights of
  several channels-first images, as one GEMM per image adding into the FP16 dw (beta 1, MIOpen's `GemmWrw1x1_stride1`), so
  dw is rounded to FP16 once per image instead of once (cuDNN's `REDUCED_PRECISION_REDUCTION`). No workspace. BF16 has no
  reduced form. Compute type unset, FP32 or FP16.
- **Determinism** (`*_DETERMINISTIC`): the exact and TF32 engines run the libraries' defaults (rocBLAS atomics allowed,
  hipBLASLt offered its workspace), which may split a reduction across workgroups, and report `NONDETERMINISTIC`. Their
  deterministic siblings run rocBLAS with atomics not allowed (the handle's mode restored after the plan) or offer hipBLASLt
  no workspace, as MIOpen's deterministic GEMMs do, and repack by gathers (each output element summed by one thread in a
  fixed order), so a plan gives the same bits on every run; the device tests run each plan six times and compare the bits.
  The reduced-precision-reduction engines have one form, `NONDETERMINISTIC`.

Declined:
- A filter whose (channel, spatial) axes nest in neither KCRS nor KRSC order
- A convolution whose column matrix for one row of the outermost spatial axis exceeds 1 GiB
- A GEMM leading dimension beyond 32 bits, or a GEMM reduction beyond 2^30 into an FP16 or BF16 output. Other long GEMMs are
  split to fit the libraries' 32-bit arguments: a batch stride beyond 32 bits runs one call per image, and an n (or, into FP32,
  a k) beyond 2^30 runs in chunks

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
