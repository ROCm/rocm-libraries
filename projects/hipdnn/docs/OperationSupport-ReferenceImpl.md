# hipDNN CPU Reference Implementation - Operation Support

This document provides a comprehensive overview of the operations currently supported in the hipDNN CPU Reference Implementation and the details of their implementation support.

> [!IMPORTANT]
> ⚠️ **The CPU Reference Implementation is for testing and validation purposes only.** This implementation provides ground-truth results for validating GPU implementations and is not optimized for performance.

> [!NOTE]
> A GPU-based reference implementation also exists in the integration test project. See
> [GPU Reference Implementation - Operation Support](../../../dnn-providers/integration-tests/docs/OperationSupport-GpuReferenceImpl.md) for the operations it covers.

## Purpose

The CPU Reference Implementation serves as:
- **Reference Implementation**: Provides ground-truth results for validating GPU implementations
- **Testing Infrastructure**: Enables comprehensive testing of graph execution
- **Validation Tool**: Ensures correctness of GPU-accelerated operations

It is exposed through `CpuReferenceGraphExecutor`
(`test_sdk/include/hipdnn_test_sdk/utilities/cpu_graph_executor/`), which executes a serialized
graph node by node. A node is accepted only if a plan builder is registered for its
**signature key** (operation + datatypes) and that builder's `isApplicable` check passes.

## Current Operation Support

The following table lists all operations currently supported in the CPU Reference Implementation:

| Operation | Datatypes | Layouts | Implementation | Notes |
|-----------|-----------|---------|----------------|-------|
| BatchNorm Forward Inference | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | CPU Reference |  |
| BatchNorm Forward Inference with Variance | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | CPU Reference |  |
| BatchNorm Forward Training | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | CPU Reference |  |
| BatchNorm Backward | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | CPU Reference |  |
| Convolution Backward Data | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | CPU Reference |  |
| Convolution Forward | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | CPU Reference |  |
| Convolution Backward Weights | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | CPU Reference |  |
| Matmul | FP16, BFP16, FP32 | Any rank >= 2 | CPU Reference | Batch dims broadcast; ragged tensors rejected |
| MoE Grouped Matmul Forward | FP16, BFP16, FP32 | Op-defined | CPU Reference | Ragged tensors rejected |
| MoE Grouped Matmul Backward | FP16, BFP16, FP32 | Op-defined | CPU Reference |  |
| Layernorm Forward | FP16, BFP16, FP32 | All | CPU Reference | Normalizes the trailing `normalizedDimCount` dims; mean/inv-variance outputs optional |
| Layernorm Backward | FP16, BFP16, FP32 | All | CPU Reference | `mean` and `inv_variance` must be provided together or not at all |
| RMSNorm Forward | FP16, BFP16, FP32 | All | CPU Reference | Input rank >= 2; scale rank equals input rank |
| RMSNorm Backward | FP16, BFP16, FP32 | All | CPU Reference | Input rank >= 2; requires the inverse-RMS tensor |
| Reduction | FP16, BFP16, FP32 | All | CPU Reference | Ragged tensors rejected; see [Reduction Modes](#reduction-modes) |
| Resample Forward | FP16, BFP16, FP32 | Op-defined | CPU Reference | MAXPOOL / AVGPOOL modes only; optional INT32 index output |
| Resample Backward | FP16, BFP16, FP32 | Op-defined | CPU Reference | Same modes as forward; MAXPOOL requires the INT32 index tensor |
| SDPA Forward (dense) | FP16, BFP16, FP32, FP8 (E4M3, E5M2 and their FNUZ forms) | Op-defined | CPU Reference | See [SDPA Restrictions](#sdpa-restrictions) |
| SDPA Forward (ragged) | Native: BFP16, FP8 E4M3; other dtypes through the dense rewrite | Op-defined | CPU Reference | Selected when a Q/K/V/O tensor carries a `ragged_offset`; see [SDPA Restrictions](#sdpa-restrictions) |
| SDPA Backward | FP16, BFP16, FP32 | Op-defined | CPU Reference | See [SDPA Restrictions](#sdpa-restrictions) |
| BlockScale Quantize | FP32, FP16, BFP16 in; FP8, FP6, FP4 out | Op-defined | CPU Reference | See [Block-Scale Quantization](#block-scale-quantization) |
| BlockScale Dequantize | FP8, FP6, FP4 in; FP32, FP16, BFP16 out | Op-defined | CPU Reference | See [Block-Scale Quantization](#block-scale-quantization) |
| Pointwise Unary | FP16, BFP16, FP32 | All | CPU Reference | Mixed input/output pairs registered; see [Pointwise Restrictions](#pointwise-restrictions) |
| Pointwise Binary | FP16, BFP16, FP32 | All | CPU Reference | Mixed operand pairs registered; see [Pointwise Restrictions](#pointwise-restrictions) |
| Pointwise Ternary | FP16, BFP16, FP32 | All | CPU Reference | BINARY_SELECT with BOOLEAN mask |

## Implementation Details

### BatchNormalization Operations

| Operation | Plan Builder | Signature Key | Description |
|-----------|-------------|---------------|-------------|
| BatchNorm Forward Inference | `BatchnormFwdInferencePlanBuilder` | `BatchnormFwdInferenceSignatureKey` | Inference-mode forward pass |
| BatchNorm Forward Inference with Variance | `BatchnormFwdInferenceWithVariancePlanBuilder` | `BatchnormFwdInferenceWithVarianceSignatureKey` | Inference-mode with variance forward pass |
| BatchNorm Forward Training | `BatchnormTrainPlanBuilder` | `BatchnormTrainSignatureKey` | Training-mode forward pass with statistics |
| BatchNorm Backward | `BatchnormBwdPlanBuilder` | `BatchnormBwdSignatureKey` | Gradient computation |

### Convolution Operations

| Operation | Plan Builder | Signature Key | Description |
|-----------|-------------|---------------|-------------|
| Convolution Forward | `ConvolutionFwdPlanBuilder` | `ConvolutionFwdSignatureKey` | Forward convolution |
| Convolution Backward Data | `ConvolutionBwdPlanBuilder` | `ConvolutionBwdSignatureKey` | Data gradient computation |
| Convolution Backward Weights | `ConvolutionWrwPlanBuilder` | `ConvolutionWrwSignatureKey` | Weight gradient computation |

### Matmul Operations

| Operation | Plan Builder | Signature Key | Description |
|-----------|-------------|---------------|-------------|
| Matmul | `MatmulPlanBuilder` | `MatmulSignatureKey` | Generic matrix multiplication |
| MoE Grouped Matmul Forward | `MoeGroupedMatmulPlanBuilder` | `MoeGroupedMatmulSignatureKey` | Mixture-of-experts grouped matrix multiplication |
| MoE Grouped Matmul Backward | `MoeGroupedMatmulBwdPlanBuilder` | `MoeGroupedMatmulBwdSignatureKey` | Gradient of the grouped matrix multiplication |

### Normalization Operations

| Operation | Plan Builder | Signature Key | Description |
|-----------|-------------|---------------|-------------|
| Layernorm Forward | `LayernormFpropPlanBuilder` | `LayernormFpropSignatureKey` | Layer normalization forward pass |
| Layernorm Backward | `LayernormBpropPlanBuilder` | `LayernormBpropSignatureKey` | Layer normalization gradients |
| RMSNorm Forward | `RMSNormFwdPlanBuilder` | `RMSNormFwdSignatureKey` | RMS normalization forward pass |
| RMSNorm Backward | `RMSNormBwdPlanBuilder` | `RMSNormBwdSignatureKey` | RMS normalization gradients |

### Reduction, Resample and SDPA Operations

| Operation | Plan Builder | Signature Key | Description |
|-----------|-------------|---------------|-------------|
| Reduction | `ReductionPlanBuilder` | `ReductionSignatureKey` | Reduction over the dimensions where the output size is 1 |
| Resample Forward | `ResampleFwdPlanBuilder` | `ResampleFwdSignatureKey` | Pooling-style resampling |
| Resample Backward | `ResampleBwdPlanBuilder` | `ResampleBwdSignatureKey` | Resampling gradient |
| SDPA Forward (dense) | `SdpaFwdPlanBuilder` | `SdpaFwdSignatureKey` | Scaled dot-product attention forward pass |
| SDPA Forward (ragged) | `SdpaRaggedFwdPlanBuilder` | `SdpaRaggedFwdSignatureKey` | Packed (RFC-0014 ragged) attention forward pass |
| SDPA Backward | `SdpaBwdPlanBuilder` | `SdpaBwdSignatureKey` | Scaled dot-product attention gradients |

### Block-Scale Quantization Operations

| Operation | Plan Builder | Signature Key | Description |
|-----------|-------------|---------------|-------------|
| BlockScale Quantize | `BlockScaleQuantizePlanBuilder` | `BlockScaleQuantizeSignatureKey` | Quantize to a narrow type with per-block scales |
| BlockScale Dequantize | `BlockScaleDequantizePlanBuilder` | `BlockScaleDequantizeSignatureKey` | Dequantize a narrow type with per-block scales |

### Pointwise Operations

| Operation Type | Plan Builder | Signature Key | Supported Operations |
|----------------|-------------|---------------|---------------------|
| Unary Operations | `PointwisePlanBuilder` | `PointwiseSignatureKey` | RELU_FWD, SIGMOID_FWD, TANH_FWD, ABS, NEG, GELU_FWD, GELU_APPROX_TANH_FWD, SWISH_FWD |
| Binary Operations | `PointwisePlanBuilder` | `PointwiseSignatureKey` | ADD, SUB, MUL, MAX_OP, MIN_OP, CMP_GT, RELU_BWD, SIGMOID_BWD, TANH_BWD |
| Ternary Operations | `PointwisePlanBuilder` | `PointwiseSignatureKey` | BINARY_SELECT (BOOLEAN mask) |

The names are the `PointwiseMode` values of the serialized graph. The frontend spells `MAX_OP` and `MIN_OP`
as `PointwiseMode::MAX` and `PointwiseMode::MIN`. `IDENTITY` is not in the list: the reference class
implements it (`ReferencePointwiseBase`), but the registry and the implemented-mode set leave it out, so the
executor declines it. The GPU reference takes it.

`CMP_GT` outputs must have their tensor data type explicitly set to `BOOLEAN`; generic
pointwise output inference otherwise uses the graph's intermediate data type.

## Operation Restrictions

### Pointwise Restrictions

**Datatype tuples.** A node is accepted only for a registered tuple, and the key holds the mode, the datatype
of every operand and of the output, and the node's compute datatype. Every mode of a group accepts every tuple
of the group's table.

Unary modes:

| Input | Output | Compute |
|-------|--------|---------|
| FP32 | FP32 | FP32 |
| FP16 | FP16 | FP32 |
| BFP16 | BFP16 | FP32 |
| FP32 | FP16 | FP32 |
| FP32 | BFP16 | FP32 |

Binary modes `ADD`, `SUB`, `MUL`, `MAX_OP`, `MIN_OP`, `RELU_BWD`, `SIGMOID_BWD` and `TANH_BWD`:

| First input | Second input | Output | Compute |
|-------------|--------------|--------|---------|
| FP32 | FP32 | FP32 | FP32 |
| FP16 | FP16 | FP16 | FP16 |
| BFP16 | BFP16 | BFP16 | BFP16 |
| FP16 | FP16 | FP16 | FP32 |
| BFP16 | BFP16 | BFP16 | FP32 |
| FP16 | FP16 | FP32 | FP32 |
| BFP16 | BFP16 | FP32 | FP32 |
| FP16 | FP32 | FP32 | FP32 |
| BFP16 | FP32 | FP32 | FP32 |
| FP32 | FP16 | FP32 | FP32 |
| FP32 | BFP16 | FP32 | FP32 |
| FP32 | FP16 | FP32 | FP16 |
| FP32 | BFP16 | FP32 | BFP16 |

The last two rows have a 16-bit compute type and an FP32 output. They are the only tuples whose compute type is
narrower than their output.

`CMP_GT` takes the FP32, FP16 and BFP16 same-type pairs with a `BOOLEAN` output and FP32 compute.
`BINARY_SELECT` takes FP32, FP16 and BFP16 for both data operands and the output, a `BOOLEAN` mask, and FP32
compute.

**Mixed-datatype tuples.** In unary modes the CPU reference narrows an FP32 input to FP16 or BFP16 but registers
no conversion that widens a 16-bit input or goes between FP16 and BFP16. In binary modes it pairs an FP32
operand with a 16-bit one, and two operands of the same 16-bit type may produce an FP32 output, but it never
mixes FP16 with BFP16 in one node. The GPU reference registers the opposite set of mixes (see
[GPU Reference Implementation - Operation Support](../../../dnn-providers/integration-tests/docs/OperationSupport-GpuReferenceImpl.md#pointwise-modes)),
so neither reference's tuples contain the other's.

**Compute datatype.** The CPU reference evaluates every pointwise node in `double` and rounds once to the
output type, whatever the compute type of the node says. The node's `compute_data_type` only has to match
the compute column of a registered tuple, or the node is declined. The FP16 and BFP16 compute tuples above
therefore run in `double` too, not in 16-bit arithmetic. The GPU reference computes in the node's compute type
(FP32).

**Parameters.**

| Parameter | Used by | Default when unset |
|-----------|---------|--------------------|
| `relu_lower_clip` | `RELU_FWD`, `RELU_BWD` | 0 |
| `relu_upper_clip` | `RELU_FWD`, `RELU_BWD` | largest finite value |
| `relu_lower_clip_slope` | `RELU_FWD`, `RELU_BWD` | 0 |
| `swish_beta` | `SWISH_FWD` | 1 |
| `elu_alpha`, `softplus_beta` | no registered mode | not supported |

A ReLU or Swish parameter set on a unary mode other than `RELU_FWD` and `SWISH_FWD`, or on a binary mode other
than `RELU_BWD`, makes the node throw `std::runtime_error` when it executes, and any of them on `BINARY_SELECT`
throws when the plan is built. `elu_alpha` and `softplus_beta` throw when the plan is built, on any mode. The
GPU reference ignores a parameter the mode does not use instead.

**Shapes and layouts.** Every tensor is addressed through its own strides, so channels-last (NHWC), mixed
layouts across the operands and the output, and padded (non-packed) strides all work. Every input broadcasts
to the output: right-aligned, each dimension equal to the output's or 1, and an input may have a lower rank
than the output. There is no rank limit. The GPU reference addresses at most 5 dimensions
(`GpuReferencePointwise::MAX_DIMS`) and declines a deeper pointwise node, so a graph with a 6-D or deeper
output is only ever run by this reference.

### Reduction Modes

`ADD`, `AVG`, `AMAX`, `NORM1`, `NORM2`, `MUL`, `MUL_NO_ZEROS`, `MIN_OP`, `MAX_OP`. Input and output must
have the same rank; each output dimension must equal the input dimension or be 1, and at least one
dimension must be reduced.

### SDPA Restrictions

Design, the full feature table for both references, and how they are tested:
[SdpaReferenceImplementations.md](SdpaReferenceImplementations.md).

| Feature | Dense SDPA Forward | Ragged SDPA Forward | SDPA Backward |
|---------|--------------------|---------------------|---------------|
| Log-sum-exp `stats` output | Supported | Supported (`FP32`): dense `[B, H, Sq, 1]`, or ragged with its own `INT32` `ragged_offset` in any layout the ragged container accepts (`[H, T]` or `[T, H]`), with Q's per-batch lengths | Required input |
| Variable sequence lengths (`seq_len_q` / `seq_len_kv`, padding) | Supported: both INT32 tensors, one value per batch | Supported, both or neither: they bound the per-batch lengths. Without them each `ragged_offset` block, clamped to `S_max`, is the length | Supported (same rules as dense forward) |
| Diagonal band (`left_bound`, `right_bound`, `diagonal_alignment`) | cuDNN semantics: `left_bound = L` keeps L keys up to and including the diagonal, `right_bound = R` keeps R keys past it; `left_bound = 0` is rejected | Same | Same |
| Dropout with an explicit mask (`dropout_mask` FLOAT [B or 1, H or 1, Sq, Skv] + one-element `dropout_scale`; backward also `dropout_scale_inv`) | Supported | Through the dense rewrite | Supported |
| Philox dropout (`dropout_probability`, seed, offset) | Supported only with an `rng_dump` output (FLOAT [B, H, Sq, Skv]): the engine's dumped keep mask is applied as the explicit mask with scale 1/(1−p). Without `rng_dump`, not supported | Through the dense rewrite (same rule) | Not supported (the backward node has no `rng_dump`) |
| ALiBi (`alibi_mask`) | Supported: cuDNN slopes, `(skv - sq) * slope(h)` added after the bias; needs top-left causal (`right_bound = 0`) (SEM-021) | Through the dense rewrite | Supported (same rules) |
| `padding_mask` without both seq_len tensors | Not supported | Not supported | Not supported |
| Additive attention bias (`attn_mask`) | Supported | Through the dense rewrite | Supported |
| Bias gradient (`dbias` output) | - | - | Supported when a bias is set; `dbias` must have the bias tensor's shape and dtype |
| Paged KV cache (cuDNN layout; requires both seq_len tensors) | Supported: K, V or both, each with its own block size; table width `ceil(max_seq_len_kv / block_size)`; an unset extent is inferred as cuDNN does (SEM-009) | Through the dense rewrite (ragged Q over paged K/V) | - |
| Attention sink (`sink_token`, FLOAT [1, H, 1, 1]) | Supported; the log-sum-exp `stats` include the sink | Through the dense rewrite | Not supported (the backward node has no sink input) |
| Block-sparse attention (`block_mask`, UINT8, one bit per 128 x 128 tile; SEM-025) | Supported | Through the dense rewrite (below) | - |
| Softmax max / sum-exp stats outputs (cuDNN `Max`, `Sum_exp`; FLOAT [B, H, Sq, 1]) | Supported: `Max` includes the sink logit; a row with no live key and no sink has `Max` = -inf, `Sum_exp` = 0 | Through the dense rewrite (below) | - |
| Ragged tensors | Through the dense rewrite (below) | Required on every Q/K/V/O | Through the dense rewrite (below) |
| FP8 inputs | Requires Q/K/V descale tensors; descales rejected for non-FP8 inputs. cuDNN `sdpa_fp8` softmax and output scaling: `scale_s` with `descale_s`, `scale_o`, and the `amax_s` / `amax_o` outputs, each a one-element `FLOAT` tensor (SEM-024); not with dropout | Optional `FLOAT` Q/K/V descales (scalar `[1]` or per KV head `[B, H_kv, 1, 1]`) for any registered dtype; softmax descale, softmax/output scales and amax go through the dense rewrite | - |

"Not supported" means the reference reports the graph as not applicable. Every attribute is implemented,
declined, or an accepted engine hint (`generate_stats`, `mma_core_mode`, `implementation`, `compute_data_type`).
"Through the dense rewrite" means the native ragged plan declines the feature and the executor runs the
graph through the dense rewrite (below) under the dense column's rules.
Dense FP8 forward is registered for `FP8_E4M3`, `FP8_E4M3_FNUZ`, `FP8_E5M2` and `FP8_E5M2_FNUZ` inputs
with `BFP16` output or an output in the input's FP8 format; ragged
FP8 forward for `FP8_E4M3` inputs with `BFP16` output.

The ragged plan matches the GPU reference's ragged plan. It needs RFC-0014 tensors with logical
`[B, H, S, D]` dims, an `INT32` `ragged_offset` aux on every primary, and strides the ragged container
accepts: token-major, head-major, a padded token stride, and shifted or non-monotonic origins with
`seq_len`. The container rejects layouts that alias. Applicability sees only graph attributes, not offset values: layouts that depend on the values are rejected when the tensor is built (bundle load), or with `std::invalid_argument` at execute for callers passing raw pointers. Per-batch lengths come from `seq_len_q/kv` when
bound (both or neither; a `padding_mask` only with both, as `isSupportedSdpaPadding`), else from the
`ragged_offset` blocks clamped to `S_max`. A node is keyed as ragged when any of its Q/K/V/O tensors
has a `ragged_offset`. The ragged plan needs all four to be ragged, and the dense plan rejects any
ragged tensor. So a partly ragged node is declined by both plans and left to the dense rewrite, which declines it too.

Ragged graphs the native ragged plan declines, and ragged backward graphs, run through the dense
rewrite when it accepts them (`RaggedSdpaDensify.hpp`): the packed tensors are unpacked into dense padded ones with the
per-batch lengths (`seq_len` when supplied, else the offset blocks) as `seq_len_q/kv`, the dense plan
runs, and the outputs are packed back. They therefore support everything the dense forward and
backward columns list. The rewrite serves `Max` and `Sum_exp` (bound by `seq_len_q`), `INT64`
offsets and ragged Q over non-ragged paged K/V. What it declines (for example a `ragged_offset` on
paged K/V, a non-`INT32` `seq_len`) is listed in RFC 0014 §4.10; such a graph is unsupported.

The semantics the references implement (cuDNN's), and every place where cuDNN, PyTorch, vLLM and
FlashAttention-style kernels disagree, are recorded in the [SDPA semantics register](SdpaSemanticsRegister.md).
A row with no live key produces O = 0 and LSE = -inf in forward and contributes zero to every gradient in
backward.

### Resample Modes

Forward and backward accept `MAXPOOL`, `AVGPOOL_EXCLUDE_PADDING` and `AVGPOOL_INCLUDE_PADDING` with
`NEG_INF_PAD` or `ZERO_PAD` padding. The signature key's index type is either unset (no index tensor)
or `INT32` (index tensor present).

### Block-Scale Quantization

Narrow types are `FP8_E4M3`, `FP8_E5M2`, `FP6_E2M3`, `FP6_E3M2` and `FP4_E2M1`, with `FP8_E8M0` scales.
Non-narrow tuples (`FP32` input to `FP32` / `FP16` / `BFP16` output for quantize; `FP32` / `FP16` / `BFP16`
input with `FP32` scale for dequantize) are also registered.

## Use in the Integration Tests

The integration test project selects this implementation with `--verification-mode cpu` (or as the last
step of the default `auto` chain). The harness only registers bundles for CPU validation when every
node type in the graph is in its CPU-required set. That set is a subset of the operations above:
BatchNorm (all four variants), Layernorm (forward/backward), RMSNorm (forward/backward), Pointwise, and
SDPA forward and backward (see `ReferenceOpCoverage.cpp` in the integration test project). SDPA bundles are
additionally limited to `quick`-tier bundles with a bounded working set, to keep CPU time down. See the
[integration test docs](../../../dnn-providers/integration-tests/docs/running-tests.md#verification-modes)
for the verification modes.

A graph is covered only when every one of its nodes is in the set, so a fused graph such as Convolution
Forward + Pointwise is outside it. Convolution Forward and Matmul are not in the CPU set on purpose: no
golden-bearing bundle uses them, and the CPU lane has a cost gate only for SDPA. The GPU lane is the fast route
for them. It covers Convolution Forward, and covers Convolution Forward + Pointwise once Pointwise joins its
set, which `TestPointwiseReferenceParity` requires as soon as the GPU reference accepts every pointwise graph
this one does.

## Legend

### Datatypes
- **FP16**: Half-precision floating point (16-bit)
- **BFP16**: Brain floating point (16-bit)
- **FP32**: Single-precision floating point (32-bit)
- **FP8 (E4M3, E5M2, E4M3_FNUZ, E8M0)**: 8-bit floating point; E8M0 is the exponent-only block scale type
- **FP6 (E2M3, E3M2), FP4 (E2M1)**: Micro-scaling (MX) narrow floating point types, used by block-scale quantization only
- **INT32**: Index tensors (resample)
- **BOOLEAN**: Comparison outputs and `BINARY_SELECT` masks

### Layouts
- **NCHW**: Batch, Channels, Height, Width (2D, channel-first)
- **NHWC**: Batch, Height, Width, Channels (2D, channel-last)
- **NCDHW**: Batch, Channels, Depth, Height, Width (3D, channel-first)
- **NDHWC**: Batch, Depth, Height, Width, Channels (3D, channel-last)
- **All**: Any stride layout; the operation does not constrain dimension ordering
- **Any rank >= 2**: Matmul `(...batch, M, K)` x `(...batch, K, N)`
- **Op-defined**: Dimension ordering is fixed by the operation's definition, not by a layout name

> **Note:** The layout names (NCHW, NHWC, etc.) describe the **memory layout** controlled by
> strides. Dimension ordering is operation-specific: convolution and batch normalization use
> `(N, C, H, W)` / `(N, C, D, H, W)` ordering, matmul uses `(...batch, M, K)` ordering,
> and pointwise operations are dimension-agnostic. See the
> [Porting Guide](./PortingGuide.md#tensor-dimensions-and-layouts) for details.

### Implementation
- **CPU Reference**: CPU-based reference implementation for validation

## Extension Guidelines

For detailed information on adding new operations or datatypes to the CPU Reference Implementation, please refer to the [Extension Guidelines](./rfcs/0001_CpuGraphExecutorDesign.md#extension-guidelines) section in the CPU Graph Executor Design document.

## Related Documentation

- [hipDNN Operation Support](./OperationSupport.md) - Central hub for hipDNN operation support
- [GPU Reference Implementation - Operation Support](../../../dnn-providers/integration-tests/docs/OperationSupport-GpuReferenceImpl.md) - GPU counterpart in the integration test project
- [CPU Graph Executor Design](./rfcs/0001_CpuGraphExecutorDesign.md) - Detailed architecture documentation
- [Release Testing](./testing/RELEASE_TESTING.md) - Testing strategy and procedures
