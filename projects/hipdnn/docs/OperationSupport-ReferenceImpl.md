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
| SDPA Forward | FP16, BFP16, FP32, FP8 E4M3 | Op-defined | CPU Reference | See [SDPA Restrictions](#sdpa-restrictions) |
| SDPA Backward | FP16, BFP16, FP32 | Op-defined | CPU Reference | See [SDPA Restrictions](#sdpa-restrictions) |
| BlockScale Quantize | FP32, FP16, BFP16 in; FP8, FP6, FP4 out | Op-defined | CPU Reference | See [Block-Scale Quantization](#block-scale-quantization) |
| BlockScale Dequantize | FP8, FP6, FP4 in; FP32, FP16, BFP16 out | Op-defined | CPU Reference | See [Block-Scale Quantization](#block-scale-quantization) |
| Pointwise Unary | FP16, BFP16, FP32 | All | CPU Reference |  |
| Pointwise Binary | FP16, BFP16, FP32 | All | CPU Reference |  |
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
| SDPA Forward | `SdpaFwdPlanBuilder` | `SdpaFwdSignatureKey` | Scaled dot-product attention forward pass |
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
| Binary Operations | `PointwisePlanBuilder` | `PointwiseSignatureKey` | ADD, SUB, MUL, MAX, MIN, CMP_GT, RELU_BWD, SIGMOID_BWD, TANH_BWD |
| Ternary Operations | `PointwisePlanBuilder` | `PointwiseSignatureKey` | BINARY_SELECT (BOOLEAN mask) |

`CMP_GT` outputs must have their tensor data type explicitly set to `BOOLEAN`; generic
pointwise output inference otherwise uses the graph's intermediate data type.

## Operation Restrictions

### Reduction Modes

`ADD`, `AVG`, `AMAX`, `NORM1`, `NORM2`, `MUL`, `MUL_NO_ZEROS`, `MIN_OP`, `MAX_OP`. Input and output must
have the same rank; each output dimension must equal the input dimension or be 1, and at least one
dimension must be reduced.

### SDPA Restrictions

| Feature | SDPA Forward | SDPA Backward |
|---------|--------------|---------------|
| Log-sum-exp `stats` output | Supported | Required input |
| Variable sequence lengths (`seq_len_q` / `seq_len_kv`) | Not supported | Not supported |
| Dropout | Not supported | Not supported |
| ALiBi / padding mask | Not supported | - |
| Paged KV cache, block-sparse attention | Not supported | - |
| Softmax max / sum-exp stats outputs | Not supported | - |
| Ragged tensors | - | Rejected |
| FP8 inputs | Requires Q/K/V descale tensors; descales rejected for non-FP8 inputs | - |

FP8 forward is registered for `FP8_E4M3` and `FP8_E4M3_FNUZ` inputs with `BFP16` output.

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
BatchNorm (all four variants), Layernorm (forward/backward), RMSNorm (forward/backward), Pointwise and
SDPA Forward (see `ReferenceOpCoverage.cpp` in the integration test project). SDPA bundles are
additionally limited to `quick`-tier bundles with a bounded working set, to keep CPU time down. See the
[integration test docs](../../../dnn-providers/integration-tests/docs/running-tests.md#verification-modes)
for the verification modes.

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
