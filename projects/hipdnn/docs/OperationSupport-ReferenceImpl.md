# hipDNN CPU Reference Implementation - Operation Support

This document provides a comprehensive overview of the operations currently supported in the hipDNN CPU Reference Implementation and the details of their implementation support.

> [!IMPORTANT]
> ⚠️ **The CPU Reference Implementation is for testing and validation purposes only.** This implementation provides ground-truth results for validating GPU implementations and is not optimized for performance.

## Purpose

The CPU Reference Implementation serves as:
- **Reference Implementation**: Provides ground-truth results for validating GPU implementations
- **Testing Infrastructure**: Enables comprehensive testing of graph execution
- **Validation Tool**: Ensures correctness of GPU-accelerated operations

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
| Matmul | FP16, BFP16, FP32 | NCHW, NCDHW | CPU Reference |  |
| Pointwise Unary | FP16, BFP16, FP32 | All | CPU Reference |  |
| Pointwise Binary | FP16, BFP16, FP32 | All | CPU Reference |  |

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

### Pointwise Operations

| Operation Type | Plan Builder | Signature Key | Supported Operations |
|----------------|-------------|---------------|---------------------|
| Unary Operations | `PointwisePlanBuilder` | `PointwiseSignatureKey` | RELU_FWD, SIGMOID_FWD, TANH_FWD, ABS, NEG |
| Binary Operations | `PointwisePlanBuilder` | `PointwiseSignatureKey` | ADD, SUB, MUL, RELU_BWD, SIGMOID_BWD, TANH_BWD |
| Ternary Operations | `PointwisePlanBuilder` | `PointwiseSignatureKey` | None Supported Yet |

## Legend

### Datatypes
- **FP16**: Half-precision floating point (16-bit)
- **BFP16**: Brain floating point (16-bit)
- **FP32**: Single-precision floating point (32-bit)

### Layouts
- **NCHW**: Batch, Channels, Height, Width (2D, channel-first)
- **NHWC**: Batch, Height, Width, Channels (2D, channel-last)
- **NCDHW**: Batch, Channels, Depth, Height, Width (3D, channel-first)
- **NDHWC**: Batch, Depth, Height, Width, Channels (3D, channel-last)

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
- [CPU Graph Executor Design](./rfcs/0001_CpuGraphExecutorDesign.md) - Detailed architecture documentation
- [Testing Plan](./testing/TestPlan.md) - Testing strategy and procedures
