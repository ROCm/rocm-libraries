# hipDNN Operation Support

This document provides an overview of operation support in hipDNN and guides you to detailed information about specific plugin implementations.

## About hipDNN Operation Support

hipDNN is a plugin-based deep learning library that provides graph-based operation support through various backend plugins. Each plugin implements specific operations with support for different datatypes, layouts, and features.

> [!IMPORTANT]
> ⚠️ **hipDNN is in the early phase of development.** Operation support continues to expand as the library matures. Check individual plugin documentation for current support details.

## Plugin-Specific Operation Support

hipDNN operations are implemented through plugins. Each plugin provides its own set of supported operations. For detailed information about what operations are available, please refer to the plugin-specific documentation:

### Available Plugins

- **[MIOpen Provider Plugin](../../../dnn-providers/miopen-provider/docs/OperationSupport.md)** - Integration with AMD's MIOpen library for GPU-accelerated deep learning operations
  - Convolution operations (Forward, Dgrad, Wgrad)
  - Batchnorm operations (Training, Backward, Inference)
  - Fused operation graphs
  - Standalone pointwise activation
  - Standalone pointwise binary elementwise (ADD, SUB, MUL, MAX_OP, MIN_OP; the frontend spells the last two MAX and MIN)

- **[hipBLASLt Provider Plugin](../../../dnn-providers/hipblaslt-provider/docs/OperationSupport.md)** - Integration with AMD's hipBLASLt library that provides optimized GEMM operations.
  - Stand-alone Matmul (FP32/FP16/BF16 inputs and outputs, with optional fused bias and activation)
  - MX BlockScaleDequantize + GEMM (FP8 OCP, OCP MX FP6, and OCP MX FP4 inputs, including mixed pairs; gfx950/gfx1250 only; off by default, enable with `-DHIPBLASLTPROVIDER_ENABLE_MX_GEMM=ON`)

- **[HIP Kernel Provider Plugin](../../../dnn-providers/hip-kernel-provider/README.md)** - Custom HIP kernels compiled at run time with HIPRTC, plus assembly SDPA kernels. This provider has no separate operation support document.
  - HIP MLops engine: Batchnorm, Layernorm and RMSNorm (forward and backward), Resample (forward and backward)
  - ASM SDPA engine: SDPA forward and backward
  - Optional engines, off by default: HIP Flash2 SDPA forward (`ENABLE_HIP_FLASH2_ENGINE`) and the kernel ingestor engine (`HIPDNN_ENABLE_KERNEL_INGESTOR`, RFC 0017)

### Reference Implementation

- **[CPU Reference Implementation](./OperationSupport-ReferenceImpl.md)** - CPU-based reference implementation for validation and testing
  - Provides ground-truth results for validating GPU implementations
  - Supports Convolution, Batchnorm, Matmul, Layernorm, RMSNorm, Reduction, Resample, SDPA, block-scale quantization, MoE grouped matmul and Pointwise
  - Not intended for performance or production use

- **[GPU Reference Implementation](../../../dnn-providers/integration-tests/docs/OperationSupport-GpuReferenceImpl.md)** - GPU-based reference implementation in the integration test project
  - Used as the oracle for graph-only integration test bundles
  - Supports a subset of the CPU operations (e.g. Convolution forward, Batchnorm forward, Layernorm, RMSNorm, Reduction, SDPA forward, Matmul, Pointwise)
  - Not intended for performance or production use

> [!NOTE]
> The CPU and GPU references do not accept the same pointwise graphs:
>
> - **Modes.** Only the CPU reference takes `MAX_OP`, `MIN_OP`, `CMP_GT` and `BINARY_SELECT`. Only the GPU
>   reference takes `IDENTITY`.
> - **Mixed datatypes.** Only the CPU reference takes an FP32 operand beside a 16-bit one, and FP16 or BFP16
>   compute types. Only the GPU reference takes unary conversions that widen a 16-bit type or go between FP16 and
>   BFP16, and binary ops that mix FP16 with BFP16.
> - **Compute type.** The CPU reference evaluates in `double` and rounds once to the output type, whatever the
>   node's compute type says. The GPU reference computes in the node's compute type, which must be FP32.
> - **Parameters.** `RELU_FWD` and `RELU_BWD` use `relu_lower_clip`, `relu_upper_clip` and
>   `relu_lower_clip_slope`, and `SWISH_FWD` uses `swish_beta`. On any other mode a parameter makes the CPU
>   reference throw and the GPU reference ignore it. `elu_alpha` and `softplus_beta` have no mode on either
>   reference: the CPU reference throws when it builds the plan, the GPU reference ignores them.
> - **Rank.** The GPU reference addresses at most 5 dimensions and declines a deeper pointwise node. The CPU
>   reference has no rank limit and runs it.
>
> The full datatype tuples, parameter behavior and shape rules are in the
> [CPU](./OperationSupport-ReferenceImpl.md#pointwise-restrictions) and
> [GPU](../../../dnn-providers/integration-tests/docs/OperationSupport-GpuReferenceImpl.md#pointwise-modes)
> documents. `TestPointwiseReferenceParity` (integration test project) fails when either reference gains or
> loses a mode or datatype tuple without the difference being recorded there.

## Roadmap

For information about planned features, upcoming operations, and the development roadmap, please see:

- **[hipDNN Roadmap](./Roadmap.md)** - Comprehensive roadmap covering all hipDNN components, including planned plugin enhancements and new operation support

## Contributing

We welcome contributions to expand operation support in hipDNN!

For detailed contribution guidelines, please see:

- **[Contributing Guide](../CONTRIBUTING.md)** - Complete guide to contributing to hipDNN
- **[Plugin Development](./PluginDevelopment.md)** - Guide for creating and extending plugins

### Getting Started

1. Review the [hipDNN Design](./Design.md) to understand the architecture
2. Check the [Roadmap](./Roadmap.md) for planned features and contribution opportunities
3. Open a GitHub issue to discuss your planned contribution
4. Follow the [Contributing Guide](../CONTRIBUTING.md) for code quality and testing requirements
