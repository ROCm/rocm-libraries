# hipDNN GPU Reference Implementation - Operation Support

This document provides an overview of the operations currently supported by the GPU reference
implementation in the hipDNN integration test project (`dnn-providers/integration-tests/`) and the
details of their implementation support.

> [!IMPORTANT]
> ⚠️ **The GPU Reference Implementation is for testing and validation purposes only.** It exists to
> produce ground-truth results for the integration tests, favors simplicity over performance, and
> is not a production engine.

> [!NOTE]
> A CPU-based reference implementation lives in the hipDNN Test SDK. See
> [CPU Reference Implementation - Operation Support](../../../projects/hipdnn/docs/OperationSupport-ReferenceImpl.md) for the operations it covers.
> The two implementations cover different sets of operations; see
> [CPU and GPU Coverage Comparison](#cpu-and-gpu-coverage-comparison).

## Purpose

The GPU Reference Implementation serves as:
- **Oracle for graph-only bundles**: Provides the expected output when a bundle carries no golden data
- **Fast alternative to the CPU reference**: Runs the same shapes in milliseconds where the scalar CPU reference takes seconds to minutes
- **Device-side validation**: Compares engine output with the reference on the GPU, without copying large tensors to the host

## Architecture

| Component | Location | Role |
|-----------|----------|------|
| `gpu-ref` library | `gpu-ref/` | `GpuFpReference*` classes that launch the reference kernels. Kernel sources (`gpu-ref/kernels/`) are embedded at build time and compiled with hipRTC on first use; compiled kernels are cached per source, defines and function. |
| GPU graph executor | `src/harness/gpu-graph-executor/` | `GpuReferenceGraphExecutor` executes a serialized graph node by node. A node is accepted only if a plan builder is registered for its **signature key** (operation + datatypes) and that builder's `isApplicable` check passes. |
| Validators | `gpu-ref/include/hipdnn-gpu-ref/` | Device-side tensor comparison: `GpuFpReferenceValidation` (allclose), `GpuFpReferenceRmsValidation` (RMS error) and `GpuIntReferenceValidation` (integer), created through `GpuReferenceValidationFactory`. |

The executor requires device memory (`requiresDeviceMemory()` is `true`). A node type the executor does
not handle raises a `ReferenceCapabilityError`; the harness treats that as "this reference declines the
graph" and moves on to the next oracle.

### Selecting the GPU reference

| Mechanism | Behavior |
|-----------|----------|
| `--verification-mode gpu` (`HIPDNN_TEST_VERIFICATION_MODE`) | Verify bundles on the GPU reference only; **FAIL** if it cannot run the op |
| `--verification-mode auto` (default) | golden, then GPU reference, then CPU reference |
| `--reference-executor gpu\|cpu` | Selects the reference for C++ graph tests |
| `--validator auto\|cpu\|gpu` | Selects where the comparison runs. `auto` follows the reference, so GPU-reference output is compared on the device |

See the [integration test README](../README.md#verification-modes) for the full description.

## Current Operation Support

The following table lists the operations currently executable through the GPU graph executor:

| Operation | Datatypes | Layouts | Implementation | Notes |
|-----------|-----------|---------|----------------|-------|
| Convolution Forward | FP16, BFP16, FP32 | Stride-based | GPU Reference | 1D, 2D and 3D spatial; FP32 output with FP16/BFP16 inputs also registered |
| Matmul | FP16, BFP16, FP32 | Stride-based | GPU Reference | FP32 inputs with FP16/BFP16 output also registered |
| BatchNorm Forward Inference | FP16, BFP16, FP32 | Stride-based | GPU Reference | Mean and variance are required inputs |
| BatchNorm Forward Inference with Variance | FP16, BFP16, FP32 | Stride-based | GPU Reference | Same signatures as BatchNorm Forward Inference |
| BatchNorm Forward Training | FP16, BFP16, FP32 | Stride-based | GPU Reference | Mean/variance outputs optional |
| Layernorm Forward | FP16, BFP16, FP32 | Stride-based | GPU Reference | Mean/inv-variance outputs optional |
| Layernorm Backward | FP16, BFP16, FP32 | Stride-based | GPU Reference | Mean/inv-variance optional |
| RMSNorm Forward | FP16, BFP16, FP32 | Stride-based | GPU Reference |  |
| RMSNorm Backward | FP16, BFP16, FP32 | Stride-based | GPU Reference |  |
| Reduction | FP16, BFP16, FP32 | Stride-based | GPU Reference | FP32 or FP64 compute; see [Reduction Modes](#reduction-modes) |
| SDPA Forward (dense) | FP16, BFP16, FP32 | Op-defined | GPU Reference | See [SDPA Restrictions](#sdpa-restrictions) |
| SDPA Forward (ragged) | BFP16, FP8 E4M3 | Op-defined | GPU Reference | Selected when a Q/K/V/O tensor carries a `ragged_offset` |
| Pointwise Unary | FP16, BFP16, FP32 | All | GPU Reference | See [Pointwise Modes](#pointwise-modes) |
| Pointwise Binary | FP16, BFP16, FP32 | All | GPU Reference | See [Pointwise Modes](#pointwise-modes) |

### Not Yet Supported by the GPU Graph Executor

| Operation | Executor behavior | `gpu-ref` library |
|-----------|-------------------|-------------------|
| Convolution Backward Data | `ReferenceCapabilityError` ("GPU plan not yet implemented") | `GpuFpReferenceConvolution::dgrad` exists |
| Convolution Backward Weights | `ReferenceCapabilityError` | `GpuFpReferenceConvolution::wgrad` exists |
| BatchNorm Backward | `ReferenceCapabilityError` | `GpuFpReferenceBatchnorm::backward` exists |
| SDPA Backward | `ReferenceCapabilityError` | None |
| BlockScale Quantize / Dequantize | `ReferenceCapabilityError` | None |
| Resample Forward | "Unsupported node type" error | `GpuFpReferenceResample::forward` exists |
| Resample Backward | "Unsupported node type" error | None |
| MoE Grouped Matmul (forward and backward) | "Unsupported node type" error | None |
| Pointwise ternary (`BINARY_SELECT`), `MAX`, `MIN`, `CMP_GT` | Not applicable (plan builders reject the mode) | Not implemented |
| Custom operations | `ReferenceCapabilityError` | None |

Operations with a `gpu-ref` library entry but no executor plan can be called directly from C++ code
but are not reachable through graph-level verification.

## Implementation Details

### Convolution and Matmul Operations

| Operation | Plan Builder | Signature Key | Description |
|-----------|-------------|---------------|-------------|
| Convolution Forward | `GpuConvolutionFwdPlanBuilder` | `GpuConvolutionFwdSignatureKey` | Forward convolution |
| Matmul | `GpuMatmulPlanBuilder` | `GpuMatmulSignatureKey` | Generic matrix multiplication |

### BatchNormalization Operations

| Operation | Plan Builder | Signature Key | Description |
|-----------|-------------|---------------|-------------|
| BatchNorm Forward Inference | `GpuBatchnormFwdInfPlanBuilder` | `GpuBatchnormFwdInfSignatureKey` | Inference-mode forward pass |
| BatchNorm Forward Inference with Variance | `GpuBatchnormFwdInfVariancePlanBuilder` | `GpuBatchnormFwdInfVarianceSignatureKey` | Inference-mode with variance forward pass |
| BatchNorm Forward Training | `GpuBatchnormFwdTrainPlanBuilder` | `GpuBatchnormFwdTrainSignatureKey` | Training-mode forward pass with statistics |

### Normalization Operations

| Operation | Plan Builder | Signature Key | Description |
|-----------|-------------|---------------|-------------|
| Layernorm Forward | `GpuLayernormFwdPlanBuilder` | `GpuLayernormFwdSignatureKey` | Layer normalization forward pass |
| Layernorm Backward | `GpuLayernormBwdPlanBuilder` | `GpuLayernormBwdSignatureKey` | Layer normalization gradients |
| RMSNorm Forward | `GpuRMSNormFwdPlanBuilder` | `GpuRMSNormFwdSignatureKey` | RMS normalization forward pass |
| RMSNorm Backward | `GpuRMSNormBwdPlanBuilder` | `GpuRMSNormBwdSignatureKey` | RMS normalization gradients |

### Reduction and SDPA Operations

| Operation | Plan Builder | Signature Key | Description |
|-----------|-------------|---------------|-------------|
| Reduction | `GpuReductionPlanBuilder` | `GpuReductionSignatureKey` | Reduction over the requested dimensions |
| SDPA Forward (dense) | `GpuSdpaFwdPlanBuilder` | `GpuSdpaFwdSignatureKey` | Scaled dot-product attention forward pass |
| SDPA Forward (ragged) | `GpuSdpaRaggedFwdPlanBuilder` | `GpuSdpaRaggedFwdSignatureKey` | Packed (ragged) attention forward pass |

### Pointwise Operations

| Operation Type | Plan Builder | Signature Key | Supported Operations |
|----------------|-------------|---------------|---------------------|
| Unary Operations | `GpuPointwiseUnaryPlanBuilder` | `GpuPointwiseSignatureKey` | IDENTITY, ABS, NEG, RELU_FWD, SIGMOID_FWD, TANH_FWD, GELU_FWD, GELU_APPROX_TANH_FWD, SWISH_FWD |
| Binary Operations | `GpuPointwiseBinaryPlanBuilder` | `GpuPointwiseSignatureKey` | ADD, SUB, MUL, RELU_BWD, SIGMOID_BWD, TANH_BWD |

## Operation Restrictions

### Pointwise Modes

The GPU executor accepts a subset of the pointwise modes the CPU reference implements.

| Group | GPU modes | Modes the CPU reference has that GPU lacks |
|-------|-----------|---------------------------------------------|
| Unary | IDENTITY, ABS, NEG, RELU_FWD, SIGMOID_FWD, TANH_FWD, GELU_FWD, GELU_APPROX_TANH_FWD, SWISH_FWD | - (GPU additionally supports IDENTITY) |
| Binary | ADD, SUB, MUL, RELU_BWD, SIGMOID_BWD, TANH_BWD | MAX, MIN, CMP_GT |
| Ternary | - | BINARY_SELECT |

Pointwise nodes use `FP32` compute and no `BOOLEAN` tensors. A pointwise node with a runtime
pass-by-value operand is rejected, as are Convolution Forward nodes with such an operand.

### Reduction Modes

`ADD`, `AVG`, `AMAX`, `NORM1`, `NORM2`, `MUL`, `MUL_NO_ZEROS`, `MIN_OP`, `MAX_OP`. Compute type is `FP32` or `FP64`.

### SDPA Restrictions

| Feature | Dense SDPA Forward | Ragged SDPA Forward |
|---------|--------------------|---------------------|
| Log-sum-exp `stats` output | Supported | Supported (must be `FP32`; packed when an `INT32` ragged offset is used) |
| Variable sequence lengths (`seq_len_q` / `seq_len_kv`) | Not supported | Not supported; lengths come from `ragged_offset` alone |
| FP8 (descale) inputs | Not supported; no FP8 tuple registered | `FP8_E4M3` with `BFP16` output; Q/K/V descales only |
| Dropout | Not supported | Not supported |
| ALiBi / padding mask | Not supported | Not supported |
| Paged KV cache, block-sparse attention | Not supported | Not supported |
| Softmax max / sum-exp stats outputs | Not supported | - |
| Additive attention bias (`attn_mask`) | - | Not supported |

The ragged plan requires the RFC-0014 `[B, S, H, D]` token-major layout with a `ragged_offset` aux on every primary. A node is routed to the ragged plan as soon as any of its Q/K/V/O tensors has a `ragged_offset`; a
partly ragged node is rejected there instead of running as dense. The harness records the
dense-path gaps (variable sequence lengths, FP8) in `knownReferenceGaps()`
(`ReferenceOpCoverage.cpp`) so those bundles report "not applicable" instead of failing.

## Validators

`GpuReferenceValidationFactory` creates the validators the harness uses when the GPU reference is the
oracle:

| Factory | Validator | Datatypes |
|---------|-----------|-----------|
| `createGpuAllCloseValidator` | `GpuFpReferenceValidation<T>` (allclose, on the device) | FP32, FP16, BFP16, FP64 |
| `createGpuAllCloseValidator` | `GpuIntReferenceValidation<T>` (exact match) | INT8, UINT8, INT32 |
| `createGpuRmsValidator` | `GpuFpReferenceRmsValidation<T>` (relative RMS error, on the device) | FP32, FP16, BFP16, FP64 |

Unsupported datatypes throw. `[[validator_overrides]]` of type `allclose_matching_infinities` have only a
host validator.

## CPU and GPU Coverage Comparison

| Operation | CPU reference | GPU reference |
|-----------|:-------------:|:-------------:|
| BatchNorm Forward Inference | Yes | Yes |
| BatchNorm Forward Inference with Variance | Yes | Yes |
| BatchNorm Forward Training | Yes | Yes |
| BatchNorm Backward | Yes | No (library only) |
| Convolution Forward | Yes | Yes |
| Convolution Backward Data | Yes | No (library only) |
| Convolution Backward Weights | Yes | No (library only) |
| Matmul | Yes | Yes |
| MoE Grouped Matmul (forward, backward) | Yes | No |
| Layernorm Forward / Backward | Yes | Yes |
| RMSNorm Forward / Backward | Yes | Yes |
| Reduction | Yes | Yes |
| Resample Forward | Yes | No (library only) |
| Resample Backward | Yes | No |
| SDPA Forward | Yes (FP8 E4M3 / E4M3 FNUZ; no variable seq. len., no ragged) | Yes (dense; ragged incl. FP8 E4M3) |
| SDPA Backward | Yes | No |
| BlockScale Quantize / Dequantize | Yes | No |
| Pointwise Unary | Yes (no IDENTITY) | Yes (incl. IDENTITY) |
| Pointwise Binary | Yes (incl. MAX, MIN, CMP_GT) | Yes (ADD, SUB, MUL and the three backward modes) |
| Pointwise Ternary | Yes (BINARY_SELECT) | No |

## Operations Required of the GPU Reference in Bundle Verification

Support above is what the executor *can* run. The bundle harness additionally keeps a required-op set per
reference (`gpuSupportedOps()` in `src/harness/reference-validation/ReferenceOpCoverage.cpp`). A bundle
is verified against the GPU reference only when every node type in its graph is in that set;
otherwise the GPU lane reports `OUTSIDE_OP_SET` for it. The set is currently **Convolution Forward** and
**SDPA Forward**, so it is narrower than the table above. The CPU reference set is documented in
[CPU Reference Implementation - Operation Support](../../../projects/hipdnn/docs/OperationSupport-ReferenceImpl.md#use-in-the-integration-tests).

## Legend

### Datatypes
- **FP16**: Half-precision floating point (16-bit)
- **BFP16**: Brain floating point (16-bit)
- **FP32**: Single-precision floating point (32-bit)
- **FP64**: Double-precision floating point (64-bit), reduction compute type only
- **FP8 E4M3**: 8-bit floating point, ragged SDPA only
- **INT8, UINT8, INT32**: Integer datatypes, validators only

### Layouts
- **Stride-based**: Tensors are addressed through their dims and strides; the reference checks that the tensors of a node use consistent layouts
- **All**: Any stride layout; the operation does not constrain dimension ordering
- **Op-defined**: Dimension ordering is fixed by the operation's definition, not by a layout name

> **Note:** Layout names (NCHW, NHWC, etc.) describe the **memory layout** controlled by strides. See the
> [Porting Guide](../../../projects/hipdnn/docs/PortingGuide.md#tensor-dimensions-and-layouts) for details.

### Implementation
- **GPU Reference**: hipRTC-compiled GPU kernels used as a validation oracle

## Related Documentation

- [Integration Tests README](../README.md) - Bundles, verification modes and validators
- [Support Claim Enforcement](./support-claim-enforcement.md) - How engine support claims are verified
- [CPU Reference Implementation - Operation Support](../../../projects/hipdnn/docs/OperationSupport-ReferenceImpl.md) - CPU counterpart in hipDNN
- [hipDNN Operation Support](../../../projects/hipdnn/docs/OperationSupport.md) - Central hub for hipDNN operation support
