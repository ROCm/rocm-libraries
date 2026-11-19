# MIOpen Provider Plugin - Operation Support

This document provides detailed information about the operations supported by the MIOpen Provider Plugin for hipDNN.

For general information about hipDNN's operation support, please see the [hipDNN Operation Support](../../../docs/OperationSupport.md) documentation.

## Current Operation Support

The following table lists all operations currently supported in hipDNN:

| Operation | Datatypes | Layouts | Notes |
|-----------|-----------|---------|-------|
| Batchnorm Inference | FP16, BFP16, FP32 | NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | Spatial mode only<sup>1,6</sup> |
| Batchnorm Inference + Activation | FP16, BFP16, FP32 | NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | Fused graph<sup>3,6</sup> |
| Batchnorm Inference with Variance | FP16, BFP16, FP32 | NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | Spatial mode only<sup>1,6</sup> |
| Batchnorm Inference with Variance + Activation | FP16, BFP16, FP32 | NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | Fused graph<sup>3,6</sup> |
| Batchnorm Inference + DRelu + Backward | FP16, BFP16, FP32 | NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | Fused graph<sup>3,6</sup> |
| Batchnorm Training  | FP16, BFP16, FP32 | NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | Spatial mode only<sup>1,6</sup> |
| Batchnorm Training + Activation | FP16, BFP16, FP32 | NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | Fused graph<sup>3,6</sup> |
| Batchnorm Backward  | FP16, BFP16, FP32 | NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | Spatial mode only<sup>1,6</sup> |
| Convolution Dgrad   | FP16, BFP16, FP32 | NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | Cross-correlation only<sup>2,6</sup>, Deterministic<sup>5</sup> |
| Convolution Forward | FP16, BFP16, FP32 | NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | Cross-correlation only<sup>2,6</sup>, Deterministic<sup>5</sup> |
| Convolution Forward + (Bias) + Activation<sup>4</sup> | FP16, BFP16, FP32 | NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | Fused graph<sup>2,3,6</sup>, Deterministic<sup>5</sup> |
| Convolution Wgrad   | FP16, BFP16, FP32 | NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | Cross-correlation only<sup>2,6</sup>, Deterministic<sup>5</sup> |
| Pointwise Activation (standalone) | FP16, FP32 | Rank 1-4, any dimension order | Single-node graph<sup>7</sup> |
| Pointwise Binary Elementwise (standalone) | FP16, FP32 | NCHW, NCDHW | Single-node graph<sup>8</sup> |

¹ See Batchnorm Operations note below
² See Convolution Operations note below
³ See Fused Operations note below
⁴ See Detailed Requirements below
⁵ See Deterministic Engine Support section
⁶ 3D tensors are internally padded to 4D for MIOpen compatibility. For convolution, the padding also extends the padding, stride and dilation vectors with a trailing spatial dimension of length 1.
⁷ See Standalone Activation note below
⁸ See Standalone Binary Elementwise note below

## Detailed Requirements

### Convolution Forward + (Bias) + Activation
#### Convolution forward node
- Compute data type: FP32
- Y tensor
    - Virtual
    - Data type:  FP32 or the input data type (the latter only if bias is used)
#### Bias node (optional)
- Compute data type: input data type
- Output tensor
    - Virtual
    - Data type: FP32 or the input data type
#### Activation node
- Compute data type: FP32
- Activation mode: RELU_FORWARD
- Supports
    - standard ReLU: no clips set, or relu_lower_clip = 0
    - clipped ReLU: relu_upper_clip set
    - CLAMP: relu_lower_clip and relu_upper_clip set
- relu_lower_clip_slope must be unset or 0 (leaky ReLU is not supported when fused)
#### Bias, output and tensor requirements
- Bias dims, after left-padding with 1 to the convolution output rank, are `[1, K, 1, ...]` where K is the output channel count
- Bias rank may not exceed the convolution output rank
- Bias channel stride is 1 when K > 1
- Final output tensor
    - Data type: the input data type
    - Dims: the convolution output dims
    - Packed in the input's layout; strides of unit dimensions are included in the comparison
- Pass-by-value tensors are not supported

## Operation Notes

> [!NOTE]
> **Batchnorm Operations:** Currently, only spatial batchnorm mode is supported. Spatial mode computes statistics over the batch (N) and spatial dimensions (H, W, or D, H, W) for each channel.

> [!NOTE]
> **Convolution Operations:** Currently, only cross-correlation convolutions are supported. True mathematical convolution (with kernel flipping) is not yet implemented. In practice, cross-correlation is the standard operation used in modern deep learning frameworks. Pass-by-value tensors are not supported.

> [!NOTE]
> **Fused Operations:**
> - Fused batchnorm operations require F32 data type for the virtual/intermediate tensors. This can be set either explicitly on the tensor or by setting the default intermediate data type for the graph (recommended).
> - The fused graph patterns combine multiple operations:
>   - **Batchnorm Inference + Activation:** Combines batchnorm inference (using invVariance) with forward activation (ReLU)
>   - **Batchnorm Inference with Variance + Activation:** Combines batchnorm inference (using variance) with forward activation (ReLU)
>   - **Batchnorm Inference + DReLU + Backward:** Combines batchnorm inference, activation backward (DReLU), and batchnorm backward
>   - **Batchnorm Training + Activation:** Combines batchnorm training with forward activation
>   - **Convolution Forward + (Bias) + Activation:** Combines convolution forward, optional bias addition, and forward activation

> [!NOTE]
> **Activation Functions (fused):** Supports ReLU, Clipped ReLU (with configurable upper clip), and CLAMP (with configurable lower/upper clips). Leaky ReLU (non-zero `relu_lower_clip_slope`), Sigmoid and Tanh are **not** supported when fused onto a producer op; use a standalone activation node for those.

> [!NOTE]
> **Standalone Activation:** A single-node pointwise graph supports ReLU, Clipped ReLU (configurable upper clip), CLAMP (configurable lower/upper clips), Leaky ReLU (configurable negative slope), Sigmoid and Tanh. Tensors have rank 1-4. `x` and `y` have identical dims and are both packed with the same layout, in any dimension order; strides of unit dimensions are ignored. A padded, sliced, broadcast or otherwise non-packed tensor is declined because of [MIOpen issue 16024](https://github.com/ROCm/rocm-libraries/issues/16024).

> [!NOTE]
> **ReLU parameters:** An unset `relu_lower_clip` is treated as 0, an unset `relu_upper_clip` as no
> upper bound, and an unset `relu_lower_clip_slope` or a slope of exactly 0 as no leaky slope, so
> equivalent encodings select the same MIOpen activation. A non-zero `relu_lower_clip` without an
> upper clip is not supported.

> [!NOTE]
> **Standalone Binary Elementwise:** A single-node pointwise graph supports ADD, SUB, MUL, MAX and MIN
> across ranks 3-5, with packed channels-first strides and FP16/FP32 datatypes only. `in_0` must be
> the full-shape operand; only `in_1` may broadcast per-axis at matching rank, and all three tensors
> must share one datatype. SUB is executed internally as Add with `in_1` scaled by −1. In-place
> execution is not supported.

> [!NOTE]
> **Sparse Support:** All operations currently work with dense tensors only.

> [!NOTE]
> **Batchnorm Training Running Statistics:** Batchnorm training supports updating running statistics using separate read and write buffers. The previous running statistics (`prev_running_mean`, `prev_running_variance`) are read-only inputs, while the next running statistics (`next_running_mean`, `next_running_variance`) are write-only outputs. The exponential moving average is computed as: `next_running = (1 - momentum) * prev_running + momentum * batch_statistic`. The five running-statistics operands (prev/next running mean, prev/next running variance, momentum) must be given together, and saved mean and inv_variance must be given together or not at all; any other combination is declined. Training with running statistics or a fused activation also requires the saved mean / inv_variance pair; this is a limitation of this provider's plan, not of MIOpen, and is tracked as a follow-up. A saved mean and inv_variance that are both virtual are treated as not requested, unless running statistics are present; exactly one virtual is declined, and with running statistics or a fused activation the pair must be non-virtual. Only epsilon and momentum may be pass-by-value; every other batchnorm tensor is read from or written to a device buffer, and a pass-by-value tensor in such a role is declined. Momentum is a pass-by-value scalar and is not type-validated by the provider. The consumed saved and running statistics tensors must be FLOAT with shape `[1, C, 1, ...]`, validated against the batchnorm channel layout. In a single-node batchnorm graph (inference, training or backward) every consumed tensor, including the running statistics, must be non-virtual; a virtual tensor is declined.

## Deterministic Engine Support

The MIOpen provider offers a deterministic execution engine (`MIOPEN_ENGINE_DETERMINISTIC`) for convolution, unary activation, and binary pointwise operations. This engine guarantees bit-reproducible results across multiple executions with the same inputs.

To use the deterministic engine, set it as the preferred engine on your graph before building:

```cpp
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

graph.set_preferred_engine_id_ext(MIOPEN_ENGINE_DETERMINISTIC_NAME);
```

> [!NOTE]
> **Batchnorm Operations:** Batchnorm operations do not support deterministic execution in MIOpen and are only available through the default (non-deterministic) engine.

## Legend

### Datatypes
- **FP16**: Half-precision floating point (16-bit)
- **BFP16**: Brain floating point (16-bit)
- **FP32**: Single-precision floating point (32-bit)

### Layouts
- **NCL**: Batch, Channels, Length (1D, channel-first)
- **NLC**: Batch, Length, Channels (1D, channel-last)
- **NCHW**: Batch, Channels, Height, Width (2D, channel-first)
- **NHWC**: Batch, Height, Width, Channels (2D, channel-last)
- **NCDHW**: Batch, Channels, Depth, Height, Width (3D, channel-first)
- **NDHWC**: Batch, Depth, Height, Width, Channels (3D, channel-last)
