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
| Convolution Dgrad   | FP16, BFP16, FP32 | NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | Cross-correlation only<sup>2,6</sup>, Deterministic<sup>5</sup>, Applicability limits<sup>10</sup> |
| Convolution Forward | FP16, BFP16, FP32 | NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | Cross-correlation only<sup>2,6</sup>, Deterministic<sup>5</sup>, Applicability limits<sup>10</sup> |
| Convolution Forward + (Bias) + Activation<sup>4</sup> | FP16, BFP16, FP32 | NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | Fused graph<sup>2,3,6</sup>, Deterministic<sup>5</sup>, Applicability limits<sup>10</sup> |
| Convolution Wgrad   | FP16, BFP16, FP32 | NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | Cross-correlation only<sup>2,6</sup>, Deterministic<sup>5</sup>, Applicability limits<sup>10</sup> |
| Pointwise Activation (standalone) | FP16, FP32 | Rank 1-4, any dimension order | Single-node graph<sup>7</sup> |
| Pointwise Binary Elementwise (standalone) | FP16, FP32 | NCHW, NCDHW | Single-node graph<sup>8</sup> |
| Reduction (standalone) | FP16, FP32 | Any, rank 1-5 | Single-node graph<sup>9</sup> |

¹ See Batchnorm Operations note below
² See Convolution Operations note below
³ See Fused Operations note below
⁴ See Detailed Requirements below
⁵ See Deterministic Engine Support section
⁶ 3D tensors are internally padded to 4D for MIOpen compatibility. For convolution, the padding also extends the padding, stride and dilation vectors with a trailing spatial dimension of length 1.
⁷ See Standalone Activation note below
⁸ See Standalone Binary Elementwise note below
⁹ See Standalone Reduction note below
¹⁰ See Convolution Applicability note below

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
- Bias operand (the ADD operand that is not the convolution output)
    - Dims, rank and channel stride: see Bias, output and tensor requirements below
    - Data type: the input data type
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
> **Convolution Applicability:** The plugin declines the convolution graphs below. A declined graph falls to another engine, or no engine is found for it. These are limits of the plugin's MIOpen-backed convolution engines, not errors in the graph: for graphs of these kinds MIOpen can return wrong output or fault. The rules are conservative and also decline some graphs MIOpen computes correctly. The data type, layout, packing and size rules apply to the plain Convolution Forward, Dgrad and Wgrad rows; the fused row is subject to the last rule only.
> - **Data type (plain Convolution Forward, Dgrad and Wgrad):** the three tensors (`x`, `w`, `y`; `dy`, `w`, `dx`; `x`, `dy`, `dw`) must share one data type, FP32, FP16 or BFP16.
> - **Layout (plain Convolution Forward, Dgrad and Wgrad):** the three tensors must share one layout: all channels-first (NCL, NCHW, NCDHW) or all channels-last (NLC, NHWC, NDHWC). A packed tensor is judged by its exact canonical strides, including those of extent-1 dimensions; a non-packed tensor is judged by its stride order. Forward also accepts three packed tensors whose strides, read in one layout's order, never increase, extent-1 dimensions included: for example a 1x1 convolution on an `[N, C, 1, 1]` channels-first input with a channels-last output, or a batch-1 input sliced from a wider channel dimension.
> - **Packing, plain Convolution Forward:** a non-packed tensor (the strides do not address exactly the element count) is declined in a grouped convolution and in a 3D channels-last convolution. Any tensor, packed or not, is declined when its strides may let two elements share an address (a zero, negative or overlapping stride; the check is conservative). Other non-packed tensors, such as padded rows or a channel slice, are accepted.
> - **Packing, plain Convolution Dgrad and Wgrad:** a channels-first non-packed tensor is accepted; a channels-last non-packed tensor is declined.
> - **Size, plain Convolution Dgrad:** `dy` and `dx` must each have at most 2^31 - 1 elements.
> - **Fused Convolution Forward + (Bias) + Activation:** none of the rules above is applied to its `x`, `w` and `y`. The graph is declined when the bias operand does not meet the requirements under Detailed Requirements (dims, channel stride and data type), or when the final activation output does not have the data type of `x` and the dims of the convolution output, packed in the same layout as `x`. An ADD with any other operand, for example a residual of the output's shape, is not treated as a bias and the graph is declined.

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
> **Standalone Reduction:** A single-node reduction graph supports ADD, MUL, MIN, MAX, AMAX, AVG,
> NORM1 and NORM2 through MIOpen's `miopenReduceTensor`, with an FP32 compute type. Input and
> output share one datatype (FP16 or FP32) and one rank (1-5); each output dimension is either 1
> (reduced) or equal to the input dimension, and at least one dimension must be reduced. Any set
> of dimensions may be reduced, including the innermost. Strides must be positive; on the output,
> the axes longer than 1, taken by increasing stride, must each have a stride larger than the
> furthest offset the smaller axes reach (packed and padded layouts qualify). An interleaved
> output layout that does not overlap but breaks this rule, such as dims [3, 2] with strides
> [2, 3], is declined. Not supported: MUL_NO_ZEROS (MIOpen has no equivalent), BFP16 (MIOpen's
> legacy reduce kernels have no bf16 support), deterministic reduction (`is_deterministic`),
> in-place execution, pass-by-value tensors (a runtime host scalar or a value baked into the
> graph), and tensors whose lengths, strides, element count or memory extent in bytes exceed
> INT32_MAX. A reduction to a single output element is further limited to 1,073,741,568 (about
> 2^30) input elements.

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

> [!NOTE]
> **Reduction:** Reduction is only available through the default (non-deterministic) engine. A reduction node that sets `is_deterministic` is not supported.

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
