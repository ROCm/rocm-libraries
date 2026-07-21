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
| Convolution Dgrad   | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | Cross-correlation only<sup>2</sup>, Deterministic<sup>5</sup> |
| Convolution Forward | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | Cross-correlation only<sup>2</sup>, Deterministic<sup>5</sup> |
| Convolution Forward + (Bias) + Activation<sup>4</sup> | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | Fused graph<sup>2,3</sup>, Deterministic<sup>5</sup> |
| Convolution Wgrad   | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | Cross-correlation only<sup>2</sup>, Deterministic<sup>5</sup> |

¹ See Batchnorm Operations note below
² See Convolution Operations note below
³ See Fused Operations note below
⁴ See Detailed Requirements below
⁵ See Deterministic Engine Support section
⁶ 3D tensors are internally padded to 4D for MIOpen compatibility

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
    - no clipping
    - relu_lower_clip set
    - relu_lower_clip and relu_upper_clip set

## Operation Notes

> [!NOTE]
> **Batchnorm Operations:** Currently, only spatial batchnorm mode is supported. Spatial mode computes statistics over the batch (N) and spatial dimensions (H, W, or D, H, W) for each channel.

> [!NOTE]
> **Convolution Operations:** Currently, only cross-correlation convolutions are supported. True mathematical convolution (with kernel flipping) is not yet implemented. In practice, cross-correlation is the standard operation used in modern deep learning frameworks.

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
> **Activation Functions:** Supports ReLU, Clipped ReLU (with configurable upper clip), and CLAMP (with configurable lower/upper clips).

> [!NOTE]
> **Sparse Support:** All operations currently work with dense tensors only.

> [!NOTE]
> **Batchnorm Training Running Statistics:** Batchnorm training supports updating running statistics using separate read and write buffers. The previous running statistics (`prev_running_mean`, `prev_running_variance`) are read-only inputs, while the next running statistics (`next_running_mean`, `next_running_variance`) are write-only outputs. The exponential moving average is computed as: `next_running = (1 - momentum) * prev_running + momentum * batch_statistic`.

## Deterministic Engine Support

The MIOpen provider offers a deterministic execution engine (`MIOPEN_ENGINE_DETERMINISTIC`) for convolution operations. This engine guarantees bit-reproducible results across multiple executions with the same inputs.

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
