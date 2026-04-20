# hipDNN Samples

## How to Build

1. **Prerequisites:**

   - If you're new to hipDNN, see the [Consumer Quick Start](../docs/ConsumerQuickStart.md) for project setup basics.
   - Follow the instructions in [Building.md](../docs/Building.md) to install the needed dependencies, compilers, and libraries for building hipDNN projects. Specifically:
     * CMake
     * Ninja
     * ROCm (for HIP runtime)
   - A ROCm-compatible GPU is required to run the samples

2. **Build Samples:** From this `samples` directory:
   ```bash
   mkdir build && cd build
   cmake -G Ninja ..
   ninja
   ```
   - Note: If you have installed hipdnn to a custom location you just need to specify the `CMAKE_PREFIX_PATH` to point to the install location.  Ensure you specify the full path and not a relative one.

The sample executables will be created in the `build` directory.

## Running Samples

All samples accept the following command line options:
- `--verify-cpu` - Enable CPU reference validation of results
- `--help` - Displays help message

Example:
```bash
./build/conv_forward --verify-cpu
```

## Profiling Samples

Samples can be profiled using ROCprofiler-SDK (`rocprofv3`), included in ROCm version 6.2 or later.

```bash
# Profile a sample with system trace and perfetto output format
rocprofv3 --sys-trace -f pftrace -o ./profile_output -- ./bn_inference
```

Since `rocprofv3` has limited host-side tracing, `hipDeviceSynchronize()` calls can be added to gauge granular performance.

```cpp
HIP_CHECK(hipDeviceSynchronize()); // Flush previous work and record in trace

HIPDNN_FE_CHECK(graph->execute(handle, variantPack, nullptr));

HIP_CHECK(hipDeviceSynchronize()); // Await completion and record in trace for approximate delta
```

## Available Samples

All samples are templated for mixed-precision execution with Fp32, Fp16, and Bfp16 input/output data types, with Fp32 intermediate accumulation.

> [!TIP]
> 💡 Set `HIPDNN_LOG_LEVEL=info` to observe detailed logs from the samples.

The current samples include:


### [**`BnInference`**](./batchnorm/BnInference.cpp)

Executes a single-node batch normalization inference graph on a 4D input tensor using inverse variance.

- It normalizes each dimension of the input tensor `x` of shape `(N, C, H, W)`, using pre-calculated population statistics (mean and inverse variance). The result is then transformed by the learned parameters `scale` and `bias`, each with shape `(1, C, 1, 1)`:
    ```python
    y = scale * ((x - mean) * inv_variance) + bias
    ```

### [**`BnInferenceWithVariance`**](./batchnorm/BnInferenceWithVariance.cpp)

Executes a single-node batch normalization inference with variance graph on a 4D input tensor.

- It normalizes each dimension of the input tensor `x` of shape `(N, C, H, W)`, using pre-calculated population statistics. The result is then transformed by the learned parameters `scale` and `bias`, each with shape `(1, C, 1, 1)` to enable broadcasting over the batch (N) and spatial (H, W) dimensions. At a high-level, the following element-wise linear transformation is broadcast over the batch and spatial dimensions:
    ```python
    y = scale * ((x - mean) / sqrt(variance + epsilon)) + bias
    ```

### [**`BnTraining`**](./batchnorm/BnTraining.cpp)

Executes the forward pass of a batch normalization training graph on a 4D input tensor.
- For an input `x` of shape `(N, C, H, W)`, the mean and variance are calculated over the `N`, `H`, and `W` dimensions for each of the `C` channels, resulting in a `mean` and `inv_variance` of shape `(1, C, 1, 1)`. It then transforms the input and updates the running statistics:
    ```python
    y = scale * ((x - mean) * inv_variance) + bias
    next_running_mean = (1 - momentum) * prev_running_mean + momentum * batch_mean
    next_running_variance = (1 - momentum) * prev_running_variance + momentum * batch_variance
    ```
- The graph outputs the normalized tensor `y`, along with the batch mean/variance (`mean`, `inv_variance`) required for the backward pass, and the updated population statistics (`next_running_mean`, `next_running_variance`) required for inference.

### [**`FusedBnTrainingActiv`**](./batchnorm/FusedBnTrainingActiv.cpp)

Executes a fused batch normalization training and activation graph.

The fused graph consists of two operations:

1. **Batchnorm Training**: Normalizes input `x` using batch statistics, updates running statistics (optional), and outputs saved mean and inverse variance.
   ```python
   y_bn = scale * ((x - mean) * inv_variance) + bias
   ```

2. **Activation (ReLU)**: Applies ReLU activation
   ```python
   y = relu(y_bn) = max(y_bn, 0)
   ```

**Key Features:**
- Demonstrates fusion of batch normalization training and activation
- Supports both full training (updating running stats) and batch-stats-only modes
- Uses `CpuReferenceGraphExecutor` for validation

### [**`BnBackward`**](./batchnorm/BnBackward.cpp)

Executes the backward pass of a batch normalization graph to compute gradients of the loss function.
- Given the upstream gradient `dy` of shape `(N, C, H, W)`, the downstream learnable gradients are computed with the chain-rule over the batch and spatial dimensions (`N, H, W`) using saved batch statistics:
    ```python
    dbias = sum(dy)
    x_hat = (x - mean) * inv_variance
    dscale = sum(dy * x_hat)
    dx = (scale * inv_variance) * (dy - (dbias + x_hat * dscale) / nhw)
    ```
    where `nhw = N * H * W` is the number of elements averaged per channel.

### [**`FusedBnInfDReluBnBwd`**](./batchnorm/FusedBnInfDReluBnBwd.cpp)

Executes a fused 3-operation graph demonstrating batchnorm inference followed by activation backward and batchnorm backward passes.

The fused graph consists of three operations:

1. **Batchnorm Inference (Forward)**: Normalizes input `x` using saved statistics
   ```python
   bn_y = scale * ((x - mean) * inv_variance) + bias
   ```

2. **Activation Backward (ReLU)**: Computes gradient through ReLU activation
   ```python
   # ReLU backward: gradient flows through only where forward output was positive
   dx_drelu[i] = dy[i] if bn_y[i] > 0 else 0
   ```

3. **Batchnorm Backward**: Computes gradients with respect to inputs and parameters
   ```python
   dbias = sum(dx_drelu)
   x_hat = (x - mean) * inv_variance
   dscale = sum(dx_drelu * x_hat)
   dx = (scale * inv_variance) * (dx_drelu - (dbias + x_hat * dscale) / nhw)
   ```

**Key Features:**
- Demonstrates multi-operation graph fusion with virtual intermediate tensors
- The intermediate outputs (`bn_y` and `dx_drelu`) are marked as virtual, allowing the backend to optimize memory usage
- Uses `CpuReferenceGraphExecutor` for validation, which executes the entire fused graph on CPU

**Graph Flow:**
```
Inputs: x, dy, scale, bias, mean, inv_variance
        ↓
    bn_y = batchnorm_inference(x, mean, inv_variance, scale, bias)
        ↓ (virtual tensor)
    dx_drelu = activation_backward(bn_y, dy)
        ↓ (virtual tensor)
    [dx, dscale, dbias] = batchnorm_backward(dx_drelu, x, scale, mean, inv_variance)
        ↓
Outputs: dx, dscale, dbias
```

### [**`FusedBnInferenceActiv`**](./batchnorm/FusedBnInferenceActiv.cpp)

Executes a fused batch normalization inference and activation graph.

The fused graph consists of two operations:

1. **Batchnorm Inference**: Normalizes input `x` using saved statistics (mean and inverse variance)
   ```python
   bn_y = scale * ((x - mean) * inv_variance) + bias
   ```

2. **Activation (ReLU)**: Applies ReLU activation
   ```python
   y = relu(bn_y) = max(bn_y, 0)
   ```

**Key Features:**
- Demonstrates fusion of batch normalization inference and activation
- Uses `CpuReferenceGraphExecutor` for validation

### [**`FusedBnInferenceVarianceActiv`**](./batchnorm/FusedBnInferenceVarianceActiv.cpp)

Executes a fused batch normalization inference (with variance) and activation graph.

The fused graph consists of two operations:

1. **Batchnorm Inference (with Variance)**: Normalizes input `x` using saved statistics (mean and variance)
   ```python
   bn_y = scale * ((x - mean) / sqrt(variance + epsilon)) + bias
   ```

2. **Activation (ReLU)**: Applies ReLU activation
   ```python
   y = relu(bn_y) = max(bn_y, 0)
   ```

**Key Features:**
- Demonstrates fusion of batch normalization inference (using variance input) and activation
- Uses `CpuReferenceGraphExecutor` for validation

### [**`ConvFprop`**](./convolution/ConvFprop.cpp)

Executes the forward pass of a 2D convolution operation on a 4D input tensor.

- For an input tensor `x` of shape `(N, C, H_in, W_in)` and a filter tensor `w` of shape `(K, C, R, S)`, the convolution operation computes the output tensor `y` of shape `(N, K, H_out, W_out)`. At a high-level, each output element is computed by the following summation over input channels and filter spatial positions:

    ```python
    y[n, k, p, q] = sum(sum(sum(x[n, c, h, w] * w[k, c, r, s]
                                for c in range(C))
                            for r in range(R))
                        for s in range(S))
    ```

    where the input spatial indices `(h, w)` are determined by the output position `(p, q)`, stride, padding, and dilation:

    ```python
    h = p * stride_h - pad_h + r * dilation_h
    w = q * stride_w - pad_w + s * dilation_w
    ```

    The output spatial dimensions are calculated as:

    ```python
    H_out = floor((H_in + 2*pad_h - dilation_h*(R - 1) - 1) / stride_h) + 1
    W_out = floor((W_in + 2*pad_w - dilation_w*(S - 1) - 1) / stride_w) + 1
    ```

### [**`ConvDgrad`**](./convolution/ConvDgrad.cpp)

Executes the backward pass (data gradient) of a 2D convolution operation to compute input gradients.

- For an output gradient tensor `dy` of shape `(N, K, H_out, W_out)` and a filter tensor `w` of shape `(K, C, R, S)`, the convolution backward data operation computes the input gradient tensor `dx` of shape `(N, C, H_in, W_in)`. At a high-level, each input gradient element is computed by accumulating contributions from all output gradients that were influenced by that input position:

    ```python
    dx[n, c, h, w] = sum(sum(sum(dy[n, k, p, q] * w[k, c, r, s]
                                 for k in range(K))
                             for r in range(R))
                         for s in range(S))
    ```

    where for each input position `(h, w)` and filter position `(r, s)`, the corresponding output position `(p, q)` that contribute is computed as:

    ```python
    p = (h + pad_h - r * dilation_h) / stride_h  # must be integer in [0, H_out)
    q = (w + pad_w - s * dilation_w) / stride_w  # must be integer in [0, W_out)
    ```
    Only positions where both divisions yield integers within the valid output range contribute to the gradient.

    The input gradient dimensions are calculated as the inverse of the forward convolution:

    ```python
    H_in = stride_h * (H_out - 1) + dilation_h * (R - 1) + 1 - 2*pad_h
    W_in = stride_w * (W_out - 1) + dilation_w * (S - 1) + 1 - 2*pad_w
    ```

### [**`ConvWgrad`**](./convolution/ConvWgrad.cpp)

Executes the backward pass (filter gradient) of a 2D convolution operation to compute filter gradients.

- For an output gradient tensor `dy` of shape `(N, K, H_out, W_out)` and an input tensor `x` of shape `(N, C, H_in, W_in)`, the convolution backward filter operation computes the filter gradient tensor `dw` of shape `(K, C, R, S)`. At a high-level, each filter gradient element is computed by accumulating contributions from all batch samples and valid spatial positions:

    ```python
    dw[k, c, r, s] = sum(sum(sum(dy[n, k, p, q] * x[n, c, h, w]
                                 for n in range(N))
                             for p in range(H_out))
                         for q in range(W_out))
    ```

    where for each output position `(p, q)` and filter position `(r, s)`, the corresponding input position `(h, w)` is computed as:

    ```python
    h = p * stride_h - pad_h + r * dilation_h
    w = q * stride_w - pad_w + s * dilation_w
    ```

    For each valid output position, the input position `(h, w)` must fall within the input bounds `[0, H_in) × [0, W_in)` to contribute to the gradient. Input positions outside these bounds (from padding regions) do not contribute.

    The filter gradient dimensions match the original filter tensor from the forward pass:

    ```python
    dw.shape = w.shape = (K, C, R, S)
    ```

    where:
    - `K` = number of output channels (filters)
    - `C` = number of input channels per filter
    - `R, S` = filter spatial dimensions (height, width)

### [**`ConvFpropDeterministic`**](./convolution/ConvFpropDeterministic.cpp)

Executes a deterministic forward pass of a 2D convolution operation. This sample specifically targets the deterministic engine variant (`MIOPEN_ENGINE_DETERMINISTIC`), which guarantees bit-reproducible results across runs at a potential performance cost. This is useful for debugging and validation scenarios where exact reproducibility is required.

### [**`FusedConvFpropBiasActiv`**](./convolution/FusedConvFpropBiasActiv.cpp)

Executes a fused convolution forward pass with bias addition and activation function in a single graph.

The fused graph consists of three operations:

1. **Convolution Forward**: Performs standard 2D convolution
2. **Pointwise Add (Bias)**: Adds a per-channel bias vector to the convolution output
3. **Activation**: Applies an activation function (clamped ReLU) to the result

This demonstrates a common deep learning pattern where convolution, bias, and activation are fused into a single graph for improved performance.

### [**`FusedConvFpropActiv`**](./convolution/FusedConvFpropActiv.cpp)

Executes a fused convolution forward pass with activation function in a single graph.

The fused graph consists of two operations:

1. **Convolution Forward**: Performs standard convolution
    ```python
    conv_y = conv(x, w, stride, padding, dilation)
    ```

2. **Activation (Clamped ReLU)**: Applies ReLU activation with upper and lower clipping bounds to convolution output
    ```python
    y = clamp(relu(conv_y), lower_clip, upper_clip) = min(max(conv_y, lower_clip), upper_clip)
    ```

**Key Features:**
- Demonstrates kernel fusion by combining convolution and activation in a single graph
- The intermediate convolution output (`conv_y`) is marked as virtual, allowing the backend to optimize memory usage and potentially fuse operations
- Shows performance benefits of operation fusion compared to separate kernel launches
- Uses `CpuReferenceGraphExecutor` for validation, which executes the entire fused graph on CPU

**Graph Flow:**
```
Inputs: x (input tensor), w (filter weights)
          ↓
     conv_y = convolution_forward(x, w, stride, padding, dilation)
          ↓ (virtual tensor)
     y = activation_forward(conv_y, mode=RELU, lower_clip=0.2, upper_clip=0.7)
          ↓
Output: y (activated convolution result)
```

**Performance Benefits:**
- Reduces memory bandwidth by avoiding intermediate tensor writes/reads
- Eliminates kernel launch overhead between operations
- Enables better cache utilization by processing data in a single pass

### [**`SdpaFprop`**](./sdpa/SdpaFprop.cpp)

Executes the forward pass of a scaled dot-product attention (SDPA) operation on rank-4 input tensors.

- For query `Q`, key `K`, and value `V` tensors of shape `(B, H, S, D)`, the attention output is computed as:

    ```python
    O = softmax(Q @ K^T / sqrt(D)) @ V
    ```

    where:
    - `B` = batch size
    - `H` = number of attention heads
    - `S` = sequence length
    - `D` = head dimension

- Supports both `BHSD` (row-major) and `BSHD` (sequence-major) memory layouts via strides.
- Configurations without engine support are gracefully skipped.

### [**`SerializationRoundTrip`**](./serialization/SerializationRoundTrip.cpp)

Demonstrates graph serialization and deserialization (round-trip) using hipDNN's JSON and binary serialization formats. The sample builds a convolution forward graph, serializes and deserializes it in both formats, then executes the deserialized graphs to verify correctness. This shows how graphs can be saved, transmitted, and restored for deployment or caching scenarios.

### [**`KnobsUsage`**](./knobs/KnobsUsage.cpp)

Demonstrates how to use hipDNN's engine configuration knobs system for runtime parameter tuning.

**What This Sample Shows:**
1. **Querying Available Knobs**: How to discover what knobs an engine supports
2. **Knob Metadata**: Understanding knob types, constraints, and default values
3. **Setting Knob Values**: Creating execution plans with custom knob settings
4. **Knob Validation**: Validating settings against knob constraints
5. **Knob Value Types**: Using integer, float, and string knobs
6. **Real Execution**: Running graphs with different knob configurations

**Common Knobs:**
- `global.benchmarking` (int64, 0-1): Enable MIOpen kernel benchmarking for optimal performance
- `global.workspace_size_limit` (int64, dynamic): Limit workspace memory for convolution operations

**Usage:**
```bash
./knobs_usage                # Run all demonstrations including graph execution
./knobs_usage --skip-execution  # Show knob API demonstrations only (faster)
```

**Key Features:**
- Comprehensive demonstration of the knobs API
- Shows both query and configuration workflows
- Demonstrates validation and error handling
- Includes real graph execution with different knob configurations
- Educational sample with detailed console output

**Sample Output Sections:**
- Section 1: Query available knobs and their metadata
- Section 2: Use knob lookup map for specific knobs
- Section 3: Create execution plan with default knobs
- Section 4: Set custom knob values
- Section 5: Validate knob settings against constraints
- Section 6: Demonstrate different knob value types
- Section 7-8: Execute graphs with different knob configurations

**Related Documentation:**
- [hipDNN Knobs Documentation](../docs/Knobs.md) - Complete knobs guide
- [HowTo Guide](../docs/HowTo.md#configuring-engine-knobs) - Quick start
- [MIOpen Provider Knobs](../../dnn-providers/miopen-provider/docs/Knobs.md) - Provider-specific knobs

> [!NOTE]
> This sample is educational and demonstrates the knobs API. It is not a performance benchmark or validation test.
