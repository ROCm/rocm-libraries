# hipDNN Samples

## How to Build

1. **Prerequisites:**
   - Build and install hipDNN following the [Building documentation](../docs/Building.md)
   - Ensure ROCm is installed (typically in `/opt/rocm`)
   - A ROCm-compatible GPU is required to run the samples
   - Have `ninja` build tool available: `apt install ninja-build`

2. **Build Samples:** From this `samples` directory:
   ```bash
   mkdir build && cd build
   cmake -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -G Ninja ..
   ninja
   ```

The sample executables will be created in the `build` directory.

## Running Samples

All samples accept the following command line options:
- `--cpu-validation` - Enable CPU reference validation of results
- `--help` - Displays help message

Example:
```bash
./build/conv_forward --cpu-validation
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

### [**`BnTraining`**](./batchnorm/BnTraining.cpp)

Executes the forward pass of a batch normalization training graph on a 4D input tensor.
- For an input `x` of shape `(N, C, H, W)`, the mean and variance are calculated over the `N`, `H`, and `W` dimensions for each of the `C` channels, resulting in a `mean` and `inv_variance` of shape `(1, C, 1, 1)`. It then transforms the input and updates the running statistics:
    ```python
    y = scale * ((x - mean) * inv_variance) + bias
    next_running_mean = (1 - momentum) * prev_running_mean + momentum * batch_mean
    next_running_variance = (1 - momentum) * prev_running_variance + momentum * batch_variance
    ```
- The graph outputs the normalized tensor `y`, along with the batch mean/variance (`mean`, `inv_variance`) required for the backward pass, and the updated population statistics (`next_running_mean`, `next_running_variance`) required for inference.

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
