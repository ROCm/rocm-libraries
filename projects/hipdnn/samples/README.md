# hipDNN Samples

## How to Build

1. **Prerequisites:** First install hipDNN following the [Building documentation](../docs/Building.md).

2. **Build Samples:** From this `samples` directory:
   ```bash
   mkdir build && cd build
   cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ ..
   ninja
   ```

The sample executables will be created in the `build` directory.

## Available Samples

All samples are templated for mixed-precision execution with Fp32, Fp16, and Bfp16 input/output data types, with Fp32 intermediate accumulation.

> [!TIP]
> 💡 Set `HIPDNN_LOG_LEVEL=info` to observe detailed logs from the samples.

The current samples include:

### [**`BnInference`**](./batchnorm/BnInference.cpp)

Executes a single-node batch normalization inference graph on a 4D input tensor.

- It normalizes each dimension of the input tensor `x` of shape `(N, C, H, W)`, using pre-calculated population statistics. The result is then transformed by the learned parameters `scale` and `bias`, each with shape `(1, C, 1, 1)`. At a high-level, the following element-wise linear transformation is broadcast over the batch and spatial dimensions (`N, H, W`):
    ```python
    y = scale * (x - mean) * inv_variance + bias
    ```
    where `y` would then be propagated as input to the subsequent layer.

### [**`BnTraining`**](./batchnorm/BnTraining.cpp)

Executes the forward pass of a batch normalization training graph on a 4D input tensor.
- For an input `x` of shape `(N, C, H, W)`, the mean and variance are calculated over the `N`, `H`, and `W` dimensions for each of the `C` channels or mini-batches, resulting in a `mean` and `inv_variance` of shape `(1, C, 1, 1)`. It then transforms the input and updates the running statistics:
    ```python
    y = scale * (x - mean) * inv_variance + bias
    next_running_mean = (1 - momentum) * prev_running_mean + momentum * batch_mean
    next_running_variance = (1 - momentum) * prev_running_variance + momentum * batch_variance
    ```
- The graph outputs the normalized tensor `y`, along with the mini-batch statistics (`mean`, `inv_variance`) required for the backward pass, and the updated population statistics (`next_running_mean`, `next_running_variance`) required for inference.

### [**`BnBackwards`**](./batchnorm/BnBackwards.cpp)

Executes the backward pass of a batch normalization graph to compute gradients of the loss function.
- Given the upstream differentiable gradient `dy` of shape `(N, C, H, W)`, the downstream learnable gradients are computed with the chain-rule over the batch and spatial dimensions (`N, H, W`) with saved mini-batch statistics:
    ```python
    dbias = sum(dy)
    x_hat = (x - mean) * inv_variance
    dscale = sum(dy * x_hat)
    d_x = scale * inv_variance * (dy - (dbias / nhw) - (x_hat * dscale / nhw))
    ```
    where `nhw = N * H * W`.
    
- For training, `d_x` would subsequently be passed to the preceding layer, and `d_scale` and `d_bias` can be used by an optimizer to update the learnable parameters `scale` and `bias`.

### [**`ConvForward`**](./convolution/ConvForward.cpp)

Executes the forward pass of a convolution operation on a 4D input tensor.

- For an input tensor `x` of shape `(N, C, H, W)` and a filter tensor `w` of shape `(K, C, R, S)`, the convolution operation computes the output tensor `y` of shape `(N, K, P, Q)`:
    ```python
    y[n, k, p, q] = sum(x[n, c, h, w] * w[k, c, r, s])
    ```
    where the summation iterates over the input channels `C` and the spatial dimensions of the filter `(R, S)`, and the indices `(h, w)` are determined by the convolution stride, padding, and dilation. For each output position `(p, q)`, the corresponding input indices `(h, w)` are computed as:
    ```python
    h = p * stride_height - padding_height + r * dilation_height
    w = q * stride_width - padding_width + s * dilation_width
    ```
