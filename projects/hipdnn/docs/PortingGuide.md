# hipDNN Porting Guide: cuDNN to hipDNN Migration

This guide helps developers migrate cuDNN Frontend code to hipDNN. It focuses on practical API differences, common patterns, and working examples.

## Table of Contents
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Common Pitfalls](#common-pitfalls)
- [Working Examples](#working-examples)

## Quick Start

### Minimal hipDNN Project

**CMakeLists.txt**:
```cmake
cmake_minimum_required(VERSION <your_minimum>)


project(my_hipdnn_project VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# CRITICAL: Enable PIC for plugin system compatibility
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find required packages
find_package(hipdnn_frontend CONFIG REQUIRED)

# Create executable
add_executable(my_app main.cpp)

# Link libraries
target_link_libraries(my_app PRIVATE
    hipdnn_frontend
)
```

**Build commands**:
```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm -GNinja
ninja
```

## Core Concepts

### Key Differences: cuDNN vs hipDNN

| Aspect | cuDNN Frontend | hipDNN Frontend |
|--------|----------------|-----------------|
| **Namespace** | cudnn_frontend | hipdnn_frontend |
| **Handle Creation** | cudnnCreate(&handle) | hipdnnCreate(&handle) |
| **Handle Destruction** | cudnnDestroy(handle) | hipdnnDestroy(handle) |
| **Heuristics Modes** | All cuDNN heuristic modes | Currently only HeurMode_t::FALLBACK |
| **Operation Support** | All cuDNN operations | [Operation Support](./OperationSupport.md) |
| **Device Memory Utility** | Surface<type> | MigratableMemory<type> |
| **Device Memory Access** | Surface<type>::devPtr | MigratableMemory<type>::deviceData() |

### Tensor Dimensions and Layouts

Tensor dimension ordering in hipDNN is **operation-specific**, following the same conventions as
PyTorch and cuDNN. The **memory layout** (channel-first vs channel-last) is always controlled by
strides and stride order, not by the order of the tensor dimension vector that always holds values as [N,C,H,W] or [N,C,D,H,W]. For example, memory arranged as NCHW corresponds to stride order {3,2,1,0} (W is the most tightly packed), and NDHWC corresponds to stride order {4,0,3,2,1} (C is the most tightly packed). Use the `TensorLayout` constants and
`generateStrides()` utility to compute strides for common layouts.

#### Convolution

| Tensor | Shape (4D) | Shape (5D) | Description |
|--------|-----------|------------|-------------|
| Input (x) | `(N, C, H, W)` | `(N, C, D, H, W)` | Batch, channels, spatial dims |
| Weights (w) | `(K, C/groups, R, S)` | `(K, C/groups, T, R, S)` | Output channels, input channels per group, kernel spatial dims |
| Output (y) | `(N, K, H_out, W_out)` | `(N, K, D_out, H_out, W_out)` | Batch, output channels, output spatial dims |

```cpp
// Convolution example: dims always follow (N, C, spatial...) ordering
auto x = TensorAttributes()
             .set_dim({1, 64, 28, 28})   // N=1, C=64, H=28, W=28
             .set_stride(generateStrides({1, 64, 28, 28}, TensorLayout::NHWC.strideOrder));

auto w = TensorAttributes()
             .set_dim({128, 64, 3, 3})   // K=128, C=64, R=3, S=3
             .set_stride(generateStrides({128, 64, 3, 3}, TensorLayout::NHWC.strideOrder));
```

#### Batch Normalization

| Tensor | Shape | Description |
|--------|-------|-------------|
| Input (x) | `(N, C, H, W)` or `(N, C, D, H, W)` | Same ordering as convolution |
| Scale, Bias, Mean, Variance | `(1, C, 1, 1)` or `(1, C, 1, 1, 1)` | Per-channel parameters |
| Output (y) | Same as input | Shape preserved |

Statistics are computed per-channel over the batch and spatial dimensions.

#### Layer Normalization

| Tensor | Shape | Description |
|--------|-------|-------------|
| Input (x) | `(N, ...)` | Batch first, then feature dims |
| Scale, Bias | `(1, ...)` | Batch dim = 1, remaining dims match input feature dims |
| Mean, Inv Variance | Stats dims | Batch dims from input, normalized dims = 1 |

Normalization is performed over the feature dimensions (all dims where scale > 1).

#### Matrix Multiplication

| Tensor | Shape | Description |
|--------|-------|-------------|
| A | `(...batch, M, K)` | Leading batch dims, last two are matrix dims |
| B | `(...batch, K, N)` | K must match A's last dim |
| C (output) | `(...batch, M, N)` | Batch dims are broadcast |

Batch dimensions support broadcasting (dims must be equal or divisible).

```cpp
// Matmul example: A(batch, M, K) @ B(batch, K, N) -> C(batch, M, N)
auto a = TensorAttributes()
             .set_dim({4, 128, 64})    // batch=4, M=128, K=64
             .set_stride({128*64, 64, 1});

auto b = TensorAttributes()
             .set_dim({4, 64, 256})    // batch=4, K=64, N=256
             .set_stride({64*256, 256, 1});
```

#### Pointwise Operations

Pointwise operations (ReLU, Sigmoid, Add, Mul, etc.) are **dimension-agnostic** — they accept
tensors of any shape. For binary and ternary operations, inputs are broadcast using NumPy-style
broadcasting rules (dimensions compared right-to-left; compatible if equal or 1).

## Common Pitfalls

### 1. CMAKE_POSITION_INDEPENDENT_CODE

**Error**: TLS model mismatch, plugin load failures

**Cause**: hipDNN uses a plugin system that requires position-independent code

**Fix**:
```cmake
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
```

### 2. Lack of Heuristic Modes

**Error**: Missing Heuristic modes A and B

**Cause**: The heuristic implementation in hipDNN has yet to be implemented

**Fix**: Use a combination of `graph::get_ranked_engine_ids()` and `graph::set_preferred_engine_id_ext()` if you need more detailed control over engine selection.

### 3. Device Memory Utilities

**Error**: Different memory utilities for allocating device memory.

**Cause**: The memory utilities are typically consumer dependent, and written on an as-needed basis.  cuDNN provides a Surface utility for their samples, for example.

**Fix**: `MigratableMemory<type>` is a utility that can automatically migrate data between host and device, and is a decent stand-in.  If you want to manage things like dims / strides more carefully, there's also a `Tensor` utility class.  Both of these classes can be found in the `hipdnn_data_sdk::utilities` namespace.

## Working Examples

### Complete Batch Normalization Inference -> ReLU Backward -> Batch Normalization Backwards (hipDNN)

```cpp
#include <hipdnn_frontend.hpp>
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk;
using namespace hipdnn_data_sdk::utilities;

// Error checking macros
// Assume an error-checking macro called REQUIRE()

void test_batchnorm_simple()
{
    namespace fe = hipdnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto BN_X = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("X")
                                 .set_dim({4, 32, 16, 16})
                                 .set_stride({32 * 16 * 16, 1, 32 * 16, 32}));

    auto scale = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({1, 32, 1, 1})
                                  .set_stride({32, 1, 32, 32})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto bias = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({1, 32, 1, 1})
                                 .set_stride({32, 1, 32, 32})
                                 .set_data_type(fe::DataType_t::FLOAT));
    auto mean = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("mean")
                                 .set_dim({1, 32, 1, 1})
                                 .set_stride({32, 1, 32, 32})
                                 .set_data_type(fe::DataType_t::FLOAT));
    auto inv_variance = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_name("inv_variance")
                                         .set_dim({1, 32, 1, 1})
                                         .set_stride({32, 1, 32, 32})
                                         .set_data_type(fe::DataType_t::FLOAT));

    auto batchnorm_inference_attributes = fe::graph::Batchnorm_inference_attributes();
    auto BN_Y = graph.batchnorm_inference(BN_X, mean, inv_variance, scale, bias, batchnorm_inference_attributes);

    auto DY = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("DY")
                               .set_dim({4, 32, 16, 16})
                               .set_stride({32 * 16 * 16, 1, 32 * 16, 32}));

    auto relu_backward_attribues = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_BWD);
    auto DX_drelu = graph.pointwise(DY, BN_Y, relu_backward_attribues);
    DX_drelu->set_data_type(fe::DataType_t::HALF);

    auto DBN_options = fe::graph::Batchnorm_backward_attributes().set_saved_mean_and_inv_variance(mean, inv_variance);
    auto [DX, dscale, dbias] = graph.batchnorm_backward(DX_drelu, BN_X, scale, DBN_options);
    DX->set_output(true);
    dscale->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    dbias->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    // Create the hipDNN handle
    hipdnnHandle_t handle;
    hipdnnCreate(&handle);

    REQUIRE(graph.validate());

    REQUIRE(graph.build_operation_graph(handle));

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}));

    REQUIRE(graph.check_support());

    REQUIRE(graph.build_plans());

    MigratableMemory<half> BN_X_tensor(4 * 32 * 16 * 16);
    MigratableMemory<half> DY_tensor(4 * 32 * 16 * 16);
    MigratableMemory<float> Mean_tensor(32);
    MigratableMemory<float> Inv_variance_tensor(32);
    MigratableMemory<float> Scale_tensor(32);
    MigratableMemory<float> Bias_tensor(32);
    MigratableMemory<float> Dscale_tensor(32);
    MigratableMemory<float> Dbias_tensor(32);
    MigratableMemory<half> DX_tensor(4 * 32 * 16 * 16);

    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size));
    MigratableMemory<int8_t> workspace(workspace_size);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {
        {BN_X, BN_X_tensor.deviceData()},
        {DY, DY_tensor.deviceData()},
        {mean, Mean_tensor.deviceData()},
        {inv_variance, Inv_variance_tensor.deviceData()},
        {scale, Scale_tensor.deviceData()},
        {bias, Bias_tensor.deviceData()},
        {dscale, Dscale_tensor.deviceData()},
        {dbias, Dbias_tensor.deviceData()},
        {DX, DX_tensor.deviceData()}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.deviceData()));

    hipdnnDestroy(handle);
}
```

### Complete Batch Normalization Inference -> ReLU Backward -> Batch Normalization Backwards (cuDNN)

```cpp
#include <cudnn_frontend.h>

// Error checking macros
// Assume an error-checking macro called REQUIRE()
// Assume a memory-managing utility class called Surface (from sample utils)

void test_batchnorm_simple()
{
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto BN_X = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("X")
                                 .set_dim({4, 32, 16, 16})
                                 .set_stride({32 * 16 * 16, 1, 32 * 16, 32}));

    auto scale        = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({1, 32, 1, 1})
                                  .set_stride({32, 1, 32, 32})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto bias         = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({1, 32, 1, 1})
                                 .set_stride({32, 1, 32, 32})
                                 .set_data_type(fe::DataType_t::FLOAT));
    auto mean         = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("mean")
                                 .set_dim({1, 32, 1, 1})
                                 .set_stride({32, 1, 32, 32})
                                 .set_data_type(fe::DataType_t::FLOAT));
    auto inv_variance = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_name("inv_variance")
                                         .set_dim({1, 32, 1, 1})
                                         .set_stride({32, 1, 32, 32})
                                         .set_data_type(fe::DataType_t::FLOAT));

    auto batchnorm_inference_attributes = fe::graph::Batchnorm_inference_attributes();
    auto BN_Y = graph.batchnorm_inference(BN_X, mean, inv_variance, scale, bias, batchnorm_inference_attributes);

    auto DY = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("DY")
                               .set_dim({4, 32, 16, 16})
                               .set_stride({32 * 16 * 16, 1, 32 * 16, 32}));

    auto relu_backward_attribues = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_BWD);
    auto DX_drelu                = graph.pointwise(DY, BN_Y, relu_backward_attribues);
    DX_drelu->set_data_type(fe::DataType_t::HALF);

    auto DBN_options = fe::graph::Batchnorm_backward_attributes().set_saved_mean_and_inv_variance(mean, inv_variance);
    auto [DX, dscale, dbias] = graph.batchnorm_backward(DX_drelu, BN_X, scale, DBN_options);
    DX->set_output(true);
    dscale->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    dbias->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);

    REQUIRE(graph.validate());

    REQUIRE(graph.build_operation_graph(handle));

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good());

    REQUIRE(graph.check_support());

    REQUIRE(graph.build_plans());

    Surface<half> BN_X_tensor(4 * 32 * 16 * 16, false);
    Surface<half> DY_tensor(4 * 32 * 16 * 16, false);
    Surface<float> Mean_tensor(32, false);
    Surface<float> Inv_variance_tensor(32, false);
    Surface<float> Scale_tensor(32, false);
    Surface<float> Bias_tensor(32, false);
    Surface<float> Dscale_tensor(32, false);
    Surface<float> Dbias_tensor(32, false);
    Surface<half> DX_tensor(4 * 32 * 16 * 16, false);

    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size));
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {BN_X, BN_X_tensor.devPtr},
        {DY, DY_tensor.devPtr},
        {mean, Mean_tensor.devPtr},
        {inv_variance, Inv_variance_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {bias, Bias_tensor.devPtr},
        {dscale, Dscale_tensor.devPtr},
        {dbias, Dbias_tensor.devPtr},
        {DX, DX_tensor.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr));

    cudnnDestroy(handle);
}
```

## Additional Resources

- **hipDNN Samples**: [Samples](../samples/)
- **Building Reference**: [Building](./Building.md)
- **HowTo Guide**: [How To Guide](./HowTo.md)
