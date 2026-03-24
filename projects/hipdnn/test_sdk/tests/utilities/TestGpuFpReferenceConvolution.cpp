// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn_gpu_ref/GpuFpReferenceConvolution.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_gpu_ref;

namespace
{

template <typename DataType>
void runGpuVsCpuConvFwd(const std::vector<int64_t>& xDims,
                        const std::vector<int64_t>& wDims,
                        const std::vector<int64_t>& yDims,
                        const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& dilations,
                        const std::vector<int64_t>& prePadding,
                        const std::vector<int64_t>& postPadding,
                        float tolerance)
{
    // Create tensors
    Tensor<DataType> xTensor(xDims);
    Tensor<DataType> wTensor(wDims);
    Tensor<DataType> yCpu(yDims);
    Tensor<DataType> yGpu(yDims);

    // Fill with deterministic random values (small range to avoid overflow in reduced precision)
    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), seed);
    wTensor.fillWithRandomValues(
        static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), seed + 1);

    // Run CPU reference
    CpuFpReferenceConvolution::fprop<DataType, DataType, DataType, float>(
        xTensor, wTensor, yCpu, strides, dilations, prePadding, postPadding);

    // Run GPU reference
    GpuFpReferenceConvolution::fprop<DataType, DataType, DataType, float>(
        xTensor, wTensor, yGpu, strides, dilations, prePadding, postPadding);

    // Compare element-wise
    auto totalElements = static_cast<size_t>(yDims[0] * yDims[1] * yDims[2] * yDims[3]);
    const auto* cpuData = yCpu.memory().hostData();
    const auto* gpuData = yGpu.memory().hostData();

    for(size_t i = 0; i < totalElements; ++i)
    {
        auto cpuVal = static_cast<float>(cpuData[i]);
        auto gpuVal = static_cast<float>(gpuData[i]);
        ASSERT_NEAR(cpuVal, gpuVal, tolerance)
            << "Mismatch at element " << i << ": cpu=" << cpuVal << " gpu=" << gpuVal;
    }
}

// Overload for uniform padding
template <typename DataType>
void runGpuVsCpuConvFwd(const std::vector<int64_t>& xDims,
                        const std::vector<int64_t>& wDims,
                        const std::vector<int64_t>& yDims,
                        const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& dilations,
                        const std::vector<int64_t>& padding,
                        float tolerance)
{
    runGpuVsCpuConvFwd<DataType>(xDims, wDims, yDims, strides, dilations, padding, padding,
                                tolerance);
}

} // namespace

// ============================================================================
// GpuTestConvFwdRefFp32
// ============================================================================

TEST(GpuTestConvFwdRefFp32, BasicConvolution)
{
    SKIP_IF_NO_DEVICES();
    // 1x1x4x4 input, 1x1x3x3 weight -> 1x1x2x2 output
    runGpuVsCpuConvFwd<float>({1, 1, 4, 4}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {0, 0},
                              1e-5f);
}

TEST(GpuTestConvFwdRefFp32, WithPadding)
{
    SKIP_IF_NO_DEVICES();
    // 1x1x3x3 input, 1x1x3x3 weight, pad=1 -> 1x1x3x3 output
    runGpuVsCpuConvFwd<float>({1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1},
                              1e-5f);
}

TEST(GpuTestConvFwdRefFp32, WithStride)
{
    SKIP_IF_NO_DEVICES();
    // 1x1x5x5 input, 1x1x3x3 weight, stride=2 -> 1x1x2x2 output
    runGpuVsCpuConvFwd<float>({1, 1, 5, 5}, {1, 1, 3, 3}, {1, 1, 2, 2}, {2, 2}, {1, 1}, {0, 0},
                              1e-5f);
}

TEST(GpuTestConvFwdRefFp32, WithDilation)
{
    SKIP_IF_NO_DEVICES();
    // 1x1x7x7 input, 1x1x3x3 weight, dilation=2 -> 1x1x3x3 output
    runGpuVsCpuConvFwd<float>({1, 1, 7, 7}, {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1}, {2, 2}, {0, 0},
                              1e-5f);
}

TEST(GpuTestConvFwdRefFp32, MultiChannel)
{
    SKIP_IF_NO_DEVICES();
    // 1x3x4x4 input, 2x3x3x3 weight -> 1x2x2x2 output
    runGpuVsCpuConvFwd<float>({1, 3, 4, 4}, {2, 3, 3, 3}, {1, 2, 2, 2}, {1, 1}, {1, 1}, {0, 0},
                              1e-4f);
}

TEST(GpuTestConvFwdRefFp32, MultiBatch)
{
    SKIP_IF_NO_DEVICES();
    // 2x1x4x4 input, 1x1x3x3 weight -> 2x1x2x2 output
    runGpuVsCpuConvFwd<float>({2, 1, 4, 4}, {1, 1, 3, 3}, {2, 1, 2, 2}, {1, 1}, {1, 1}, {0, 0},
                              1e-5f);
}

TEST(GpuTestConvFwdRefFp32, GroupedConvolution)
{
    SKIP_IF_NO_DEVICES();
    // 1x4x4x4 input, 4x2x3x3 weight (2 groups, 2 channels per group, 2 output channels per group)
    runGpuVsCpuConvFwd<float>({1, 4, 4, 4}, {4, 2, 3, 3}, {1, 4, 2, 2}, {1, 1}, {1, 1}, {0, 0},
                              1e-4f);
}

TEST(GpuTestConvFwdRefFp32, PointwiseConvolution)
{
    SKIP_IF_NO_DEVICES();
    // 1x1 kernel (pointwise)
    // 1x3x4x4 input, 2x3x1x1 weight -> 1x2x4x4 output
    runGpuVsCpuConvFwd<float>({1, 3, 4, 4}, {2, 3, 1, 1}, {1, 2, 4, 4}, {1, 1}, {1, 1}, {0, 0},
                              1e-4f);
}

TEST(GpuTestConvFwdRefFp32, AsymmetricPadding)
{
    SKIP_IF_NO_DEVICES();
    // 1x1x3x3 input, 1x1x3x3 weight, prePad={1,0} postPad={0,1} -> 1x1x2x2 output
    runGpuVsCpuConvFwd<float>({1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0},
                              {0, 1}, 1e-5f);
}

TEST(GpuTestConvFwdRefFp32, SingleElementOutput)
{
    SKIP_IF_NO_DEVICES();
    // 1x1x3x3 input, 1x1x3x3 weight -> 1x1x1x1 output
    runGpuVsCpuConvFwd<float>({1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 1, 1}, {1, 1}, {1, 1}, {0, 0},
                              1e-5f);
}

// ============================================================================
// GpuTestConvFwdRefFp16
// ============================================================================

TEST(GpuTestConvFwdRefFp16, BasicConvolution)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<half>({1, 1, 4, 4}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {0, 0},
                             5e-2f);
}

TEST(GpuTestConvFwdRefFp16, MultiChannel)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<half>({1, 3, 4, 4}, {2, 3, 3, 3}, {1, 2, 2, 2}, {1, 1}, {1, 1}, {0, 0},
                             5e-2f);
}

TEST(GpuTestConvFwdRefFp16, WithPadding)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<half>({1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1},
                             5e-2f);
}

TEST(GpuTestConvFwdRefFp16, GroupedConvolution)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<half>({1, 4, 4, 4}, {4, 2, 3, 3}, {1, 4, 2, 2}, {1, 1}, {1, 1}, {0, 0},
                             5e-2f);
}

// ============================================================================
// GpuTestConvFwdRefBfp16
// ============================================================================

TEST(GpuTestConvFwdRefBfp16, BasicConvolution)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<bfloat16>({1, 1, 4, 4}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {0, 0},
                                 0.1f);
}

TEST(GpuTestConvFwdRefBfp16, MultiChannel)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<bfloat16>({1, 3, 4, 4}, {2, 3, 3, 3}, {1, 2, 2, 2}, {1, 1}, {1, 1}, {0, 0},
                                 0.1f);
}

TEST(GpuTestConvFwdRefBfp16, GroupedConvolution)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<bfloat16>({1, 4, 4, 4}, {4, 2, 3, 3}, {1, 4, 2, 2}, {1, 1}, {1, 1}, {0, 0},
                                 0.1f);
}

// ============================================================================
// GpuTestConvFwdRefPerformance
// ============================================================================

TEST(GpuTestConvFwdRefPerformance, MediumTensorTimingComparison)
{
    SKIP_IF_NO_DEVICES();

    // 2x64x14x14 -> 2x128x12x12 with 3x3 kernel
    const std::vector<int64_t> xDims = {2, 64, 14, 14};
    const std::vector<int64_t> wDims = {128, 64, 3, 3};
    const std::vector<int64_t> yDims = {2, 128, 12, 12};
    const std::vector<int64_t> strides = {1, 1};
    const std::vector<int64_t> dilations = {1, 1};
    const std::vector<int64_t> padding = {0, 0};

    Tensor<float> xTensor(xDims);
    Tensor<float> wTensor(wDims);
    Tensor<float> yCpu(yDims);
    Tensor<float> yGpu(yDims);

    const unsigned int seed = 123;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Warm-up run (includes HipRTC compilation on first call)
    GpuFpReferenceConvolution::fprop<float, float, float, float>(xTensor, wTensor, yGpu, strides,
                                                                 dilations, padding);

    // Time CPU
    auto cpuStart = std::chrono::high_resolution_clock::now();
    CpuFpReferenceConvolution::fprop<float, float, float, float>(xTensor, wTensor, yCpu, strides,
                                                                 dilations, padding);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuUs = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count();
    auto cpuMs = static_cast<double>(cpuUs) / 1000.0;

    // Time GPU (kernel already compiled from warm-up)
    hipEvent_t gpuStart = nullptr;
    hipEvent_t gpuStop = nullptr;
    static_cast<void>(hipEventCreate(&gpuStart));
    static_cast<void>(hipEventCreate(&gpuStop));

    static_cast<void>(hipEventRecord(gpuStart, nullptr));
    GpuFpReferenceConvolution::fprop<float, float, float, float>(xTensor, wTensor, yGpu, strides,
                                                                 dilations, padding);
    static_cast<void>(hipEventRecord(gpuStop, nullptr));
    static_cast<void>(hipEventSynchronize(gpuStop));

    float gpuMs = 0.0f;
    static_cast<void>(hipEventElapsedTime(&gpuMs, gpuStart, gpuStop));

    static_cast<void>(hipEventDestroy(gpuStart));
    static_cast<void>(hipEventDestroy(gpuStop));

    std::cout << "[MediumTensor] CPU: " << cpuMs << " ms, GPU: " << gpuMs << " ms\n";

    // Verify correctness
    auto totalElements = static_cast<size_t>(yDims[0] * yDims[1] * yDims[2] * yDims[3]);
    const auto* cpuData = yCpu.memory().hostData();
    const auto* gpuData = yGpu.memory().hostData();

    for(size_t i = 0; i < totalElements; ++i)
    {
        ASSERT_NEAR(cpuData[i], gpuData[i], 1e-3f)
            << "Mismatch at element " << i << ": cpu=" << cpuData[i] << " gpu=" << gpuData[i];
    }
}

TEST(GpuTestConvFwdRefPerformance, LargeTensorTimingComparison)
{
    SKIP_IF_NO_DEVICES();

    // 8x128x28x28 -> 8x256x26x26 with 3x3 kernel
    const std::vector<int64_t> xDims = {8, 128, 28, 28};
    const std::vector<int64_t> wDims = {256, 128, 3, 3};
    const std::vector<int64_t> yDims = {8, 256, 26, 26};
    const std::vector<int64_t> strides = {1, 1};
    const std::vector<int64_t> dilations = {1, 1};
    const std::vector<int64_t> padding = {0, 0};

    Tensor<float> xTensor(xDims);
    Tensor<float> wTensor(wDims);
    Tensor<float> yCpu(yDims);
    Tensor<float> yGpu(yDims);

    const unsigned int seed = 456;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Warm-up run
    GpuFpReferenceConvolution::fprop<float, float, float, float>(xTensor, wTensor, yGpu, strides,
                                                                 dilations, padding);

    // Time CPU
    auto cpuStart = std::chrono::high_resolution_clock::now();
    CpuFpReferenceConvolution::fprop<float, float, float, float>(xTensor, wTensor, yCpu, strides,
                                                                 dilations, padding);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuUs = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count();
    auto cpuMs = static_cast<double>(cpuUs) / 1000.0;

    // Time GPU
    hipEvent_t gpuStart = nullptr;
    hipEvent_t gpuStop = nullptr;
    static_cast<void>(hipEventCreate(&gpuStart));
    static_cast<void>(hipEventCreate(&gpuStop));

    static_cast<void>(hipEventRecord(gpuStart, nullptr));
    GpuFpReferenceConvolution::fprop<float, float, float, float>(xTensor, wTensor, yGpu, strides,
                                                                 dilations, padding);
    static_cast<void>(hipEventRecord(gpuStop, nullptr));
    static_cast<void>(hipEventSynchronize(gpuStop));

    float gpuMs = 0.0f;
    static_cast<void>(hipEventElapsedTime(&gpuMs, gpuStart, gpuStop));

    static_cast<void>(hipEventDestroy(gpuStart));
    static_cast<void>(hipEventDestroy(gpuStop));

    std::cout << "[LargeTensor] CPU: " << cpuMs << " ms, GPU: " << gpuMs << " ms\n";

    // Verify correctness (slightly relaxed tolerance for larger accumulations)
    auto totalElements = static_cast<size_t>(yDims[0] * yDims[1] * yDims[2] * yDims[3]);
    const auto* cpuData = yCpu.memory().hostData();
    const auto* gpuData = yGpu.memory().hostData();

    for(size_t i = 0; i < totalElements; ++i)
    {
        ASSERT_NEAR(cpuData[i], gpuData[i], 1e-2f)
            << "Mismatch at element " << i << ": cpu=" << cpuData[i] << " gpu=" << gpuData[i];
    }
}
