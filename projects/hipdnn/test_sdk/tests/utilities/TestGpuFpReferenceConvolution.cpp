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

// Compare two tensors element-by-element using the tensor iterator (works for any layout/strides)
template <typename T>
void compareTensors(const TensorBase<T>& expected, const TensorBase<T>& actual, float tolerance)
{
    auto expectedIt = expected.cbegin();
    auto actualIt = actual.cbegin();
    size_t i = 0;
    for(; expectedIt != expected.cend(); ++expectedIt, ++actualIt, ++i)
    {
        auto cpuVal = static_cast<float>(*static_cast<const T*>(*expectedIt));
        auto gpuVal = static_cast<float>(*static_cast<const T*>(*actualIt));
        ASSERT_NEAR(cpuVal, gpuVal, tolerance)
            << "Mismatch at element " << i << ": expected=" << cpuVal << " actual=" << gpuVal;
    }
}

// --- Forward convolution helper ---
template <typename DataType, typename AccType = double>
void runGpuVsCpuConvFwd(const std::vector<int64_t>& xDims,
                        const std::vector<int64_t>& wDims,
                        const std::vector<int64_t>& yDims,
                        const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& dilations,
                        const std::vector<int64_t>& prePadding,
                        const std::vector<int64_t>& postPadding,
                        float tolerance,
                        const TensorLayout& xLayout = TensorLayout::NCHW,
                        const TensorLayout& yLayout = TensorLayout::NCHW)
{
    Tensor<DataType> xTensor(xDims, xLayout);
    Tensor<DataType> wTensor(wDims); // weights always NCHW-like layout
    Tensor<DataType> yCpu(yDims, yLayout);
    Tensor<DataType> yGpu(yDims, yLayout);

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), seed);
    wTensor.fillWithRandomValues(
        static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), seed + 1);

    CpuFpReferenceConvolution::fprop<DataType, DataType, DataType, AccType>(
        xTensor, wTensor, yCpu, strides, dilations, prePadding, postPadding);

    GpuFpReferenceConvolution::fprop<DataType, DataType, AccType>(
        xTensor, wTensor, yGpu, strides, dilations, prePadding, postPadding);

    compareTensors(yCpu, yGpu, tolerance);
}

// Overload for uniform padding
template <typename DataType, typename AccType = double>
void runGpuVsCpuConvFwd(const std::vector<int64_t>& xDims,
                        const std::vector<int64_t>& wDims,
                        const std::vector<int64_t>& yDims,
                        const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& dilations,
                        const std::vector<int64_t>& padding,
                        float tolerance,
                        const TensorLayout& xLayout = TensorLayout::NCHW,
                        const TensorLayout& yLayout = TensorLayout::NCHW)
{
    runGpuVsCpuConvFwd<DataType, AccType>(
        xDims, wDims, yDims, strides, dilations, padding, padding, tolerance, xLayout, yLayout);
}

// --- Backward data gradient helper ---
template <typename DataType, typename AccType = double>
void runGpuVsCpuConvBwd(const std::vector<int64_t>& xDims,
                        const std::vector<int64_t>& wDims,
                        const std::vector<int64_t>& yDims,
                        const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& dilations,
                        const std::vector<int64_t>& prePadding,
                        const std::vector<int64_t>& postPadding,
                        float tolerance,
                        const TensorLayout& xLayout = TensorLayout::NCHW,
                        const TensorLayout& yLayout = TensorLayout::NCHW)
{
    Tensor<DataType> wTensor(wDims);
    Tensor<DataType> dyTensor(yDims, yLayout);
    Tensor<DataType> dxCpu(xDims, xLayout);
    Tensor<DataType> dxGpu(xDims, xLayout);

    const unsigned int seed = 42;
    wTensor.fillWithRandomValues(static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), seed);
    dyTensor.fillWithRandomValues(
        static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), seed + 1);

    CpuFpReferenceConvolution::dgrad<DataType, DataType, DataType, AccType>(
        dxCpu, wTensor, dyTensor, strides, dilations, prePadding, postPadding);

    GpuFpReferenceConvolution::dgrad<DataType, DataType, AccType>(
        dxGpu, wTensor, dyTensor, strides, dilations, prePadding, postPadding);

    compareTensors(dxCpu, dxGpu, tolerance);
}

// Overload for uniform padding
template <typename DataType, typename AccType = double>
void runGpuVsCpuConvBwd(const std::vector<int64_t>& xDims,
                        const std::vector<int64_t>& wDims,
                        const std::vector<int64_t>& yDims,
                        const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& dilations,
                        const std::vector<int64_t>& padding,
                        float tolerance,
                        const TensorLayout& xLayout = TensorLayout::NCHW,
                        const TensorLayout& yLayout = TensorLayout::NCHW)
{
    runGpuVsCpuConvBwd<DataType, AccType>(
        xDims, wDims, yDims, strides, dilations, padding, padding, tolerance, xLayout, yLayout);
}

// --- Backward weight gradient helper ---
template <typename DataType, typename AccType = double>
void runGpuVsCpuConvWrw(const std::vector<int64_t>& xDims,
                        const std::vector<int64_t>& wDims,
                        const std::vector<int64_t>& yDims,
                        const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& dilations,
                        const std::vector<int64_t>& prePadding,
                        const std::vector<int64_t>& postPadding,
                        float tolerance,
                        const TensorLayout& xLayout = TensorLayout::NCHW,
                        const TensorLayout& yLayout = TensorLayout::NCHW)
{
    Tensor<DataType> xTensor(xDims, xLayout);
    Tensor<DataType> dyTensor(yDims, yLayout);
    Tensor<DataType> dwCpu(wDims);
    Tensor<DataType> dwGpu(wDims);

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), seed);
    dyTensor.fillWithRandomValues(
        static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), seed + 1);

    CpuFpReferenceConvolution::wgrad<DataType, DataType, DataType, AccType>(
        xTensor, dwCpu, dyTensor, strides, dilations, prePadding, postPadding);

    GpuFpReferenceConvolution::wgrad<DataType, DataType, AccType>(
        xTensor, dwGpu, dyTensor, strides, dilations, prePadding, postPadding);

    compareTensors(dwCpu, dwGpu, tolerance);
}

// Overload for uniform padding
template <typename DataType, typename AccType = double>
void runGpuVsCpuConvWrw(const std::vector<int64_t>& xDims,
                        const std::vector<int64_t>& wDims,
                        const std::vector<int64_t>& yDims,
                        const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& dilations,
                        const std::vector<int64_t>& padding,
                        float tolerance,
                        const TensorLayout& xLayout = TensorLayout::NCHW,
                        const TensorLayout& yLayout = TensorLayout::NCHW)
{
    runGpuVsCpuConvWrw<DataType, AccType>(
        xDims, wDims, yDims, strides, dilations, padding, padding, tolerance, xLayout, yLayout);
}

} // namespace

// ============================================================================
// GpuTestConvFwdRefFp32 — existing tests updated for new API
// ============================================================================

TEST(GpuTestConvFwdRefFp32, BasicConvolution)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>(
        {1, 1, 4, 4}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 1e-5f);
}

TEST(GpuTestConvFwdRefFp32, WithPadding)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 1e-5f);
}

TEST(GpuTestConvFwdRefFp32, WithStride)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>(
        {1, 1, 5, 5}, {1, 1, 3, 3}, {1, 1, 2, 2}, {2, 2}, {1, 1}, {0, 0}, 1e-5f);
}

TEST(GpuTestConvFwdRefFp32, WithDilation)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>(
        {1, 1, 7, 7}, {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1}, {2, 2}, {0, 0}, 1e-5f);
}

TEST(GpuTestConvFwdRefFp32, MultiChannel)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>(
        {1, 3, 4, 4}, {2, 3, 3, 3}, {1, 2, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 1e-4f);
}

TEST(GpuTestConvFwdRefFp32, MultiBatch)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>(
        {2, 1, 4, 4}, {1, 1, 3, 3}, {2, 1, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 1e-5f);
}

TEST(GpuTestConvFwdRefFp32, GroupedConvolution)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>(
        {1, 4, 4, 4}, {4, 2, 3, 3}, {1, 4, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 1e-4f);
}

TEST(GpuTestConvFwdRefFp32, PointwiseConvolution)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>(
        {1, 3, 4, 4}, {2, 3, 1, 1}, {1, 2, 4, 4}, {1, 1}, {1, 1}, {0, 0}, 1e-4f);
}

TEST(GpuTestConvFwdRefFp32, AsymmetricPadding)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 1e-5f);
}

TEST(GpuTestConvFwdRefFp32, SingleElementOutput)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 1, 1}, {1, 1}, {1, 1}, {0, 0}, 1e-5f);
}

// ============================================================================
// GpuTestConvFwdRefFp16
// ============================================================================

TEST(GpuTestConvFwdRefFp16, BasicConvolution)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<half>(
        {1, 1, 4, 4}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 5e-2f);
}

TEST(GpuTestConvFwdRefFp16, MultiChannel)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<half>(
        {1, 3, 4, 4}, {2, 3, 3, 3}, {1, 2, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 5e-2f);
}

TEST(GpuTestConvFwdRefFp16, WithPadding)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<half>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 5e-2f);
}

TEST(GpuTestConvFwdRefFp16, GroupedConvolution)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<half>(
        {1, 4, 4, 4}, {4, 2, 3, 3}, {1, 4, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 5e-2f);
}

// ============================================================================
// GpuTestConvFwdRefBfp16
// ============================================================================

TEST(GpuTestConvFwdRefBfp16, BasicConvolution)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<bfloat16>(
        {1, 1, 4, 4}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 0.1f);
}

TEST(GpuTestConvFwdRefBfp16, MultiChannel)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<bfloat16>(
        {1, 3, 4, 4}, {2, 3, 3, 3}, {1, 2, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 0.1f);
}

TEST(GpuTestConvFwdRefBfp16, GroupedConvolution)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<bfloat16>(
        {1, 4, 4, 4}, {4, 2, 3, 3}, {1, 4, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 0.1f);
}

// ============================================================================
// GpuTestConvFwdRefNhwcFp32 — NHWC layout tests
// ============================================================================

TEST(GpuTestConvFwdRefNhwcFp32, BasicConvolution)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>({1, 1, 4, 4},
                              {1, 1, 3, 3},
                              {1, 1, 2, 2},
                              {1, 1},
                              {1, 1},
                              {0, 0},
                              1e-5f,
                              TensorLayout::NHWC,
                              TensorLayout::NHWC);
}

TEST(GpuTestConvFwdRefNhwcFp32, WithPadding)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>({1, 1, 3, 3},
                              {1, 1, 3, 3},
                              {1, 1, 3, 3},
                              {1, 1},
                              {1, 1},
                              {1, 1},
                              1e-5f,
                              TensorLayout::NHWC,
                              TensorLayout::NHWC);
}

TEST(GpuTestConvFwdRefNhwcFp32, WithStride)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>({1, 1, 5, 5},
                              {1, 1, 3, 3},
                              {1, 1, 2, 2},
                              {2, 2},
                              {1, 1},
                              {0, 0},
                              1e-5f,
                              TensorLayout::NHWC,
                              TensorLayout::NHWC);
}

TEST(GpuTestConvFwdRefNhwcFp32, GroupedConvolution)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>({1, 4, 4, 4},
                              {4, 2, 3, 3},
                              {1, 4, 2, 2},
                              {1, 1},
                              {1, 1},
                              {0, 0},
                              1e-4f,
                              TensorLayout::NHWC,
                              TensorLayout::NHWC);
}

TEST(GpuTestConvFwdRefNhwcFp32, MultiChannel)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>({1, 3, 4, 4},
                              {2, 3, 3, 3},
                              {1, 2, 2, 2},
                              {1, 1},
                              {1, 1},
                              {0, 0},
                              1e-4f,
                              TensorLayout::NHWC,
                              TensorLayout::NHWC);
}

// ============================================================================
// GpuTestConvFwdRef3dFp32 — 3D convolution tests
// ============================================================================

TEST(GpuTestConvFwdRef3dFp32, BasicNcdhw)
{
    SKIP_IF_NO_DEVICES();
    // 1x1x4x4x4 input, 1x1x3x3x3 weight -> 1x1x2x2x2 output
    runGpuVsCpuConvFwd<float>(
        {1, 1, 4, 4, 4}, {1, 1, 3, 3, 3}, {1, 1, 2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 1e-5f);
}

TEST(GpuTestConvFwdRef3dFp32, WithPadding)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>(
        {1, 1, 3, 3, 3}, {1, 1, 3, 3, 3}, {1, 1, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1e-5f);
}

TEST(GpuTestConvFwdRef3dFp32, WithStride)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>(
        {1, 1, 5, 5, 5}, {1, 1, 3, 3, 3}, {1, 1, 2, 2, 2}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, 1e-5f);
}

TEST(GpuTestConvFwdRef3dFp32, WithDilation)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>(
        {1, 1, 7, 7, 7}, {1, 1, 3, 3, 3}, {1, 1, 3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {0, 0, 0}, 1e-5f);
}

TEST(GpuTestConvFwdRef3dFp32, MultiChannel)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>(
        {1, 3, 4, 4, 4}, {2, 3, 3, 3, 3}, {1, 2, 2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 1e-4f);
}

TEST(GpuTestConvFwdRef3dFp32, Ndhwc)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>({1, 3, 4, 4, 4},
                              {2, 3, 3, 3, 3},
                              {1, 2, 2, 2, 2},
                              {1, 1, 1},
                              {1, 1, 1},
                              {0, 0, 0},
                              1e-4f,
                              TensorLayout::NDHWC,
                              TensorLayout::NDHWC);
}

// ============================================================================
// GpuTestConvFwdRefAlphaBeta — alpha/beta scaling tests
// ============================================================================

TEST(GpuTestConvFwdRefAlphaBeta, AlphaOnly)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> yRef({1, 1, 2, 2});
    Tensor<float> yScaled({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Compute with alpha=1.0
    GpuFpReferenceConvolution::fprop<float>(xTensor, wTensor, yRef, {1, 1}, {1, 1}, {0, 0});

    // Compute with alpha=2.0
    GpuFpReferenceConvolution::fprop<float>(xTensor, wTensor, yScaled, {1, 1}, {1, 1}, {0, 0}, 2.0);

    const auto* refData = yRef.memory().hostData();
    const auto* scaledData = yScaled.memory().hostData();

    for(size_t i = 0; i < 4; ++i)
    {
        ASSERT_NEAR(scaledData[i], 2.0f * refData[i], 1e-5f) << "Alpha scaling failed at " << i;
    }
}

TEST(GpuTestConvFwdRefAlphaBeta, BetaAccumulate)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> yTensor({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    yTensor.fillWithValue(1.0f);

    // Pre-fill y with 1.0, then compute with alpha=1.0, beta=1.0
    // Result should be conv(x,w) + 1.0
    Tensor<float> yNoAccum({1, 1, 2, 2});
    GpuFpReferenceConvolution::fprop<float>(xTensor, wTensor, yNoAccum, {1, 1}, {1, 1}, {0, 0});
    GpuFpReferenceConvolution::fprop<float>(
        xTensor, wTensor, yTensor, {1, 1}, {1, 1}, {0, 0}, 1.0, 1.0);

    const auto* noAccumData = yNoAccum.memory().hostData();
    const auto* accumData = yTensor.memory().hostData();

    for(size_t i = 0; i < 4; ++i)
    {
        ASSERT_NEAR(accumData[i], noAccumData[i] + 1.0f, 1e-5f)
            << "Beta accumulation failed at " << i;
    }
}

TEST(GpuTestConvFwdRefAlphaBeta, BetaZeroSkipsRead)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> yBetaZero({1, 1, 2, 2});
    Tensor<float> yDefault({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Pre-fill with garbage — should be ignored when beta=0
    yBetaZero.fillWithValue(999.0f);

    GpuFpReferenceConvolution::fprop<float>(xTensor, wTensor, yDefault, {1, 1}, {1, 1}, {0, 0});
    GpuFpReferenceConvolution::fprop<float>(
        xTensor, wTensor, yBetaZero, {1, 1}, {1, 1}, {0, 0}, 1.0, 0.0);

    const auto* defaultData = yDefault.memory().hostData();
    const auto* betaZeroData = yBetaZero.memory().hostData();

    for(size_t i = 0; i < 4; ++i)
    {
        ASSERT_NEAR(betaZeroData[i], defaultData[i], 1e-5f)
            << "Beta=0 should ignore pre-filled data at " << i;
    }
}

// ============================================================================
// GpuTestConvBwdRefFp32 — backward data gradient tests
// ============================================================================

TEST(GpuTestConvBwdRefFp32, BasicConvolution)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvBwd<float>(
        {1, 1, 4, 4}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 1e-5f);
}

TEST(GpuTestConvBwdRefFp32, WithPadding)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvBwd<float>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 1e-5f);
}

TEST(GpuTestConvBwdRefFp32, WithStride)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvBwd<float>(
        {1, 1, 5, 5}, {1, 1, 3, 3}, {1, 1, 2, 2}, {2, 2}, {1, 1}, {0, 0}, 1e-5f);
}

TEST(GpuTestConvBwdRefFp32, WithDilation)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvBwd<float>(
        {1, 1, 7, 7}, {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1}, {2, 2}, {0, 0}, 1e-5f);
}

TEST(GpuTestConvBwdRefFp32, MultiChannel)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvBwd<float>(
        {1, 3, 4, 4}, {2, 3, 3, 3}, {1, 2, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 1e-4f);
}

TEST(GpuTestConvBwdRefFp32, GroupedConvolution)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvBwd<float>(
        {1, 4, 4, 4}, {4, 2, 3, 3}, {1, 4, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 1e-4f);
}

TEST(GpuTestConvBwdRefFp32, NhwcLayout)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvBwd<float>({1, 3, 4, 4},
                              {2, 3, 3, 3},
                              {1, 2, 2, 2},
                              {1, 1},
                              {1, 1},
                              {0, 0},
                              1e-4f,
                              TensorLayout::NHWC,
                              TensorLayout::NHWC);
}

// ============================================================================
// GpuTestConvBwdRefFp16
// ============================================================================

TEST(GpuTestConvBwdRefFp16, BasicConvolution)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvBwd<half>(
        {1, 1, 4, 4}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 5e-2f);
}

TEST(GpuTestConvBwdRefFp16, MultiChannel)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvBwd<half>(
        {1, 3, 4, 4}, {2, 3, 3, 3}, {1, 2, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 5e-2f);
}

// ============================================================================
// GpuTestConvWrwRefFp32 — backward weight gradient tests
// ============================================================================

TEST(GpuTestConvWrwRefFp32, BasicConvolution)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvWrw<float>(
        {1, 1, 4, 4}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 1e-5f);
}

TEST(GpuTestConvWrwRefFp32, WithPadding)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvWrw<float>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 1e-5f);
}

TEST(GpuTestConvWrwRefFp32, WithStride)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvWrw<float>(
        {1, 1, 5, 5}, {1, 1, 3, 3}, {1, 1, 2, 2}, {2, 2}, {1, 1}, {0, 0}, 1e-5f);
}

TEST(GpuTestConvWrwRefFp32, WithDilation)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvWrw<float>(
        {1, 1, 7, 7}, {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1}, {2, 2}, {0, 0}, 1e-5f);
}

TEST(GpuTestConvWrwRefFp32, MultiChannel)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvWrw<float>(
        {1, 3, 4, 4}, {2, 3, 3, 3}, {1, 2, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 1e-4f);
}

TEST(GpuTestConvWrwRefFp32, GroupedConvolution)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvWrw<float>(
        {1, 4, 4, 4}, {4, 2, 3, 3}, {1, 4, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 1e-4f);
}

TEST(GpuTestConvWrwRefFp32, NhwcLayout)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvWrw<float>({1, 3, 4, 4},
                              {2, 3, 3, 3},
                              {1, 2, 2, 2},
                              {1, 1},
                              {1, 1},
                              {0, 0},
                              1e-4f,
                              TensorLayout::NHWC,
                              TensorLayout::NHWC);
}

// ============================================================================
// GpuTestConvWrwRefFp16
// ============================================================================

TEST(GpuTestConvWrwRefFp16, BasicConvolution)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvWrw<half>(
        {1, 1, 4, 4}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 5e-2f);
}

TEST(GpuTestConvWrwRefFp16, MultiChannel)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvWrw<half>(
        {1, 3, 4, 4}, {2, 3, 3, 3}, {1, 2, 2, 2}, {1, 1}, {1, 1}, {0, 0}, 5e-2f);
}

// ============================================================================
// GpuTestConvFwdRefInt8 — int8 input with int32 or float output
// ============================================================================

TEST(GpuTestConvFwdRefInt8, Int8ToInt32)
{
    SKIP_IF_NO_DEVICES();

    Tensor<int8_t> xTensor({1, 1, 4, 4});
    Tensor<int8_t> wTensor({1, 1, 3, 3});
    Tensor<int32_t> yGpu({1, 1, 2, 2});

    // Fill with small values that won't overflow
    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(static_cast<int8_t>(-3), static_cast<int8_t>(3), seed);
    wTensor.fillWithRandomValues(static_cast<int8_t>(-2), static_cast<int8_t>(2), seed + 1);

    GpuFpReferenceConvolution::fprop<int8_t, int32_t, int32_t>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {0, 0});

    // Verify manually: compute expected output
    const auto* x = xTensor.memory().hostData();
    const auto* w = wTensor.memory().hostData();
    const auto* y = yGpu.memory().hostData();

    // Each output element should be the sum of element-wise products of a 3x3 patch
    for(int ho = 0; ho < 2; ++ho)
    {
        for(int wo = 0; wo < 2; ++wo)
        {
            int32_t expected = 0;
            for(int kh = 0; kh < 3; ++kh)
            {
                for(int kw = 0; kw < 3; ++kw)
                {
                    expected += static_cast<int32_t>(x[(ho + kh) * 4 + (wo + kw)])
                                * static_cast<int32_t>(w[kh * 3 + kw]);
                }
            }
            ASSERT_EQ(y[ho * 2 + wo], expected)
                << "Int8->Int32 mismatch at (" << ho << "," << wo << ")";
        }
    }
}

TEST(GpuTestConvFwdRefInt8, Int8ToFloat)
{
    SKIP_IF_NO_DEVICES();

    Tensor<int8_t> xTensor({1, 1, 4, 4});
    Tensor<int8_t> wTensor({1, 1, 3, 3});
    Tensor<float> yGpu({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(static_cast<int8_t>(-3), static_cast<int8_t>(3), seed);
    wTensor.fillWithRandomValues(static_cast<int8_t>(-2), static_cast<int8_t>(2), seed + 1);

    GpuFpReferenceConvolution::fprop<int8_t, float, float>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {0, 0});

    const auto* x = xTensor.memory().hostData();
    const auto* w = wTensor.memory().hostData();
    const auto* y = yGpu.memory().hostData();

    for(int ho = 0; ho < 2; ++ho)
    {
        for(int wo = 0; wo < 2; ++wo)
        {
            float expected = 0.0f;
            for(int kh = 0; kh < 3; ++kh)
            {
                for(int kw = 0; kw < 3; ++kw)
                {
                    expected += static_cast<float>(x[(ho + kh) * 4 + (wo + kw)])
                                * static_cast<float>(w[kh * 3 + kw]);
                }
            }
            ASSERT_NEAR(y[ho * 2 + wo], expected, 1e-5f)
                << "Int8->Float mismatch at (" << ho << "," << wo << ")";
        }
    }
}

// ============================================================================
// GpuTestConvFwdRefTf32 — TF32 truncation test
// ============================================================================

TEST(GpuTestConvFwdRefTf32, DiffersFromNonTf32)
{
    SKIP_IF_NO_DEVICES();

    // Use values with enough mantissa bits to show TF32 truncation
    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> yNoTf32({1, 1, 2, 2});
    Tensor<float> yTf32({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Regular computation with float accumulation
    GpuFpReferenceConvolution::fprop<float, float, float>(
        xTensor, wTensor, yNoTf32, {1, 1}, {1, 1}, {0, 0}, 1.0, 0.0, false);

    // TF32 computation
    GpuFpReferenceConvolution::fprop<float, float, float>(
        xTensor, wTensor, yTf32, {1, 1}, {1, 1}, {0, 0}, 1.0, 0.0, true);

    const auto* noTf32Data = yNoTf32.memory().hostData();
    const auto* tf32Data = yTf32.memory().hostData();

    // TF32 results should be close but not identical to full-precision
    bool hasDifference = false;
    for(size_t i = 0; i < 4; ++i)
    {
        if(std::abs(noTf32Data[i] - tf32Data[i]) > 1e-10f)
        {
            hasDifference = true;
        }
        // But should still be close
        ASSERT_NEAR(noTf32Data[i], tf32Data[i], 0.1f)
            << "TF32 too far from full precision at " << i;
    }
    ASSERT_TRUE(hasDifference) << "TF32 should produce different results from full precision";
}

// ============================================================================
// GpuTestConvFwdRefPerformance — timing comparisons
// ============================================================================

TEST(GpuTestConvFwdRefPerformance, MediumTensorTimingComparison)
{
    SKIP_IF_NO_DEVICES();

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
    GpuFpReferenceConvolution::fprop<float, float, float>(
        xTensor, wTensor, yGpu, strides, dilations, padding);

    // Time CPU
    auto cpuStart = std::chrono::high_resolution_clock::now();
    CpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yCpu, strides, dilations, padding);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuUs = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count();
    auto cpuMs = static_cast<double>(cpuUs) / 1000.0;

    // Time GPU (kernel already compiled from warm-up)
    hipEvent_t gpuStart = nullptr;
    hipEvent_t gpuStop = nullptr;
    static_cast<void>(hipEventCreate(&gpuStart));
    static_cast<void>(hipEventCreate(&gpuStop));

    static_cast<void>(hipEventRecord(gpuStart, nullptr));
    GpuFpReferenceConvolution::fprop<float, float, float>(
        xTensor, wTensor, yGpu, strides, dilations, padding);
    static_cast<void>(hipEventRecord(gpuStop, nullptr));
    static_cast<void>(hipEventSynchronize(gpuStop));

    float gpuMs = 0.0f;
    static_cast<void>(hipEventElapsedTime(&gpuMs, gpuStart, gpuStop));

    static_cast<void>(hipEventDestroy(gpuStart));
    static_cast<void>(hipEventDestroy(gpuStop));

    std::cout << "[MediumTensor] CPU: " << cpuMs << " ms, GPU: " << gpuMs << " ms\n";

    compareTensors(yCpu, yGpu, 1e-3f);
}

TEST(GpuTestConvFwdRefPerformance, LargeTensorTimingComparison)
{
    SKIP_IF_NO_DEVICES();

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
    GpuFpReferenceConvolution::fprop<float, float, float>(
        xTensor, wTensor, yGpu, strides, dilations, padding);

    // Time CPU
    auto cpuStart = std::chrono::high_resolution_clock::now();
    CpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yCpu, strides, dilations, padding);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuUs = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count();
    auto cpuMs = static_cast<double>(cpuUs) / 1000.0;

    // Time GPU
    hipEvent_t gpuStart = nullptr;
    hipEvent_t gpuStop = nullptr;
    static_cast<void>(hipEventCreate(&gpuStart));
    static_cast<void>(hipEventCreate(&gpuStop));

    static_cast<void>(hipEventRecord(gpuStart, nullptr));
    GpuFpReferenceConvolution::fprop<float, float, float>(
        xTensor, wTensor, yGpu, strides, dilations, padding);
    static_cast<void>(hipEventRecord(gpuStop, nullptr));
    static_cast<void>(hipEventSynchronize(gpuStop));

    float gpuMs = 0.0f;
    static_cast<void>(hipEventElapsedTime(&gpuMs, gpuStart, gpuStop));

    static_cast<void>(hipEventDestroy(gpuStart));
    static_cast<void>(hipEventDestroy(gpuStop));

    std::cout << "[LargeTensor] CPU: " << cpuMs << " ms, GPU: " << gpuMs << " ms\n";

    compareTensors(yCpu, yGpu, 1e-2f);
}
