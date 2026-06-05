// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "ConvShapeCase.hpp"
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn_gpu_ref/GpuFpReferenceConvolution.hpp>

#include <string>
#include <vector>

// ============================================================================
// Shared infrastructure for weight-gradient (wgrad) GPU-vs-CPU reference tests.
// ============================================================================

namespace gpu_conv_wgrad_ref_test
{

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_gpu_ref;

using gpu_conv_ref_test::assertAllClose;
using gpu_conv_ref_test::ConvShapeCase;
using ConvWgradShapeCase = ConvShapeCase;

// Core helper: fills x and dy tensors, runs GPU and CPU wgrad, compares dw results.
// For wgrad: x and dy are inputs, dw is the output (weight gradient).
template <typename XDataType, typename DwDataType, typename DyDataType, typename ComputeDataType>
void compareGpuVsCpuConvWrw(Tensor<XDataType>& xTensor,
                            Tensor<DyDataType>& dyTensor,
                            Tensor<DwDataType>& dwCpu,
                            Tensor<DwDataType>& dwGpu,
                            const std::vector<int64_t>& strides,
                            const std::vector<int64_t>& dilations,
                            const std::vector<int64_t>& prePadding,
                            const std::vector<int64_t>& postPadding,
                            float tolerance,
                            float fillRange)
{
    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(
        static_cast<XDataType>(-fillRange), static_cast<XDataType>(fillRange), seed);
    dyTensor.fillWithRandomValues(
        static_cast<DyDataType>(-fillRange), static_cast<DyDataType>(fillRange), seed + 1);

    CpuFpReferenceConvolution::wgrad<XDataType, DwDataType, DyDataType, ComputeDataType>(
        xTensor, dwCpu, dyTensor, strides, dilations, prePadding, postPadding);

    GpuFpReferenceConvolution::wgrad<XDataType, DwDataType, DyDataType, ComputeDataType>(
        xTensor, dwGpu, dyTensor, strides, dilations, prePadding, postPadding);

    assertAllClose(dwCpu, dwGpu, tolerance);
}

// Convenience wrapper for uniform-type wgrad tests.
// When layout is non-null, x and dy use channel-last strides.
// dw (weight gradient) always uses default packed (KCRS) strides.
template <typename DataType, typename ComputeDataType = double>
void runGpuVsCpuConvWrw(const std::vector<int64_t>& xDims,
                        const std::vector<int64_t>& wDims,
                        const std::vector<int64_t>& dyDims,
                        const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& dilations,
                        const std::vector<int64_t>& prePadding,
                        const std::vector<int64_t>& postPadding,
                        float tolerance,
                        const TensorLayout* layout = nullptr,
                        float fillRange = 1.0f)
{
    auto xTensor = layout != nullptr ? Tensor<DataType>(xDims, *layout) : Tensor<DataType>(xDims);
    auto dyTensor
        = layout != nullptr ? Tensor<DataType>(dyDims, *layout) : Tensor<DataType>(dyDims);
    auto dwCpu = Tensor<DataType>(wDims);
    auto dwGpu = Tensor<DataType>(wDims);

    compareGpuVsCpuConvWrw<DataType, DataType, DataType, ComputeDataType>(xTensor,
                                                                          dyTensor,
                                                                          dwCpu,
                                                                          dwGpu,
                                                                          strides,
                                                                          dilations,
                                                                          prePadding,
                                                                          postPadding,
                                                                          tolerance,
                                                                          fillRange);
}

// ============================================================================
// ConvWgradShapeSuite — parameterized fixture for shape-based GPU-vs-CPU tests
// ============================================================================

template <typename DataType>
class ConvWgradShapeSuite : public ::testing::TestWithParam<ConvWgradShapeCase>
{
protected:
    static float tolerance(const ConvWgradShapeCase& tc)
    {
        constexpr double FILL_RANGE = 1.0;
        auto dyDims = tc.computeOutputDims();
        return hipdnn_test_sdk::utilities::conv::
            calculateConvWrwTolerance<DataType, DataType, double>(
                -FILL_RANGE, FILL_RANGE, -FILL_RANGE, FILL_RANGE, dyDims);
    }

    void runConvWgradShapeTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& tc = GetParam();
        auto dyDims = tc.computeOutputDims();
        runGpuVsCpuConvWrw<DataType>(tc.xDims,
                                     tc.wDims,
                                     dyDims,
                                     tc.strides,
                                     tc.dilations,
                                     tc.padding,
                                     tc.padding,
                                     tolerance(tc),
                                     tc.layout);
    }
};

} // namespace gpu_conv_wgrad_ref_test
