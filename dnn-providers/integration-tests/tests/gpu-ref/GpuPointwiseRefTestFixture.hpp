// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "PointwiseShapeCase.hpp"
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>

#include <hipdnn-gpu-ref/GpuReferencePointwise.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/CpuReferencePointwise.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/PointwiseErrorClassification.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/ReferencePointwiseBase.hpp>

namespace gpu_pointwise_ref_test
{

using namespace hipdnn_gpu_ref;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;

// Calculates tolerance based on the operation and input data range.
template <typename InputType, typename OutputType = InputType>
float getDynamicTolerance(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                          float scale)
{
    auto errorClass = pointwise::classifyPointwiseOp(operation);
    auto effectiveScale = static_cast<double>(pointwise::isBoundedOutput(operation) ? 1.0f : scale);
    return pointwise::calculatePointwiseTolerance<OutputType, InputType>(effectiveScale,
                                                                         errorClass);
}

template <typename DataType>
void runGpuVsCpuPointwiseUnary(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                               const std::vector<int64_t>& ioDims,
                               float fillRange = 1.0f)
{
    const unsigned int seed = getGlobalTestSeed();
    auto inputTensor = Tensor<DataType>(ioDims);
    auto outputCpu = Tensor<DataType>(ioDims);
    auto outputGpu = Tensor<DataType>(ioDims);

    inputTensor.fillWithRandomValues(
        static_cast<DataType>(-fillRange), static_cast<DataType>(fillRange), seed);

    CpuReferencePointwiseImpl<DataType, DataType>::pointwiseCompute(
        operation, outputCpu, inputTensor);

    GpuReferencePointwise::pointwiseCompute<DataType, DataType>(operation, outputGpu, inputTensor);

    assertAllClose(outputCpu, outputGpu, getDynamicTolerance<DataType>(operation, fillRange));
}

template <typename DataType>
void runGpuVsCpuPointwiseBinary(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                                const std::vector<int64_t>& ioDims,
                                float fillRange = 1.0f)
{
    const unsigned int seed = getGlobalTestSeed();
    auto input0Tensor = Tensor<DataType>(ioDims);
    auto input1Tensor = Tensor<DataType>(ioDims);

    auto outputCpu = Tensor<DataType>(ioDims);
    auto outputGpu = Tensor<DataType>(ioDims);

    input0Tensor.fillWithRandomValues(
        static_cast<DataType>(-fillRange), static_cast<DataType>(fillRange), seed);
    input1Tensor.fillWithRandomValues(
        static_cast<DataType>(-fillRange), static_cast<DataType>(fillRange), seed);

    CpuReferencePointwiseImpl<DataType, DataType, DataType>::pointwiseCompute(
        operation, outputCpu, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<DataType, DataType, DataType>(
        operation, outputGpu, input0Tensor, input1Tensor);

    assertAllClose(outputCpu, outputGpu, getDynamicTolerance<DataType>(operation, fillRange));
}

template <typename DataType>
class PointwiseTestSuite
    : public ::testing::TestWithParam<gpu_pointwise_ref_test::PointwiseTestCase>
{
protected:
    void runPointwiseUnaryTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& tc = GetParam();
        runGpuVsCpuPointwiseUnary<DataType>(tc.operation, tc.ioDims);
    }

    void runPointwiseBinaryTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& tc = GetParam();
        runGpuVsCpuPointwiseBinary<DataType>(tc.operation, tc.ioDims);
    }
};

} // namespace gpu_pointwise_ref_test
