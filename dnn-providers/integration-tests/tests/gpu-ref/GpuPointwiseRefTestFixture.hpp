// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "PointwiseShapeCase.hpp"
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>

#include <hipdnn-gpu-ref/GpuFpReferenceCommon.hpp>
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
using namespace hipdnn_gpu_ref::common::gpu_fp_reference_tensor;
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

    // LOG, SQRT, RSQRT and RECIPROCAL are only defined on positive values, and RECIPROCAL and
    // RSQRT blow up near zero, so keep their inputs away from it
    const bool positiveDomain
        = operation == hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::LOG
          || operation == hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SQRT
          || operation == hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RSQRT
          || operation == hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RECIPROCAL;
    const float fillMin = positiveDomain ? 0.25f * fillRange : -fillRange;
    fillWithRandomValues(
        inputTensor, static_cast<DataType>(fillMin), static_cast<DataType>(fillRange), seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    inputTensor.memory().hostData();

    CpuReferencePointwiseImpl<DataType, double, DataType>::pointwiseCompute(
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

    fillWithRandomValues(
        input0Tensor, static_cast<DataType>(-fillRange), static_cast<DataType>(fillRange), seed);
    fillWithRandomValues(
        input1Tensor, static_cast<DataType>(-fillRange), static_cast<DataType>(fillRange), seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    CpuReferencePointwiseImpl<DataType, double, DataType, DataType>::pointwiseCompute(
        operation, outputCpu, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<DataType, DataType, DataType>(
        operation, outputGpu, input0Tensor, input1Tensor);

    assertAllClose(outputCpu, outputGpu, getDynamicTolerance<DataType>(operation, fillRange));
}

// Fills a mask with a deterministic mix of true and false.
inline void fillMaskWithAlternatingValues(Tensor<bool>& mask)
{
    auto* data = mask.memory().hostData();
    for(size_t i = 0; i < mask.elementSpace(); ++i)
    {
        data[i] = ((i * 7 + i / 3) % 2) == 0;
    }
    mask.memory().markHostModified();
}

template <typename DataType>
void runGpuVsCpuPointwiseTernary(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                                 const std::vector<int64_t>& ioDims,
                                 float fillRange = 1.0f)
{
    const unsigned int seed = getGlobalTestSeed();
    auto input0Tensor = Tensor<DataType>(ioDims);
    auto input1Tensor = Tensor<DataType>(ioDims);
    auto maskTensor = Tensor<bool>(ioDims);

    auto outputCpu = Tensor<DataType>(ioDims);
    auto outputGpu = Tensor<DataType>(ioDims);

    fillWithRandomValues(
        input0Tensor, static_cast<DataType>(-fillRange), static_cast<DataType>(fillRange), seed);
    fillWithRandomValues(input1Tensor,
                         static_cast<DataType>(-fillRange),
                         static_cast<DataType>(fillRange),
                         seed + 1);
    fillMaskWithAlternatingValues(maskTensor);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    CpuReferencePointwiseImpl<DataType, double, DataType, DataType, bool>::pointwiseCompute(
        operation, outputCpu, input0Tensor, input1Tensor, maskTensor);

    GpuReferencePointwise::pointwiseCompute<DataType, DataType, DataType, bool>(
        operation, outputGpu, input0Tensor, input1Tensor, maskTensor);

    // Selection copies a value, so the result is exact
    assertAllEqual(outputCpu, outputGpu);
}

// Runs a unary mode on inputs that random fills never produce: signed zeros, NaN, infinities and a
// negative number, which is outside the domain of LOG and SQRT. The GPU result must match the CPU
// reference, including NaN and infinity.
template <typename DataType>
void runGpuVsCpuPointwiseUnarySpecialValues(
    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> values = {0.0f, -0.0f, -1.0f, nan, inf, -inf};
    const std::vector<int64_t> dims = {1, 1, 1, static_cast<int64_t>(values.size())};

    Tensor<DataType> inputTensor(dims);
    Tensor<DataType> outputCpuTensor(dims);
    Tensor<DataType> outputGpuTensor(dims);
    for(size_t i = 0; i < values.size(); ++i)
    {
        inputTensor.setHostValue(
            static_cast<DataType>(values[i]), 0, 0, 0, static_cast<int64_t>(i));
    }

    CpuReferencePointwiseImpl<DataType, double, DataType>::pointwiseCompute(
        operation, outputCpuTensor, inputTensor);

    GpuReferencePointwise::pointwiseCompute<DataType, DataType>(
        operation, outputGpuTensor, inputTensor);

    assertAllCloseWithSpecialValues(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<DataType>(operation, 5.0f));
}

// Runs a binary mode on values that rounding or NaN ordering could expose: NaN in either operand,
// infinities and signed zeros. The result must match the CPU reference exactly.
template <typename OutputType>
void runGpuVsCpuPointwiseBinarySpecialValues(
    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> values0 = {nan, 1.0f, nan, -inf, 2.0f, inf, -0.0f, 3.0f};
    const std::vector<float> values1 = {1.0f, nan, nan, inf, -inf, inf, 0.0f, 3.0f};
    const std::vector<int64_t> dims = {1, 1, 1, static_cast<int64_t>(values0.size())};

    Tensor<float> input0Tensor(dims);
    Tensor<float> input1Tensor(dims);
    Tensor<OutputType> outputCpuTensor(dims);
    Tensor<OutputType> outputGpuTensor(dims);
    for(size_t i = 0; i < values0.size(); ++i)
    {
        input0Tensor.setHostValue(values0[i], 0, 0, 0, static_cast<int64_t>(i));
        input1Tensor.setHostValue(values1[i], 0, 0, 0, static_cast<int64_t>(i));
    }

    CpuReferencePointwiseImpl<OutputType, double, float, float>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<OutputType, float, float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllEqual(outputCpuTensor, outputGpuTensor);
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

    void runPointwiseUnarySpecialValuesTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& tc = GetParam();
        runGpuVsCpuPointwiseUnarySpecialValues<DataType>(tc.operation);
    }

    void runPointwiseBinaryTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& tc = GetParam();
        runGpuVsCpuPointwiseBinary<DataType>(tc.operation, tc.ioDims);
    }

    void runPointwiseTernaryTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& tc = GetParam();
        runGpuVsCpuPointwiseTernary<DataType>(tc.operation, tc.ioDims);
    }
};

} // namespace gpu_pointwise_ref_test
