// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ReductionShapeCatalog.hpp"
#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceReduction.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceReduction.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

namespace gpu_reduction_ref_test
{

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_gpu_ref;

template <typename InputDataType,
          typename OutputDataType = InputDataType,
          typename ComputeDataType = double>
void runGpuVsCpuReduction(const std::vector<int64_t>& inputDims,
                          const std::vector<int64_t>& outputDims,
                          const TensorLayout& layout,
                          hipdnn_flatbuffers_sdk::data_objects::ReductionMode mode,
                          float fillRange = 1.0f)
{
    const unsigned int seed = getGlobalTestSeed();

    auto inputTensor = Tensor<InputDataType>(inputDims, layout);
    auto outputTensorCpu = Tensor<OutputDataType>(outputDims, layout);
    auto outputTensorGpu = Tensor<OutputDataType>(outputDims, layout);

    inputTensor.fillWithRandomValues(
        static_cast<InputDataType>(-fillRange), static_cast<InputDataType>(fillRange), seed);

    CpuFpReferenceReduction::reduce<InputDataType, OutputDataType, ComputeDataType>(
        inputTensor, outputTensorCpu, mode);
    GpuFpReferenceReduction::reduce<InputDataType, OutputDataType, ComputeDataType>(
        inputTensor, outputTensorGpu, mode);

    assertAllClose(outputTensorCpu, outputTensorGpu, reduction::getTolerance<OutputDataType>());
}

// ===========================================================================
// ReductionTestSuite — parameterized fixture for shape-based CPU-vs-GPU tests
// ===========================================================================

template <typename DataType>
class ReductionTestSuite : public ::testing::TestWithParam<ReductionTestCase>
{
protected:
    void runReductionTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& tc = GetParam();
        runGpuVsCpuReduction<DataType>(tc.inputDims, tc.outputDims, tc.layout, tc.mode, 1.0f);
    }
};

} // namespace gpu_reduction_ref_test
