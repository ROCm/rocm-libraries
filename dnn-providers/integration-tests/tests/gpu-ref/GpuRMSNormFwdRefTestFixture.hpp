// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "RMSNormShapeCatalog.hpp"
#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceRMSNorm.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceRMSNorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

namespace gpu_rmsnorm_fwd_ref_test
{

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::rmsnorm;
using namespace hipdnn_gpu_ref;
using namespace gpu_rmsnorm_ref_test;

template <typename InputDataType,
          typename OutputDataType = InputDataType,
          typename ScaleDataType = InputDataType,
          typename ComputeDataType = double>
void runGpuVsCpuRMSNormFwd(const std::vector<int64_t>& ioDims,
                           const std::vector<int64_t>& scaleDims,
                           const TensorLayout& layout,
                           double epsilon,
                           float fillRange = 1.0f,
                           bool includeInvRms = false,
                           bool includeBias = false)
{
    const unsigned int seed = getGlobalTestSeed();

    auto inputTensor = Tensor<InputDataType>(ioDims, layout);
    auto scaleTensor = Tensor<ScaleDataType>(scaleDims, layout);
    auto outputCpu = Tensor<OutputDataType>(ioDims, layout);
    auto outputGpu = Tensor<OutputDataType>(ioDims, layout);

    inputTensor.fillWithRandomValues(
        static_cast<InputDataType>(-fillRange), static_cast<InputDataType>(fillRange), seed);
    scaleTensor.fillWithRandomValues(
        static_cast<ScaleDataType>(-fillRange), static_cast<ScaleDataType>(fillRange), seed + 1);

    std::vector<int64_t> invRmsDims = ioDims;
    for(size_t i = 0; i < invRmsDims.size(); ++i)
    {
        if(scaleDims[i] != 1)
        {
            invRmsDims[i] = 1;
        }
    }
    auto invRmsTensorCpu
        = includeInvRms ? Tensor<ComputeDataType>(invRmsDims, layout) : Tensor<ComputeDataType>({});
    auto invRmsTensorGpu
        = includeInvRms ? Tensor<ComputeDataType>(invRmsDims, layout) : Tensor<ComputeDataType>({});

    auto biasTensor
        = includeBias ? Tensor<ScaleDataType>(scaleDims, layout) : Tensor<ScaleDataType>({});
    if(includeBias)
    {
        biasTensor.fillWithRandomValues(static_cast<ScaleDataType>(-fillRange),
                                        static_cast<ScaleDataType>(fillRange),
                                        seed + 2);
    }

    CpuFpReferenceRMSNorm::forward<InputDataType, ScaleDataType, OutputDataType, ComputeDataType>(
        inputTensor,
        scaleTensor,
        outputCpu,
        epsilon,
        includeInvRms ? &invRmsTensorCpu : nullptr,
        includeBias ? &biasTensor : nullptr);

    GpuFpReferenceRMSNorm::fprop<InputDataType, ScaleDataType, OutputDataType, ComputeDataType>(
        inputTensor,
        scaleTensor,
        outputGpu,
        epsilon,
        includeInvRms ? &invRmsTensorGpu : nullptr,
        includeBias ? &biasTensor : nullptr);

    assertAllClose(outputCpu, outputGpu, getTolerance<OutputDataType>());
    if(includeInvRms)
    {
        assertAllClose(invRmsTensorCpu, invRmsTensorGpu, getTolerance<ComputeDataType>());
    }
}

// ============================================================================
// RMSNormFwdTestSuite — parameterized fixture for shape-based CPU-vs-GPU tests
// ============================================================================

template <typename DataType>
class RMSNormFwdTestSuite : public ::testing::TestWithParam<RMSNormTestCase>
{
protected:
    void runRMSNormFwdTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& tc = GetParam();
        runGpuVsCpuRMSNormFwd<DataType>(tc.ioDims, tc.scaleDims, tc.layout, 1e-5, 1.0f, true, true);
    }
};

} // namespace gpu_rmsnorm_fwd_ref_test
