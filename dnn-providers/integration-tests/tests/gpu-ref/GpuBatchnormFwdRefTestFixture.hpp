// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BatchnormShapeCatalog.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceBatchnorm.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <vector>

namespace gpu_batchnorm_fwd_ref_test
{

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::batchnorm;
using namespace hipdnn_gpu_ref;
using namespace gpu_batchnorm_ref_test;

template <typename InputDataType,
          typename OutputDataType = InputDataType,
          typename ScaleBiasDataType = InputDataType,
          typename MeanVarDataType = InputDataType,
          typename ComputeDataType = double>
void runGpuVsCpuBatchnormFwd(const std::vector<int64_t>& ioDims,
                             const TensorLayout& layout,
                             float fillRange = 1.0f)
{
    unsigned int seed = getGlobalTestSeed();

    std::vector<int64_t> affineDims(ioDims.size(), 1);
    affineDims[1] = ioDims[1];

    auto inputTensor = Tensor<InputDataType>(ioDims, layout);
    auto scaleTensor = Tensor<ScaleBiasDataType>(affineDims, layout);
    auto biasTensor = Tensor<ScaleBiasDataType>(affineDims, layout);
    auto estimatedMeanTensor = Tensor<MeanVarDataType>(affineDims, layout);
    auto invVarTensor = Tensor<MeanVarDataType>(affineDims, layout);
    auto outputCpu = Tensor<OutputDataType>(ioDims, layout);
    auto outputGpu = Tensor<OutputDataType>(ioDims, layout);

    inputTensor.fillWithRandomValues(
        static_cast<InputDataType>(-fillRange), static_cast<InputDataType>(fillRange), seed++);
    scaleTensor.fillWithRandomValues(static_cast<ScaleBiasDataType>(-fillRange),
                                     static_cast<ScaleBiasDataType>(fillRange),
                                     seed++);
    biasTensor.fillWithRandomValues(static_cast<ScaleBiasDataType>(-fillRange),
                                    static_cast<ScaleBiasDataType>(fillRange),
                                    seed++);
    estimatedMeanTensor.fillWithRandomValues(
        static_cast<MeanVarDataType>(-fillRange), static_cast<MeanVarDataType>(fillRange), seed++);
    invVarTensor.fillWithRandomValues(
        static_cast<MeanVarDataType>(-fillRange), static_cast<MeanVarDataType>(fillRange), seed++);

    CpuFpReferenceBatchnorm::fwdInference<InputDataType,
                                          ScaleBiasDataType,
                                          MeanVarDataType,
                                          OutputDataType,
                                          ComputeDataType>(
        inputTensor, scaleTensor, biasTensor, estimatedMeanTensor, invVarTensor, outputCpu);

    GpuFpReferenceBatchnorm::fwdInference<InputDataType,
                                          ScaleBiasDataType,
                                          MeanVarDataType,
                                          OutputDataType,
                                          ComputeDataType>(
        inputTensor, scaleTensor, biasTensor, estimatedMeanTensor, invVarTensor, outputGpu);

    assertAllClose(outputCpu, outputGpu, getToleranceInference<OutputDataType>());
}

// ============================================================================
// BatchnormFwdTestSuite — parameterized fixture for shape-based CPU-vs-GPU tests
// ============================================================================

using BnFwdInfTestCase = std::tuple<TensorLayout, BatchnormTestCase>;

template <typename InputDataType,
          typename OutputDataType = InputDataType,
          typename ScaleBiasDataType = InputDataType,
          typename MeanVarDataType = InputDataType,
          typename ComputeDataType = double>
class BatchnormFwdTestSuite : public ::testing::TestWithParam<BnFwdInfTestCase>
{
protected:
    void runBatchnormFwdTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& tc = GetParam();
        const auto& [layout, bnTestCase] = tc;
        runGpuVsCpuBatchnormFwd<InputDataType,
                                OutputDataType,
                                ScaleBiasDataType,
                                MeanVarDataType,
                                ComputeDataType>(bnTestCase.ioDims, layout, 1e-5f);
    }
};

} // namespace gpu_batchnorm_fwd_ref_test
