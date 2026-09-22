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

namespace gpu_batchnorm_fwd_train_ref_test
{

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::batchnorm;
using namespace hipdnn_gpu_ref;
using namespace gpu_batchnorm_ref_test;

template <typename InputDataType,
          typename OutputDataType,
          typename ScaleBiasDataType,
          typename MeanVarDataType,
          typename ComputeDataType>
void runGpuVsCpuBatchnormFwdTrain(const std::vector<int64_t>& ioDims,
                                  const std::vector<int64_t>& affineDims,
                                  const TensorLayout& layout,
                                  float fillRange = 1.0f,
                                  bool includeSaveStats = false,
                                  bool includeRunningStats = false)
{
    constexpr double EPSILON = 1e-5;
    constexpr double MOMENTUM = 0.1;

    auto inputTensor = Tensor<InputDataType>(ioDims, layout);
    auto scaleTensor = Tensor<ScaleBiasDataType>(affineDims, layout);
    auto biasTensor = Tensor<ScaleBiasDataType>(affineDims, layout);
    auto outputCpu = Tensor<OutputDataType>(ioDims, layout);
    auto outputGpu = Tensor<OutputDataType>(ioDims, layout);
    auto meanTensorCpu = includeSaveStats ? Tensor<MeanVarDataType>(affineDims, layout)
                                          : Tensor<MeanVarDataType>({});
    auto invVarTensorCpu = includeSaveStats ? Tensor<MeanVarDataType>(affineDims, layout)
                                            : Tensor<MeanVarDataType>({});
    auto meanTensorGpu = includeSaveStats ? Tensor<MeanVarDataType>(affineDims, layout)
                                          : Tensor<MeanVarDataType>({});
    auto invVarTensorGpu = includeSaveStats ? Tensor<MeanVarDataType>(affineDims, layout)
                                            : Tensor<MeanVarDataType>({});
    auto prevRunningMeanTensor = includeRunningStats ? Tensor<MeanVarDataType>(affineDims, layout)
                                                     : Tensor<MeanVarDataType>({});
    auto prevRunningVarTensor = includeRunningStats ? Tensor<MeanVarDataType>(affineDims, layout)
                                                    : Tensor<MeanVarDataType>({});
    auto nextRunningMeanTensorCpu = includeRunningStats
                                        ? Tensor<MeanVarDataType>(affineDims, layout)
                                        : Tensor<MeanVarDataType>({});
    auto nextRunningVarTensorCpu = includeRunningStats ? Tensor<MeanVarDataType>(affineDims, layout)
                                                       : Tensor<MeanVarDataType>({});
    auto nextRunningMeanTensorGpu = includeRunningStats
                                        ? Tensor<MeanVarDataType>(affineDims, layout)
                                        : Tensor<MeanVarDataType>({});
    auto nextRunningVarTensorGpu = includeRunningStats ? Tensor<MeanVarDataType>(affineDims, layout)
                                                       : Tensor<MeanVarDataType>({});

    const auto seed = getGlobalTestSeed();
    inputTensor.fillWithRandomValues(
        static_cast<InputDataType>(-fillRange), static_cast<InputDataType>(fillRange), seed);
    scaleTensor.fillWithRandomValues(static_cast<ScaleBiasDataType>(-fillRange),
                                     static_cast<ScaleBiasDataType>(fillRange),
                                     seed + 1);
    biasTensor.fillWithRandomValues(static_cast<ScaleBiasDataType>(-fillRange),
                                    static_cast<ScaleBiasDataType>(fillRange),
                                    seed + 2);
    if(includeRunningStats)
    {
        prevRunningMeanTensor.fillWithRandomValues(static_cast<MeanVarDataType>(-fillRange),
                                                   static_cast<MeanVarDataType>(fillRange),
                                                   seed + 3);
        prevRunningVarTensor.fillWithRandomValues(
            static_cast<MeanVarDataType>(1.0e-05f),
            static_cast<MeanVarDataType>(std::fabs(fillRange)),
            seed + 4); // Ensure variance stays positive!
    }

    CpuFpReferenceBatchnorm::fwdTraining<InputDataType,
                                         ScaleBiasDataType,
                                         MeanVarDataType,
                                         OutputDataType,
                                         ComputeDataType>(
        inputTensor,
        scaleTensor,
        biasTensor,
        outputCpu,
        EPSILON,
        MOMENTUM,
        includeSaveStats ? &meanTensorCpu : nullptr,
        includeSaveStats ? &invVarTensorCpu : nullptr,
        includeRunningStats ? &prevRunningMeanTensor : nullptr,
        includeRunningStats ? &prevRunningVarTensor : nullptr,
        includeRunningStats ? &nextRunningMeanTensorCpu : nullptr,
        includeRunningStats ? &nextRunningVarTensorCpu : nullptr);

    GpuFpReferenceBatchnorm::fwdTraining<InputDataType,
                                         ScaleBiasDataType,
                                         MeanVarDataType,
                                         OutputDataType,
                                         ComputeDataType>(
        inputTensor,
        scaleTensor,
        biasTensor,
        outputGpu,
        EPSILON,
        MOMENTUM,
        includeSaveStats ? &meanTensorGpu : nullptr,
        includeSaveStats ? &invVarTensorGpu : nullptr,
        includeRunningStats ? &prevRunningMeanTensor : nullptr,
        includeRunningStats ? &prevRunningVarTensor : nullptr,
        includeRunningStats ? &nextRunningMeanTensorGpu : nullptr,
        includeRunningStats ? &nextRunningVarTensorGpu : nullptr);

    assertAllClose(outputCpu, outputGpu, getToleranceTraining<OutputDataType>());
    if(includeSaveStats)
    {
        assertAllClose(meanTensorCpu, meanTensorGpu, getToleranceTraining<MeanVarDataType>());
        assertAllClose(invVarTensorCpu, invVarTensorGpu, getToleranceTraining<MeanVarDataType>());
    }
    if(includeRunningStats)
    {
        assertAllClose(nextRunningMeanTensorCpu,
                       nextRunningMeanTensorGpu,
                       getToleranceTraining<MeanVarDataType>());
        assertAllClose(nextRunningVarTensorCpu,
                       nextRunningVarTensorGpu,
                       getToleranceTraining<MeanVarDataType>());
    }
}

template <typename InputDataType,
          typename OutputDataType = InputDataType,
          typename ScaleBiasDataType = InputDataType,
          typename MeanVarDataType = InputDataType,
          typename ComputeDataType = double>
class BatchnormFwdTrainTestSuite
    : public ::testing::TestWithParam<std::tuple<TensorLayout, BatchnormTestCase>>
{
protected:
    void runBatchnormFwdTrainTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& tc = GetParam();
        const auto& [layout, bnTestCase] = tc;
        const auto& ioDims = bnTestCase.ioDims;
        std::vector<int64_t> affineDims(ioDims.size(), 1);
        affineDims[1] = ioDims[1];
        runGpuVsCpuBatchnormFwdTrain<InputDataType,
                                     OutputDataType,
                                     ScaleBiasDataType,
                                     MeanVarDataType,
                                     ComputeDataType>(ioDims, affineDims, layout, 1.0f, true, true);
    }
};

} // namespace gpu_batchnorm_fwd_train_ref_test
