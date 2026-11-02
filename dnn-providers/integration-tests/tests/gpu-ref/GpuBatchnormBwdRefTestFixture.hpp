// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BatchnormTestCase.hpp"
#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceBatchnorm.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <tuple>
#include <vector>

namespace gpu_batchnorm_bwd_ref_test
{

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::batchnorm;
using namespace hipdnn_gpu_ref;
using namespace gpu_batchnorm_ref_test;

template <typename DyDataType,
          typename XDataType,
          typename ScaleBiasDataType,
          typename MeanVarianceDataType,
          typename DxDataType,
          typename ComputeDataType>
void runGpuVsCpuBatchnormBackward(const std::vector<int64_t>& dims,
                                  const TensorLayout& layout,
                                  bool useSavedStats)
{
    constexpr double EPSILON = 1e-5;
    std::vector<int64_t> affineDims(dims.size(), 1);
    affineDims[1] = dims[1];

    Tensor<DyDataType> dy(dims, layout);
    Tensor<XDataType> x(dims, layout);
    Tensor<ScaleBiasDataType> scale(affineDims, layout);
    Tensor<DxDataType> dxCpu(dims, layout);
    Tensor<DxDataType> dxGpu(dims, layout);
    Tensor<ScaleBiasDataType> dscaleCpu(affineDims, layout);
    Tensor<ScaleBiasDataType> dscaleGpu(affineDims, layout);
    Tensor<ScaleBiasDataType> dbiasCpu(affineDims, layout);
    Tensor<ScaleBiasDataType> dbiasGpu(affineDims, layout);
    auto mean = useSavedStats ? Tensor<MeanVarianceDataType>(affineDims, layout)
                              : Tensor<MeanVarianceDataType>({});
    auto invVariance = useSavedStats ? Tensor<MeanVarianceDataType>(affineDims, layout)
                                     : Tensor<MeanVarianceDataType>({});

    const auto seed = getGlobalTestSeed();
    dy.fillWithRandomValues(static_cast<DyDataType>(-1.0f), static_cast<DyDataType>(1.0f), seed);
    x.fillWithRandomValues(static_cast<XDataType>(-1.0f), static_cast<XDataType>(1.0f), seed + 1);
    scale.fillWithRandomValues(
        static_cast<ScaleBiasDataType>(-1.0f), static_cast<ScaleBiasDataType>(1.0f), seed + 2);
    if(useSavedStats)
    {
        mean.fillWithRandomValues(static_cast<MeanVarianceDataType>(-1.0f),
                                  static_cast<MeanVarianceDataType>(1.0f),
                                  seed + 3);
        invVariance.fillWithRandomValues(static_cast<MeanVarianceDataType>(0.25f),
                                         static_cast<MeanVarianceDataType>(2.0f),
                                         seed + 4);
    }

    CpuFpReferenceBatchnorm::backward<DyDataType,
                                      XDataType,
                                      ScaleBiasDataType,
                                      MeanVarianceDataType,
                                      DxDataType,
                                      ComputeDataType>(dy,
                                                       x,
                                                       scale,
                                                       dxCpu,
                                                       dscaleCpu,
                                                       dbiasCpu,
                                                       useSavedStats ? &mean : nullptr,
                                                       useSavedStats ? &invVariance : nullptr,
                                                       EPSILON);
    GpuFpReferenceBatchnorm::backward<DyDataType,
                                      XDataType,
                                      ScaleBiasDataType,
                                      MeanVarianceDataType,
                                      DxDataType,
                                      ComputeDataType>(dy,
                                                       x,
                                                       scale,
                                                       dxGpu,
                                                       dscaleGpu,
                                                       dbiasGpu,
                                                       useSavedStats ? &mean : nullptr,
                                                       useSavedStats ? &invVariance : nullptr,
                                                       EPSILON);

    assertAllClose(dxCpu, dxGpu, getToleranceBackward<DxDataType>());
    assertAllClose(dscaleCpu, dscaleGpu, getToleranceBackward<ScaleBiasDataType>());
    assertAllClose(dbiasCpu, dbiasGpu, getToleranceBackward<ScaleBiasDataType>());
}

using BatchnormBwdTestParam = std::tuple<TensorLayout, BatchnormTestCase, bool>;

template <typename DyDataType,
          typename XDataType,
          typename ScaleBiasDataType,
          typename MeanVarianceDataType,
          typename DxDataType,
          typename ComputeDataType>
class BatchnormBwdTestSuite : public ::testing::TestWithParam<BatchnormBwdTestParam>
{
protected:
    void runBatchnormBwdTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& [layout, testCase, useSavedStats] = GetParam();
        runGpuVsCpuBatchnormBackward<DyDataType,
                                     XDataType,
                                     ScaleBiasDataType,
                                     MeanVarianceDataType,
                                     DxDataType,
                                     ComputeDataType>(testCase.ioDims, layout, useSavedStats);
    }
};

} // namespace gpu_batchnorm_bwd_ref_test
