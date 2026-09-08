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

// ============================================================================
// Shared infrastructure for backward RMSNorm GPU-vs-CPU reference tests.
// ============================================================================

namespace gpu_rmsnorm_bwd_ref_test
{

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::rmsnorm;
using namespace hipdnn_gpu_ref;
using namespace gpu_rmsnorm_ref_test;

template <typename GradOutputDataType,
          typename InputDataType = GradOutputDataType,
          typename ScaleDataType = float,
          typename GradInputDataType = GradOutputDataType,
          typename ComputeDataType = double>
void runGpuVsCpuRMSNormBwd(const std::vector<int64_t>& ioDims,
                           const std::vector<int64_t>& scaleDims,
                           const TensorLayout& layout,
                           float fillRange = 1.0f,
                           bool includeBias = false)
{
    const unsigned int seed = getGlobalTestSeed();

    auto dyTensor = Tensor<GradOutputDataType>(ioDims, layout);
    auto xTensor = Tensor<InputDataType>(ioDims, layout);
    auto scaleTensor = Tensor<ScaleDataType>(scaleDims, layout);
    auto dxCpu = Tensor<GradInputDataType>(ioDims, layout);
    auto dxGpu = Tensor<GradInputDataType>(ioDims, layout);
    auto dscaleCpu = Tensor<ScaleDataType>(scaleDims, layout);
    auto dscaleGpu = Tensor<ScaleDataType>(scaleDims, layout);

    dyTensor.fillWithRandomValues(static_cast<GradOutputDataType>(-fillRange),
                                  static_cast<GradOutputDataType>(fillRange),
                                  seed);
    xTensor.fillWithRandomValues(
        static_cast<InputDataType>(-fillRange), static_cast<InputDataType>(fillRange), seed + 1);
    scaleTensor.fillWithRandomValues(
        static_cast<ScaleDataType>(-fillRange), static_cast<ScaleDataType>(fillRange), seed + 2);

    std::vector<int64_t> invRmsDims = ioDims;
    for(size_t i = 0; i < invRmsDims.size(); ++i)
    {
        if(scaleDims[i] != 1)
        {
            invRmsDims[i] = 1;
        }
    }
    auto invRmsTensor = Tensor<ComputeDataType>(invRmsDims, layout);
    invRmsTensor.fillWithRandomValues(static_cast<ComputeDataType>(1.0e-05f),
                                      static_cast<ComputeDataType>(std::fabs(fillRange)),
                                      seed + 3); // Ensure invRms is always positive!

    auto dbiasCpu
        = includeBias ? Tensor<ScaleDataType>(scaleDims, layout) : Tensor<ScaleDataType>({});
    auto dbiasGpu
        = includeBias ? Tensor<ScaleDataType>(scaleDims, layout) : Tensor<ScaleDataType>({});

    CpuFpReferenceRMSNorm::backward<GradOutputDataType,
                                    InputDataType,
                                    ScaleDataType,
                                    GradInputDataType,
                                    ComputeDataType>(dyTensor,
                                                     xTensor,
                                                     scaleTensor,
                                                     invRmsTensor,
                                                     dxCpu,
                                                     dscaleCpu,
                                                     includeBias ? &dbiasCpu : nullptr);

    GpuFpReferenceRMSNorm::
        bprop<GradOutputDataType, InputDataType, ScaleDataType, GradInputDataType, ComputeDataType>(
            dyTensor,
            xTensor,
            scaleTensor,
            invRmsTensor,
            dxGpu,
            dscaleGpu,
            includeBias ? &dbiasGpu : nullptr);

    assertAllClose(dxCpu, dxGpu, getTolerance<GradInputDataType>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<ScaleDataType>());
    if(includeBias)
    {
        assertAllClose(dbiasCpu, dbiasGpu, getTolerance<ScaleDataType>());
    }
}

// ============================================================================
// RMSNormBwdTestSuite — parameterized fixture for shape-based CPU-vs-GPU tests
// ============================================================================

template <typename DataType>
class RMSNormBwdShapeSuite : public ::testing::TestWithParam<RMSNormTestCase>
{
protected:
    void runRMSNormBwdShapeTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& tc = GetParam();
        runGpuVsCpuRMSNormBwd<DataType>(tc.ioDims, tc.scaleDims, tc.layout, 1.0f, true);
    }
};

} // namespace gpu_rmsnorm_bwd_ref_test
