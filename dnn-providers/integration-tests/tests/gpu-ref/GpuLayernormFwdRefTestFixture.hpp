// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "LayernormShapeCatalog.hpp"
#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceLayernorm.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceLayernorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

namespace gpu_layernorm_fwd_ref_test
{

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::layernorm;
using namespace hipdnn_gpu_ref;
using namespace gpu_layernorm_ref_test;

template <typename XDataType,
          class YDataType = XDataType,
          class ScaleBiasDataType = XDataType,
          class MeanRstdDataType = ScaleBiasDataType,
          class ComputeDataType = double>
void runGpuVsCpuLayernormFwd(const std::vector<int64_t>& ioDims,
                             const std::vector<int64_t>& scaleBiasDims,
                             const std::vector<int64_t>& meanRstdDims,
                             const double epsilon,
                             const int64_t normalizedDimCount,
                             const TensorLayout& layout,
                             const float tolerance,
                             const bool optionalTensors = true,
                             const float fillRange = 1.0f)
{
    const unsigned int seed = getGlobalTestSeed();

    auto xTensor = Tensor<XDataType>(ioDims, layout);
    xTensor.fillWithRandomValues(
        static_cast<XDataType>(-fillRange), static_cast<XDataType>(fillRange), seed);
    auto scaleTensor = Tensor<ScaleBiasDataType>(scaleBiasDims, layout);
    scaleTensor.fillWithRandomValues(static_cast<ScaleBiasDataType>(-fillRange),
                                     static_cast<ScaleBiasDataType>(fillRange),
                                     seed + 1);
    auto biasTensor = Tensor<ScaleBiasDataType>(scaleBiasDims, layout);
    biasTensor.fillWithRandomValues(static_cast<ScaleBiasDataType>(-fillRange),
                                    static_cast<ScaleBiasDataType>(fillRange),
                                    seed + 2);
    auto yCpu = Tensor<YDataType>(ioDims, layout);
    auto yGpu = Tensor<YDataType>(ioDims, layout);
    auto meanCpu = optionalTensors ? Tensor<MeanRstdDataType>(meanRstdDims, layout)
                                   : Tensor<MeanRstdDataType>({});
    auto meanGpu = optionalTensors ? Tensor<MeanRstdDataType>(meanRstdDims, layout)
                                   : Tensor<MeanRstdDataType>({});
    auto rstdCpu = optionalTensors ? Tensor<MeanRstdDataType>(meanRstdDims, layout)
                                   : Tensor<MeanRstdDataType>({});
    auto rstdGpu = optionalTensors ? Tensor<MeanRstdDataType>(meanRstdDims, layout)
                                   : Tensor<MeanRstdDataType>({});

    GpuFpReferenceLayernorm::
        fprop<XDataType, ScaleBiasDataType, YDataType, MeanRstdDataType, ComputeDataType>(
            xTensor,
            &scaleTensor,
            &biasTensor,
            yGpu,
            epsilon,
            normalizedDimCount,
            optionalTensors ? &meanGpu : nullptr,
            optionalTensors ? &rstdGpu : nullptr);

    CpuFpReferenceLayernorm::
        fprop<XDataType, ScaleBiasDataType, YDataType, MeanRstdDataType, ComputeDataType>(
            xTensor,
            &scaleTensor,
            &biasTensor,
            yCpu,
            epsilon,
            normalizedDimCount,
            optionalTensors ? &meanCpu : nullptr,
            optionalTensors ? &rstdCpu : nullptr);

    assertAllClose(yCpu, yGpu, tolerance, "y");
    if(optionalTensors)
    {
        assertAllClose(meanCpu, meanGpu, tolerance, "mean");
        assertAllClose(rstdCpu, rstdGpu, tolerance, "rstd");
    }
}

// ===============================================================================
// LayernormFwdShapeSuite — parameterized fixture for shape-based GPU-vs-CPU tests
// ===============================================================================

template <typename InputDataType,
          typename OutputDataType,
          typename ScaleBiasDataType,
          typename MeanInvVarianceDataType>
class LayernormFwdTestSuite
    : public ::testing::TestWithParam<gpu_layernorm_ref_test::LayernormTestCase>
{
protected:
    void runLayernormFwdTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& testCase = GetParam();
        runGpuVsCpuLayernormFwd<InputDataType,
                                OutputDataType,
                                ScaleBiasDataType,
                                MeanInvVarianceDataType>(
            testCase.dims,
            testCase.computeNormalizedDims(),
            testCase.computeBatchDims(),
            LAYERNORM_DEFAULT_EPSILON,
            testCase.computeNormalizedDimCount(),
            testCase.layout,
            hipdnn_test_sdk::utilities::layernorm::getTolerance<InputDataType>(),
            testCase.optionalTensors);
    }
};

// "Pure" = input, output, scale/bias, and mean/invvariance all the same type.
template <typename DataType>
using LayernormPureFwdTestSuite = LayernormFwdTestSuite<DataType, DataType, DataType, DataType>;

// "Mixed" = input/output same type (FP16/BFP16), but scale/bias and mean/invvariance are FP32.
template <typename DataType>
using LayernormMixedFwdTestSuite = LayernormFwdTestSuite<DataType, DataType, float, float>;

// "Upcast" = input is FP16/BFP16 but output widens to FP32 (scale/bias and
// mean/invvariance are FP32).
template <typename DataType>
using LayernormUpcastFwdTestSuite = LayernormFwdTestSuite<DataType, float, float, float>;

} // namespace gpu_layernorm_fwd_ref_test
