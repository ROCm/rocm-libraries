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

namespace gpu_layernorm_bwd_ref_test
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
void runGpuVsCpuLayernormBwd(const std::vector<int64_t>& ioDims,
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

    auto dyTensor = Tensor<YDataType>(ioDims, layout);
    dyTensor.fillWithRandomValues(
        static_cast<YDataType>(-fillRange), static_cast<YDataType>(fillRange), seed);
    auto xTensor = Tensor<XDataType>(ioDims, layout);
    xTensor.fillWithRandomValues(
        static_cast<XDataType>(-fillRange), static_cast<XDataType>(fillRange), seed + 1);
    auto scaleTensor = Tensor<ScaleBiasDataType>(scaleBiasDims, layout);
    scaleTensor.fillWithRandomValues(static_cast<ScaleBiasDataType>(-fillRange),
                                     static_cast<ScaleBiasDataType>(fillRange),
                                     seed + 2);
    auto dxCpu = Tensor<XDataType>(ioDims, layout);
    auto dxGpu = Tensor<XDataType>(ioDims, layout);
    auto dscaleCpu = Tensor<ScaleBiasDataType>(scaleBiasDims, layout);
    auto dscaleGpu = Tensor<ScaleBiasDataType>(scaleBiasDims, layout);
    auto dbiasCpu = Tensor<ScaleBiasDataType>(scaleBiasDims, layout);
    auto dbiasGpu = Tensor<ScaleBiasDataType>(scaleBiasDims, layout);
    auto meanTensor = optionalTensors ? Tensor<MeanRstdDataType>(meanRstdDims, layout)
                                      : Tensor<MeanRstdDataType>({});
    auto rstdTensor = optionalTensors ? Tensor<MeanRstdDataType>(meanRstdDims, layout)
                                      : Tensor<MeanRstdDataType>({});
    if(optionalTensors)
    {
        meanTensor.fillWithRandomValues(static_cast<MeanRstdDataType>(-fillRange),
                                        static_cast<MeanRstdDataType>(fillRange),
                                        seed + 3);
        rstdTensor.fillWithRandomValues(static_cast<MeanRstdDataType>(-fillRange),
                                        static_cast<MeanRstdDataType>(fillRange),
                                        seed + 4);
    }

    GpuFpReferenceLayernorm::
        bprop<XDataType, ScaleBiasDataType, YDataType, MeanRstdDataType, ComputeDataType>(
            dyTensor,
            xTensor,
            scaleTensor,
            dxGpu,
            dscaleGpu,
            dbiasGpu,
            epsilon,
            optionalTensors ? &meanTensor : nullptr,
            optionalTensors ? &rstdTensor : nullptr,
            normalizedDimCount);

    CpuFpReferenceLayernorm::
        bprop<XDataType, ScaleBiasDataType, YDataType, MeanRstdDataType, ComputeDataType>(
            dyTensor,
            xTensor,
            scaleTensor,
            dxCpu,
            dscaleCpu,
            dbiasCpu,
            epsilon,
            optionalTensors ? &meanTensor : nullptr,
            optionalTensors ? &rstdTensor : nullptr,
            normalizedDimCount);

    assertAllClose(dxCpu, dxGpu, tolerance, "dx");
    assertAllClose(dscaleCpu, dscaleGpu, tolerance, "dscale");
    assertAllClose(dbiasCpu, dbiasGpu, tolerance, "dbias");
}

// ===============================================================================
// LayernormBwdShapeSuite — parameterized fixture for shape-based GPU-vs-CPU tests
// ===============================================================================

template <typename InputDataType,
          typename OutputDataType,
          typename ScaleBiasDataType,
          typename MeanInvVarianceDataType>
class LayernormBwdTestSuite
    : public ::testing::TestWithParam<gpu_layernorm_ref_test::LayernormTestCase>
{
protected:
    void runLayernormBwdTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& testCase = GetParam();
        runGpuVsCpuLayernormBwd<InputDataType,
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
using LayernormPureBwdTestSuite = LayernormBwdTestSuite<DataType, DataType, DataType, DataType>;

// "Mixed" = input/output same type (FP16/BFP16), but scale/bias and mean/invvariance are FP32.
template <typename DataType>
using LayernormMixedBwdTestSuite = LayernormBwdTestSuite<DataType, DataType, float, float>;

// "Upcast" = input is FP16/BFP16 but output widens to FP32 (scale/bias and
// mean/invvariance are FP32).
template <typename DataType>
using LayernormUpcastBwdTestSuite = LayernormBwdTestSuite<DataType, float, float, float>;

} // namespace gpu_layernorm_bwd_ref_test
