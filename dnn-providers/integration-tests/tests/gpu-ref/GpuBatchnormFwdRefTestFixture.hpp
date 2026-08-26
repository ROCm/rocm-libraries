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

// Used to generate random values for input tensor in a normal distribution around a mean
// and standard deviation that can be used in the mean/invVar tensor values.
template <typename T>
struct NormalDistGenerator
{
    NormalDistGenerator(unsigned seed, float mean, float stddev)
        : _seed(seed)
        , _mean(mean)
        , _stddev(stddev)
    {
    }
    void operator()(T* data, size_t count) const
    {
        std::mt19937 generator(_seed);
        std::normal_distribution<float> distribution(_mean, _stddev);

        for(size_t i = 0; i < count; ++i)
        {
            data[i] = static_cast<T>(distribution(generator));
        }
    }

private:
    unsigned int _seed;
    float _mean;
    float _stddev;
};

template <typename InputDataType,
          typename OutputDataType = InputDataType,
          typename ScaleBiasDataType = InputDataType,
          typename MeanVarDataType = InputDataType,
          typename ComputeDataType = double>
void runGpuVsCpuBatchnormFwd(const std::vector<int64_t>& ioDims, const TensorLayout& layout)
{
    std::vector<int64_t> affineDims(ioDims.size(), 1);
    affineDims[1] = ioDims[1];

    auto inputTensor = Tensor<InputDataType>(ioDims, layout);
    auto scaleTensor = Tensor<ScaleBiasDataType>(affineDims, layout);
    auto biasTensor = Tensor<ScaleBiasDataType>(affineDims, layout);
    auto estimatedMeanTensor = Tensor<MeanVarDataType>(affineDims, layout);
    auto invVarTensor = Tensor<MeanVarDataType>(affineDims, layout);
    auto outputCpu = Tensor<OutputDataType>(ioDims, layout);
    auto outputGpu = Tensor<OutputDataType>(ioDims, layout);

    constexpr float MEAN = 1e-3f;
    constexpr float STDDEV = 1e-2f;
    constexpr float INV_VARIANCE = 1.0f / (STDDEV * STDDEV);

    unsigned int seed = getGlobalTestSeed();
    inputTensor.fillWithValues(NormalDistGenerator<InputDataType>(seed++, MEAN, STDDEV), true);

    const float scaleBiasRange = 1e-5f;
    scaleTensor.fillWithRandomValues(static_cast<ScaleBiasDataType>(-scaleBiasRange),
                                     static_cast<ScaleBiasDataType>(scaleBiasRange),
                                     seed++);
    biasTensor.fillWithRandomValues(static_cast<ScaleBiasDataType>(-scaleBiasRange),
                                    static_cast<ScaleBiasDataType>(scaleBiasRange),
                                    seed++);
    estimatedMeanTensor.fillWithValue(static_cast<MeanVarDataType>(MEAN));
    invVarTensor.fillWithValue(static_cast<MeanVarDataType>(INV_VARIANCE));

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
                                ComputeDataType>(bnTestCase.ioDims, layout);
    }
};

} // namespace gpu_batchnorm_fwd_ref_test
