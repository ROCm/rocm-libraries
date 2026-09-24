// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "BatchnormTestCase.hpp"
#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceBatchnorm.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

namespace
{

using namespace hipdnn_data_sdk::types;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_gpu_ref;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::batchnorm;
using namespace gpu_batchnorm_ref_test;

template <typename DataType>
void runGpuVsCpuBatchnormBackward(const std::vector<int64_t>& dims,
                                  const TensorLayout& layout,
                                  bool useSavedStats)
{
    constexpr double EPSILON = 1e-5;
    std::vector<int64_t> affineDims(dims.size(), 1);
    affineDims[1] = dims[1];

    Tensor<DataType> dy(dims, layout);
    Tensor<DataType> x(dims, layout);
    Tensor<float> scale(affineDims, layout);
    Tensor<DataType> dxCpu(dims, layout);
    Tensor<DataType> dxGpu(dims, layout);
    Tensor<float> dscaleCpu(affineDims, layout);
    Tensor<float> dscaleGpu(affineDims, layout);
    Tensor<float> dbiasCpu(affineDims, layout);
    Tensor<float> dbiasGpu(affineDims, layout);
    Tensor<float> mean(affineDims, layout);
    Tensor<float> invVariance(affineDims, layout);

    const auto seed = getGlobalTestSeed();
    dy.fillWithRandomValues(static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), seed);
    x.fillWithRandomValues(static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), seed + 1);
    scale.fillWithRandomValues(-1.0f, 1.0f, seed + 2);
    mean.fillWithRandomValues(-1.0f, 1.0f, seed + 3);
    invVariance.fillWithRandomValues(0.25f, 2.0f, seed + 4);

    CpuFpReferenceBatchnorm::backward<DataType, DataType, float, float, DataType, float>(
        dy,
        x,
        scale,
        dxCpu,
        dscaleCpu,
        dbiasCpu,
        useSavedStats ? &mean : nullptr,
        useSavedStats ? &invVariance : nullptr,
        EPSILON);
    GpuFpReferenceBatchnorm::backward<DataType, DataType, float, float, DataType, float>(
        dy,
        x,
        scale,
        dxGpu,
        dscaleGpu,
        dbiasGpu,
        useSavedStats ? &mean : nullptr,
        useSavedStats ? &invVariance : nullptr,
        EPSILON);

    assertAllClose(dxCpu, dxGpu, getToleranceBackward<DataType>());
    assertAllClose(dscaleCpu, dscaleGpu, getToleranceBackward<float>());
    assertAllClose(dbiasCpu, dbiasGpu, getToleranceBackward<float>());
}

TEST(TestGpuBatchnormBwdRefFp32, MatchesCpuWithSavedStats)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuBatchnormBackward<float>({2, 3, 4, 5}, TensorLayout::NCHW, true);
}

TEST(TestGpuBatchnormBwdRefFp32, MatchesCpuWithCalculatedStats)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuBatchnormBackward<float>({2, 3, 4, 5}, TensorLayout::NHWC, false);
}

TEST(TestGpuBatchnormBwdRefFp16, MatchesCpuWithSavedStats)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuBatchnormBackward<half>({2, 3, 4, 5}, TensorLayout::NHWC, true);
}

TEST(TestGpuBatchnormBwdRefFp16, MatchesCpuWithCalculatedStats)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuBatchnormBackward<half>({2, 3, 4, 5}, TensorLayout::NCHW, false);
}

TEST(TestGpuBatchnormBwdRefBfp16, MatchesCpuWithSavedStats)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuBatchnormBackward<bfloat16>({2, 3, 4, 5}, TensorLayout::NCHW, true);
}

TEST(TestGpuBatchnormBwdRefBfp16, MatchesCpuWithCalculatedStats)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuBatchnormBackward<bfloat16>({2, 3, 4, 5}, TensorLayout::NHWC, false);
}

TEST(TestGpuBatchnormBwdRef, ThreeDimensionalChannelLastMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuBatchnormBackward<float>({3, 2, 7}, TensorLayout::NLC, true);
}

TEST(TestGpuBatchnormBwdRef, FiveDimensionalChannelFirstMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuBatchnormBackward<float>({2, 3, 2, 3, 4}, TensorLayout::NCDHW, false);
}

TEST(TestGpuBatchnormBwdRefValidation, RejectsIncompleteSavedStats)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 3, 4});
    Tensor<float> x({2, 3, 4});
    Tensor<float> scale({1, 3, 1});
    Tensor<float> dx({2, 3, 4});
    Tensor<float> dscale({1, 3, 1});
    Tensor<float> dbias({1, 3, 1});
    Tensor<float> mean({1, 3, 1});

    EXPECT_THROW((GpuFpReferenceBatchnorm::backward<float, float, float, float, float, float>(
                     dy, x, scale, dx, dscale, dbias, &mean, nullptr)),
                 std::invalid_argument);
}

} // namespace
