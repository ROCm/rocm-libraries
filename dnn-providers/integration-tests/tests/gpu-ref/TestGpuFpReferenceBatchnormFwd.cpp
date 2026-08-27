// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "GpuBatchnormFwdRefTestFixture.hpp"
#include <cstdint>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <limits>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::batchnorm;
using namespace hipdnn_gpu_ref;
using namespace gpu_batchnorm_ref_test;
using namespace gpu_batchnorm_fwd_ref_test;

// --- Validation configurations ---

TEST(TestGpuBatchnormFwdRefValidation, ThrowsOnInputRankTooSmall)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({4, 8});
    Tensor<float> y({4, 8});
    Tensor<float> scale({1, 8});
    Tensor<float> bias({1, 8});
    Tensor<float> estMean({1, 8});
    Tensor<float> invVar({1, 8});

    EXPECT_THROW(GpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, y),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormFwdRefValidation, ThrowsOnInputRankTooLarge)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({4, 8, 2, 2, 2, 2});
    Tensor<float> y({4, 8, 2, 2, 2, 2});
    Tensor<float> scale({1, 8, 1, 1, 1, 1});
    Tensor<float> bias({1, 8, 1, 1, 1, 1});
    Tensor<float> estMean({1, 8, 1, 1, 1, 1});
    Tensor<float> invVar({1, 8, 1, 1, 1, 1});

    EXPECT_THROW(GpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, y),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormFwdRefValidation, ThrowsOnOutputRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2});
    Tensor<float> y({4, 8, 2});
    Tensor<float> scale({1, 8, 1, 1});
    Tensor<float> bias({1, 8, 1, 1});
    Tensor<float> estMean({1, 8, 1, 1});
    Tensor<float> invVar({1, 8, 1, 1});

    EXPECT_THROW(GpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, y),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormFwdRefValidation, ThrowsOnAffineRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2});
    Tensor<float> y({4, 8, 2, 2});
    Tensor<float> scale({1, 8, 2});
    Tensor<float> bias({1, 8, 1, 1});
    Tensor<float> estMean({1, 8, 1, 1});
    Tensor<float> invVar({1, 8, 1, 1});

    EXPECT_THROW(GpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, y),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormFwdRefValidation, ThrowsOnAffineNotChannelOnly)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2});
    Tensor<float> y({4, 8, 2, 2});
    Tensor<float> scale({1, 8, 1, 2});
    Tensor<float> bias({1, 8, 1, 1});
    Tensor<float> estMean({1, 8, 1, 1});
    Tensor<float> invVar({1, 8, 1, 1});

    EXPECT_THROW(GpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, y),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormFwdRefValidation, ThrowsOnAffineWrongChannel)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2});
    Tensor<float> y({4, 8, 2, 2});
    Tensor<float> scale({1, 8, 1, 1});
    Tensor<float> bias({1, 8, 1, 1});
    Tensor<float> estMean({1, 4, 1, 1});
    Tensor<float> invVar({1, 8, 1, 1});

    EXPECT_THROW(GpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, y),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormFwdRefValidation, AcceptsAffineBroadcast)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2});
    Tensor<float> y({4, 8, 2, 2});
    Tensor<float> scale({1, 8, 1, 1, 1});
    Tensor<float> bias({1, 8, 1, 1});
    Tensor<float> estMean({1, 8, 1});
    Tensor<float> invVar({1, 8});

    EXPECT_NO_THROW(GpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, y));
}

TEST(TestGpuBatchnormFwdRefValidation, ThrowsOnInconsistentLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2}, TensorLayout::NHWC);
    Tensor<float> y({4, 8, 2, 2}, TensorLayout::NCHW);
    Tensor<float> scale({1, 8}, TensorLayout::NHWC);
    Tensor<float> bias({1, 8}, TensorLayout::NHWC);
    Tensor<float> estMean({1, 8}, TensorLayout::NHWC);
    Tensor<float> invVar({1, 8}, TensorLayout::NHWC);

    EXPECT_THROW(GpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, y),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormFwdRefValidation, ThrowsOnInvalidLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2}, TensorLayout::BSHD);
    Tensor<float> y({4, 8, 2, 2}, TensorLayout::BSHD);
    Tensor<float> scale({1, 8}, TensorLayout::BSHD);
    Tensor<float> bias({1, 8}, TensorLayout::BSHD);
    Tensor<float> estMean({1, 8}, TensorLayout::BSHD);
    Tensor<float> invVar({1, 8}, TensorLayout::BSHD);

    EXPECT_THROW(GpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, y),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormFwdRefValidation, ThrowsOnNonPackedIOLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 2, 1, 1}, {16, 4, 1, 1});
    Tensor<float> y({4, 2, 1, 1}, {16, 4, 1, 1});
    Tensor<float> scale({1, 2}, {2, 1});
    Tensor<float> bias({1, 2}, {2, 1});
    Tensor<float> estMean({1, 2}, {2, 1});
    Tensor<float> invVar({1, 2}, {2, 1});

    EXPECT_THROW(GpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, y),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormFwdRefValidation, ThrowsOnNonPackedAffineLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 2, 1, 1});
    Tensor<float> y({4, 2, 1, 1});
    Tensor<float> scale({1, 2}, {4, 2});
    Tensor<float> bias({1, 2}, {4, 2});
    Tensor<float> estMean({1, 2}, {4, 2});
    Tensor<float> invVar({1, 2}, {4, 2});

    EXPECT_THROW(GpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, y),
                 std::invalid_argument);
}

// --- Test 3D/4D/5D shapes ---

TEST(TestGpuBatchnormFwd3DShapes, Broadcast2D)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({3, 2, 4});
    Tensor<float> scale({1, 2});
    Tensor<float> bias({1, 2});
    Tensor<float> estMean({1, 2});
    Tensor<float> invVar({1, 2});
    Tensor<float> yCpu({3, 2, 4});
    Tensor<float> yGpu({3, 2, 4});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(-fillRange, fillRange, seed++);
    scale.fillWithRandomValues(-fillRange, fillRange, seed++);
    bias.fillWithRandomValues(-fillRange, fillRange, seed++);
    estMean.fillWithRandomValues(-fillRange, fillRange, seed++);
    invVar.fillWithRandomValues(-fillRange, fillRange, seed++);

    CpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, yCpu);
    GpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInference<float>());
}

TEST(TestGpuBatchnormFwd4DShapes, Broadcast2D)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({3, 2, 4, 4});
    Tensor<float> scale({1, 2});
    Tensor<float> bias({1, 2});
    Tensor<float> estMean({1, 2});
    Tensor<float> invVar({1, 2});
    Tensor<float> yCpu({3, 2, 4, 4});
    Tensor<float> yGpu({3, 2, 4, 4});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(-fillRange, fillRange, seed++);
    scale.fillWithRandomValues(-fillRange, fillRange, seed++);
    bias.fillWithRandomValues(-fillRange, fillRange, seed++);
    estMean.fillWithRandomValues(-fillRange, fillRange, seed++);
    invVar.fillWithRandomValues(-fillRange, fillRange, seed++);

    CpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, yCpu);
    GpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInference<float>());
}

TEST(TestGpuBatchnormFwd4DShapes, Broadcast3D)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({3, 2, 4, 4});
    Tensor<float> scale({1, 2, 1});
    Tensor<float> bias({1, 2, 1});
    Tensor<float> estMean({1, 2, 1});
    Tensor<float> invVar({1, 2, 1});
    Tensor<float> yCpu({3, 2, 4, 4});
    Tensor<float> yGpu({3, 2, 4, 4});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(-fillRange, fillRange, seed++);
    scale.fillWithRandomValues(-fillRange, fillRange, seed++);
    bias.fillWithRandomValues(-fillRange, fillRange, seed++);
    estMean.fillWithRandomValues(-fillRange, fillRange, seed++);
    invVar.fillWithRandomValues(-fillRange, fillRange, seed++);

    CpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, yCpu);
    GpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInference<float>());
}

TEST(TestGpuBatchnormFwd5DShapes, Broadcast2D)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({3, 2, 4, 4, 2});
    Tensor<float> scale({1, 2});
    Tensor<float> bias({1, 2});
    Tensor<float> estMean({1, 2});
    Tensor<float> invVar({1, 2});
    Tensor<float> yCpu({3, 2, 4, 4, 2});
    Tensor<float> yGpu({3, 2, 4, 4, 2});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(-fillRange, fillRange, seed++);
    scale.fillWithRandomValues(-fillRange, fillRange, seed++);
    bias.fillWithRandomValues(-fillRange, fillRange, seed++);
    estMean.fillWithRandomValues(-fillRange, fillRange, seed++);
    invVar.fillWithRandomValues(-fillRange, fillRange, seed++);

    CpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, yCpu);
    GpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInference<float>());
}

TEST(TestGpuBatchnormFwd5DShapes, Broadcast3D)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({3, 2, 4, 4, 2});
    Tensor<float> scale({1, 2, 1});
    Tensor<float> bias({1, 2, 1});
    Tensor<float> estMean({1, 2, 1});
    Tensor<float> invVar({1, 2, 1});
    Tensor<float> yCpu({3, 2, 4, 4, 2});
    Tensor<float> yGpu({3, 2, 4, 4, 2});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(-fillRange, fillRange, seed++);
    scale.fillWithRandomValues(-fillRange, fillRange, seed++);
    bias.fillWithRandomValues(-fillRange, fillRange, seed++);
    estMean.fillWithRandomValues(-fillRange, fillRange, seed++);
    invVar.fillWithRandomValues(-fillRange, fillRange, seed++);

    CpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, yCpu);
    GpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInference<float>());
}

TEST(TestGpuBatchnormFwd5DShapes, Broadcast4D)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({3, 2, 4, 4, 2});
    Tensor<float> scale({1, 2, 1, 1});
    Tensor<float> bias({1, 2, 1, 1});
    Tensor<float> estMean({1, 2, 1, 1});
    Tensor<float> invVar({1, 2, 1, 1});
    Tensor<float> yCpu({3, 2, 4, 4, 2});
    Tensor<float> yGpu({3, 2, 4, 4, 2});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(-fillRange, fillRange, seed++);
    scale.fillWithRandomValues(-fillRange, fillRange, seed++);
    bias.fillWithRandomValues(-fillRange, fillRange, seed++);
    estMean.fillWithRandomValues(-fillRange, fillRange, seed++);
    invVar.fillWithRandomValues(-fillRange, fillRange, seed++);

    CpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, yCpu);
    GpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInference<float>());
}

// Edge case tests with DISABLED_ prefix to avoid running in CI.
// Run the tests manually with --gtest_also_run_disabled_tests
// --gtest_filter=*ExceedsUInt32MaxElements* flags.
TEST(TestGpuBatchnormFwd5DShapes, DISABLED_ExceedsUInt32MaxElements)
{
    SKIP_IF_NO_DEVICES();
    // Test with 4,974,412,500 elements, which is greater than 4,294,967,295 UINT32_MAX
    Tensor<half> x({255, 255, 255, 50, 6});
    Tensor<half> scale({1, 255, 1, 1, 1});
    Tensor<half> bias({1, 255, 1, 1, 1});
    Tensor<half> estMean({1, 255, 1, 1, 1});
    Tensor<half> invVar({1, 255, 1, 1, 1});
    Tensor<half> yCpu({255, 255, 255, 50, 6});
    Tensor<half> yGpu({255, 255, 255, 50, 6});

    unsigned int seed = getGlobalTestSeed();
    const half fillRange(1.0);
    x.fillWithRandomValues(-fillRange, fillRange, seed++);
    scale.fillWithRandomValues(-fillRange, fillRange, seed++);
    bias.fillWithRandomValues(-fillRange, fillRange, seed++);
    estMean.fillWithRandomValues(-fillRange, fillRange, seed++);
    invVar.fillWithRandomValues(-fillRange, fillRange, seed++);

    CpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, yCpu);
    GpuFpReferenceBatchnorm::fwdInference(x, scale, bias, estMean, invVar, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInference<half>());
}

// --- Test mixed precision ---

TEST(TestGpuBatchnormFwdMixedPrecision, UpcastX)
{
    SKIP_IF_NO_DEVICES();

    using XDataType = bfloat16;
    using ScaleBiasType = float;
    using MeanVarType = float;
    using YDataType = float;
    using ComputeDataType = float;

    Tensor<XDataType> x({1, 2, 2, 2});
    Tensor<ScaleBiasType> scale({1, 2, 1, 1});
    Tensor<ScaleBiasType> bias({1, 2, 1, 1});
    Tensor<MeanVarType> estMean({1, 2, 1, 1});
    Tensor<MeanVarType> invVar({1, 2, 1, 1});
    Tensor<YDataType> yCpu({1, 2, 2, 2});
    Tensor<YDataType> yGpu({1, 2, 2, 2});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(
        static_cast<XDataType>(-fillRange), static_cast<XDataType>(fillRange), seed++);
    scale.fillWithRandomValues(
        static_cast<ScaleBiasType>(-fillRange), static_cast<ScaleBiasType>(fillRange), seed++);
    bias.fillWithRandomValues(
        static_cast<ScaleBiasType>(-fillRange), static_cast<ScaleBiasType>(fillRange), seed++);
    estMean.fillWithRandomValues(
        static_cast<MeanVarType>(-fillRange), static_cast<MeanVarType>(fillRange), seed++);
    invVar.fillWithRandomValues(
        static_cast<MeanVarType>(-fillRange), static_cast<MeanVarType>(fillRange), seed++);

    CpuFpReferenceBatchnorm::
        fwdInference<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, invVar, yCpu);
    GpuFpReferenceBatchnorm::
        fwdInference<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, invVar, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInference<YDataType>());
}

TEST(TestGpuBatchnormFwdMixedPrecision, DowncastX)
{
    SKIP_IF_NO_DEVICES();

    using XDataType = float;
    using ScaleBiasType = half;
    using MeanVarType = half;
    using YDataType = half;
    using ComputeDataType = float;

    Tensor<XDataType> x({1, 2, 2, 2});
    Tensor<ScaleBiasType> scale({1, 2, 1, 1});
    Tensor<ScaleBiasType> bias({1, 2, 1, 1});
    Tensor<MeanVarType> estMean({1, 2, 1, 1});
    Tensor<MeanVarType> invVar({1, 2, 1, 1});
    Tensor<YDataType> yCpu({1, 2, 2, 2});
    Tensor<YDataType> yGpu({1, 2, 2, 2});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(
        static_cast<XDataType>(-fillRange), static_cast<XDataType>(fillRange), seed++);
    scale.fillWithRandomValues(
        static_cast<ScaleBiasType>(-fillRange), static_cast<ScaleBiasType>(fillRange), seed++);
    bias.fillWithRandomValues(
        static_cast<ScaleBiasType>(-fillRange), static_cast<ScaleBiasType>(fillRange), seed++);
    estMean.fillWithRandomValues(
        static_cast<MeanVarType>(-fillRange), static_cast<MeanVarType>(fillRange), seed++);
    invVar.fillWithRandomValues(
        static_cast<MeanVarType>(-fillRange), static_cast<MeanVarType>(fillRange), seed++);

    CpuFpReferenceBatchnorm::
        fwdInference<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, invVar, yCpu);
    GpuFpReferenceBatchnorm::
        fwdInference<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, invVar, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInference<YDataType>());
}

TEST(TestGpuBatchnormFwdMixedPrecision, UpcastY)
{
    SKIP_IF_NO_DEVICES();

    using XDataType = half;
    using ScaleBiasType = half;
    using MeanVarType = half;
    using YDataType = float;
    using ComputeDataType = float;

    Tensor<XDataType> x({1, 2, 2, 2});
    Tensor<ScaleBiasType> scale({1, 2, 1, 1});
    Tensor<ScaleBiasType> bias({1, 2, 1, 1});
    Tensor<MeanVarType> estMean({1, 2, 1, 1});
    Tensor<MeanVarType> invVar({1, 2, 1, 1});
    Tensor<YDataType> yCpu({1, 2, 2, 2});
    Tensor<YDataType> yGpu({1, 2, 2, 2});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(
        static_cast<XDataType>(-fillRange), static_cast<XDataType>(fillRange), seed++);
    scale.fillWithRandomValues(
        static_cast<ScaleBiasType>(-fillRange), static_cast<ScaleBiasType>(fillRange), seed++);
    bias.fillWithRandomValues(
        static_cast<ScaleBiasType>(-fillRange), static_cast<ScaleBiasType>(fillRange), seed++);
    estMean.fillWithRandomValues(
        static_cast<MeanVarType>(-fillRange), static_cast<MeanVarType>(fillRange), seed++);
    invVar.fillWithRandomValues(
        static_cast<MeanVarType>(-fillRange), static_cast<MeanVarType>(fillRange), seed++);

    CpuFpReferenceBatchnorm::
        fwdInference<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, invVar, yCpu);
    GpuFpReferenceBatchnorm::
        fwdInference<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, invVar, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInference<YDataType>());
}

TEST(TestGpuBatchnormFwdMixedPrecision, DowncastY)
{
    SKIP_IF_NO_DEVICES();

    using XDataType = float;
    using ScaleBiasType = float;
    using MeanVarType = float;
    using YDataType = bfloat16;
    using ComputeDataType = float;

    Tensor<XDataType> x({1, 2, 2, 2});
    Tensor<ScaleBiasType> scale({1, 2, 1, 1});
    Tensor<ScaleBiasType> bias({1, 2, 1, 1});
    Tensor<MeanVarType> estMean({1, 2, 1, 1});
    Tensor<MeanVarType> invVar({1, 2, 1, 1});
    Tensor<YDataType> yCpu({1, 2, 2, 2});
    Tensor<YDataType> yGpu({1, 2, 2, 2});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(
        static_cast<XDataType>(-fillRange), static_cast<XDataType>(fillRange), seed++);
    scale.fillWithRandomValues(
        static_cast<ScaleBiasType>(-fillRange), static_cast<ScaleBiasType>(fillRange), seed++);
    bias.fillWithRandomValues(
        static_cast<ScaleBiasType>(-fillRange), static_cast<ScaleBiasType>(fillRange), seed++);
    estMean.fillWithRandomValues(
        static_cast<MeanVarType>(-fillRange), static_cast<MeanVarType>(fillRange), seed++);
    invVar.fillWithRandomValues(
        static_cast<MeanVarType>(-fillRange), static_cast<MeanVarType>(fillRange), seed++);

    CpuFpReferenceBatchnorm::
        fwdInference<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, invVar, yCpu);
    GpuFpReferenceBatchnorm::
        fwdInference<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, invVar, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInference<YDataType>());
}

TEST(TestGpuBatchnormFwdMixedPrecision, UpcastAffine)
{
    SKIP_IF_NO_DEVICES();

    using XDataType = bfloat16;
    using ScaleBiasType = float;
    using MeanVarType = float;
    using YDataType = half;
    using ComputeDataType = float;

    Tensor<XDataType> x({1, 2, 2, 2});
    Tensor<ScaleBiasType> scale({1, 2, 1, 1});
    Tensor<ScaleBiasType> bias({1, 2, 1, 1});
    Tensor<MeanVarType> estMean({1, 2, 1, 1});
    Tensor<MeanVarType> invVar({1, 2, 1, 1});
    Tensor<YDataType> yCpu({1, 2, 2, 2});
    Tensor<YDataType> yGpu({1, 2, 2, 2});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(
        static_cast<XDataType>(-fillRange), static_cast<XDataType>(fillRange), seed++);
    scale.fillWithRandomValues(
        static_cast<ScaleBiasType>(-fillRange), static_cast<ScaleBiasType>(fillRange), seed++);
    bias.fillWithRandomValues(
        static_cast<ScaleBiasType>(-fillRange), static_cast<ScaleBiasType>(fillRange), seed++);
    estMean.fillWithRandomValues(
        static_cast<MeanVarType>(-fillRange), static_cast<MeanVarType>(fillRange), seed++);
    invVar.fillWithRandomValues(
        static_cast<MeanVarType>(-fillRange), static_cast<MeanVarType>(fillRange), seed++);

    CpuFpReferenceBatchnorm::
        fwdInference<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, invVar, yCpu);
    GpuFpReferenceBatchnorm::
        fwdInference<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, invVar, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInference<YDataType>());
}

TEST(TestGpuBatchnormFwdMixedPrecision, DowncastAffine)
{
    SKIP_IF_NO_DEVICES();

    using XDataType = float;
    using ScaleBiasType = half;
    using MeanVarType = bfloat16;
    using YDataType = float;
    using ComputeDataType = float;

    Tensor<XDataType> x({1, 2, 2, 2});
    Tensor<ScaleBiasType> scale({1, 2, 1, 1});
    Tensor<ScaleBiasType> bias({1, 2, 1, 1});
    Tensor<MeanVarType> estMean({1, 2, 1, 1});
    Tensor<MeanVarType> invVar({1, 2, 1, 1});
    Tensor<YDataType> yCpu({1, 2, 2, 2});
    Tensor<YDataType> yGpu({1, 2, 2, 2});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(
        static_cast<XDataType>(-fillRange), static_cast<XDataType>(fillRange), seed++);
    scale.fillWithRandomValues(
        static_cast<ScaleBiasType>(-fillRange), static_cast<ScaleBiasType>(fillRange), seed++);
    bias.fillWithRandomValues(
        static_cast<ScaleBiasType>(-fillRange), static_cast<ScaleBiasType>(fillRange), seed++);
    estMean.fillWithRandomValues(
        static_cast<MeanVarType>(-fillRange), static_cast<MeanVarType>(fillRange), seed++);
    invVar.fillWithRandomValues(
        static_cast<MeanVarType>(-fillRange), static_cast<MeanVarType>(fillRange), seed++);

    CpuFpReferenceBatchnorm::
        fwdInference<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, invVar, yCpu);
    GpuFpReferenceBatchnorm::
        fwdInference<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, invVar, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInference<YDataType>());
}

TEST(TestGpuBatchnormFwdMixedPrecision, DowncastCompute)
{
    SKIP_IF_NO_DEVICES();

    using XDataType = float;
    using ScaleBiasType = float;
    using MeanVarType = float;
    using YDataType = float;
    using ComputeDataType = half;

    Tensor<XDataType> x({1, 2, 2, 2});
    Tensor<ScaleBiasType> scale({1, 2, 1, 1});
    Tensor<ScaleBiasType> bias({1, 2, 1, 1});
    Tensor<MeanVarType> estMean({1, 2, 1, 1});
    Tensor<MeanVarType> invVar({1, 2, 1, 1});
    Tensor<YDataType> yCpu({1, 2, 2, 2});
    Tensor<YDataType> yGpu({1, 2, 2, 2});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(
        static_cast<XDataType>(-fillRange), static_cast<XDataType>(fillRange), seed++);
    scale.fillWithRandomValues(
        static_cast<ScaleBiasType>(-fillRange), static_cast<ScaleBiasType>(fillRange), seed++);
    bias.fillWithRandomValues(
        static_cast<ScaleBiasType>(-fillRange), static_cast<ScaleBiasType>(fillRange), seed++);
    estMean.fillWithRandomValues(
        static_cast<MeanVarType>(-fillRange), static_cast<MeanVarType>(fillRange), seed++);
    invVar.fillWithRandomValues(
        static_cast<MeanVarType>(-fillRange), static_cast<MeanVarType>(fillRange), seed++);

    CpuFpReferenceBatchnorm::
        fwdInference<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, invVar, yCpu);
    GpuFpReferenceBatchnorm::
        fwdInference<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, invVar, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInference<YDataType>());
}

// --- Test suite instantiations ---

using TestGpuBatchnormFwdRef3DFp32 = BatchnormFwdTestSuite<float>;
using TestGpuBatchnormFwdRef3DFp16 = BatchnormFwdTestSuite<half>;
using TestGpuBatchnormFwdRef3DBfp16 = BatchnormFwdTestSuite<bfloat16>;
using TestGpuBatchnormFwdRef4DFp32 = BatchnormFwdTestSuite<float>;
using TestGpuBatchnormFwdRef4DFp16 = BatchnormFwdTestSuite<half>;
using TestGpuBatchnormFwdRef4DBfp16 = BatchnormFwdTestSuite<bfloat16>;
using TestGpuBatchnormFwdRef5DFp32 = BatchnormFwdTestSuite<float>;
using TestGpuBatchnormFwdRef5DFp16 = BatchnormFwdTestSuite<half>;
using TestGpuBatchnormFwdRef5DBfp16 = BatchnormFwdTestSuite<bfloat16>;

TEST_P(TestGpuBatchnormFwdRef3DFp32, MatchesCpuRef)
{
    this->runBatchnormFwdTest();
}
TEST_P(TestGpuBatchnormFwdRef3DFp16, MatchesCpuRef)
{
    this->runBatchnormFwdTest();
}
TEST_P(TestGpuBatchnormFwdRef3DBfp16, MatchesCpuRef)
{
    this->runBatchnormFwdTest();
}
TEST_P(TestGpuBatchnormFwdRef4DFp32, MatchesCpuRef)
{
    this->runBatchnormFwdTest();
}
TEST_P(TestGpuBatchnormFwdRef4DFp16, MatchesCpuRef)
{
    this->runBatchnormFwdTest();
}
TEST_P(TestGpuBatchnormFwdRef4DBfp16, MatchesCpuRef)
{
    this->runBatchnormFwdTest();
}
TEST_P(TestGpuBatchnormFwdRef5DFp32, MatchesCpuRef)
{
    this->runBatchnormFwdTest();
}
TEST_P(TestGpuBatchnormFwdRef5DFp16, MatchesCpuRef)
{
    this->runBatchnormFwdTest();
}
TEST_P(TestGpuBatchnormFwdRef5DBfp16, MatchesCpuRef)
{
    this->runBatchnormFwdTest();
}

// ============================================================================
// 3D (NCH/NHC) tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdRef3DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormSmall3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdRef3DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormSmall3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdRef3DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormSmall3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdRef3DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormMedium3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdRef3DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormMedium3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdRef3DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormMedium3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdRef3DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdRef3DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdRef3DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge3DTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdRef3DFp32,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     ::testing::ValuesIn(getBatchnormLargeStress3DTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdRef3DFp16,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     ::testing::ValuesIn(getBatchnormLargeStress3DTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdRef3DBfp16,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     ::testing::ValuesIn(getBatchnormLargeStress3DTestCases())));

// ============================================================================
// 4D (NCHW/NHWC) tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdRef4DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormSmall4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdRef4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormSmall4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdRef4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormSmall4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdRef4DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormMedium4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdRef4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormMedium4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdRef4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormMedium4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdRef4DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdRef4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdRef4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge4DTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdRef4DFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     ::testing::ValuesIn(getBatchnormLargeStress4DTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdRef4DFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     ::testing::ValuesIn(getBatchnormLargeStress4DTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdRef4DBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     ::testing::ValuesIn(getBatchnormLargeStress4DTestCases())));

// ============================================================================
// 5D (NCDHW/NDHWC) shape tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdRef5DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormSmall5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdRef5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormSmall5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdRef5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormSmall5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdRef5DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormMedium5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdRef5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormMedium5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdRef5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormMedium5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdRef5DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdRef5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdRef5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge5DTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdRef5DFp32,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     ::testing::ValuesIn(getBatchnormLargeStress5DTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdRef5DFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     ::testing::ValuesIn(getBatchnormLargeStress5DTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdRef5DBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     ::testing::ValuesIn(getBatchnormLargeStress5DTestCases())));
