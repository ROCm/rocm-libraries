// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "GpuBatchnormFwdInfRefTestFixture.hpp"
#include <cstdint>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
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

TEST(TestGpuBatchnormFwdInfVarRefValidation, ThrowsOnInputRankTooSmall)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({4, 8});
    Tensor<float> y({4, 8});
    Tensor<float> scale({1, 8});
    Tensor<float> bias({1, 8});
    Tensor<float> estMean({1, 8});
    Tensor<float> invVar({1, 8});

    EXPECT_THROW(
        GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, invVar, y),
        std::invalid_argument);
}

TEST(TestGpuBatchnormFwdInfVarRefValidation, ThrowsOnInputRankTooLarge)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({4, 8, 2, 2, 2, 2});
    Tensor<float> y({4, 8, 2, 2, 2, 2});
    Tensor<float> scale({1, 8, 1, 1, 1, 1});
    Tensor<float> bias({1, 8, 1, 1, 1, 1});
    Tensor<float> estMean({1, 8, 1, 1, 1, 1});
    Tensor<float> invVar({1, 8, 1, 1, 1, 1});

    EXPECT_THROW(
        GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, invVar, y),
        std::invalid_argument);
}

TEST(TestGpuBatchnormFwdInfVarRefValidation, ThrowsOnOutputRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2});
    Tensor<float> y({4, 8, 2});
    Tensor<float> scale({1, 8, 1, 1});
    Tensor<float> bias({1, 8, 1, 1});
    Tensor<float> estMean({1, 8, 1, 1});
    Tensor<float> invVar({1, 8, 1, 1});

    EXPECT_THROW(
        GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, invVar, y),
        std::invalid_argument);
}

TEST(TestGpuBatchnormFwdInfVarRefValidation, ThrowsOnAffineRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2});
    Tensor<float> y({4, 8, 2, 2});
    Tensor<float> scale({1, 8, 2});
    Tensor<float> bias({1, 8, 1, 1});
    Tensor<float> estMean({1, 8, 1, 1});
    Tensor<float> invVar({1, 8, 1, 1});

    EXPECT_THROW(
        GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, invVar, y),
        std::invalid_argument);
}

TEST(TestGpuBatchnormFwdInfVarRefValidation, ThrowsOnAffineNotChannelOnly)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2});
    Tensor<float> y({4, 8, 2, 2});
    Tensor<float> scale({1, 8, 1, 2});
    Tensor<float> bias({1, 8, 1, 1});
    Tensor<float> estMean({1, 8, 1, 1});
    Tensor<float> invVar({1, 8, 1, 1});

    EXPECT_THROW(
        GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, invVar, y),
        std::invalid_argument);
}

TEST(TestGpuBatchnormFwdInfVarRefValidation, ThrowsOnAffineWrongChannel)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2});
    Tensor<float> y({4, 8, 2, 2});
    Tensor<float> scale({1, 8, 1, 1});
    Tensor<float> bias({1, 8, 1, 1});
    Tensor<float> estMean({1, 4, 1, 1});
    Tensor<float> invVar({1, 8, 1, 1});

    EXPECT_THROW(
        GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, invVar, y),
        std::invalid_argument);
}

TEST(TestGpuBatchnormFwdInfVarRefValidation, AcceptsAffineBroadcast)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2});
    Tensor<float> y({4, 8, 2, 2});
    Tensor<float> scale({1, 8, 1, 1, 1});
    Tensor<float> bias({1, 8, 1, 1});
    Tensor<float> estMean({1, 8, 1});
    Tensor<float> invVar({1, 8});

    EXPECT_NO_THROW(
        GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, invVar, y));
}

TEST(TestGpuBatchnormFwdInfVarRefValidation, ThrowsOnInconsistentLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2}, TensorLayout::NHWC);
    Tensor<float> y({4, 8, 2, 2}, TensorLayout::NCHW);
    Tensor<float> scale({1, 8}, TensorLayout::NHWC);
    Tensor<float> bias({1, 8}, TensorLayout::NHWC);
    Tensor<float> estMean({1, 8}, TensorLayout::NHWC);
    Tensor<float> invVar({1, 8}, TensorLayout::NHWC);

    EXPECT_THROW(
        GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, invVar, y),
        std::invalid_argument);
}

TEST(TestGpuBatchnormFwdInfVarRefValidation, ThrowsOnInvalidLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2}, TensorLayout::BSHD);
    Tensor<float> y({4, 8, 2, 2}, TensorLayout::BSHD);
    Tensor<float> scale({1, 8}, TensorLayout::BSHD);
    Tensor<float> bias({1, 8}, TensorLayout::BSHD);
    Tensor<float> estMean({1, 8}, TensorLayout::BSHD);
    Tensor<float> invVar({1, 8}, TensorLayout::BSHD);

    EXPECT_THROW(
        GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, invVar, y),
        std::invalid_argument);
}

TEST(TestGpuBatchnormFwdInfVarRefValidation, ThrowsOnNonPackedIOLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 2, 1, 1}, {16, 4, 1, 1});
    Tensor<float> y({4, 2, 1, 1}, {16, 4, 1, 1});
    Tensor<float> scale({1, 2}, {2, 1});
    Tensor<float> bias({1, 2}, {2, 1});
    Tensor<float> estMean({1, 2}, {2, 1});
    Tensor<float> invVar({1, 2}, {2, 1});

    EXPECT_THROW(
        GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, invVar, y),
        std::invalid_argument);
}

TEST(TestGpuBatchnormFwdInfVarRefValidation, ThrowsOnNonPackedAffineLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 2, 1, 1});
    Tensor<float> y({4, 2, 1, 1});
    Tensor<float> scale({1, 2}, {4, 2});
    Tensor<float> bias({1, 2}, {4, 2});
    Tensor<float> estMean({1, 2}, {4, 2});
    Tensor<float> invVar({1, 2}, {4, 2});

    EXPECT_THROW(
        GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, invVar, y),
        std::invalid_argument);
}

TEST(TestGpuBatchnormFwdInfVarRefValidation, ThrowsOnZeroChannelDim)
{
    SKIP_IF_NO_DEVICES();

    // Use `ShallowTensor` since `Tensor` has 0 dimension checks on object construction
    std::array<float, 6> backing = {1.0f, 2.0f, 3.0f, 4, 5, 6};
    const std::vector<int64_t> ioDims = {1, 0, 2, 3};
    const std::vector<int64_t> ioStrides = {6, 6, 3, 1};
    ShallowTensor<float> x(backing.data(), ioDims, ioStrides);
    ShallowTensor<float> y(backing.data(), ioDims, ioStrides);

    const std::vector<int64_t> affineDims = {1, 1, 1, 1};
    const std::vector<int64_t> affineStrides = {1, 1, 1, 1};
    ShallowTensor<float> scale(backing.data(), affineDims, affineStrides);
    ShallowTensor<float> bias(backing.data(), affineDims, affineStrides);
    ShallowTensor<float> estMean(backing.data(), affineDims, affineStrides);
    ShallowTensor<float> invVar(backing.data(), affineDims, affineStrides);

    EXPECT_THROW(
        GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, invVar, y),
        std::invalid_argument);
}

TEST(TestGpuBatchnormFwdInfVarRefValidation, ThrowsOnZeroBatchDim)
{
    SKIP_IF_NO_DEVICES();

    // Use `ShallowTensor` since `Tensor` has 0 dimension checks on object construction
    std::array<float, 6> backing = {1.0f, 2.0f, 3.0f, 4, 5, 6};
    const std::vector<int64_t> ioDims = {0, 1, 2, 3};
    const std::vector<int64_t> ioStrides = {6, 6, 3, 1};
    ShallowTensor<float> x(backing.data(), ioDims, ioStrides);
    ShallowTensor<float> y(backing.data(), ioDims, ioStrides);

    const std::vector<int64_t> affineDims = {1, 1, 1, 1};
    const std::vector<int64_t> affineStrides = {1, 1, 1, 1};
    ShallowTensor<float> scale(backing.data(), affineDims, affineStrides);
    ShallowTensor<float> bias(backing.data(), affineDims, affineStrides);
    ShallowTensor<float> estMean(backing.data(), affineDims, affineStrides);
    ShallowTensor<float> invVar(backing.data(), affineDims, affineStrides);

    EXPECT_THROW(
        GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, invVar, y),
        std::invalid_argument);
}

TEST(TestGpuBatchnormFwdInfVarRefValidation, ThrowsOnZeroSpatialDim)
{
    SKIP_IF_NO_DEVICES();

    // Use `ShallowTensor` since `Tensor` has 0 dimension checks on object construction
    std::array<float, 6> backing = {1.0f, 2.0f, 3.0f, 4, 5, 6};
    const std::vector<int64_t> ioDims = {1, 1, 0, 3};
    const std::vector<int64_t> ioStrides = {3, 3, 3, 1};
    ShallowTensor<float> x(backing.data(), ioDims, ioStrides);
    ShallowTensor<float> y(backing.data(), ioDims, ioStrides);

    const std::vector<int64_t> affineDims = {1, 1, 1, 1};
    const std::vector<int64_t> affineStrides = {1, 1, 1, 1};
    ShallowTensor<float> scale(backing.data(), affineDims, affineStrides);
    ShallowTensor<float> bias(backing.data(), affineDims, affineStrides);
    ShallowTensor<float> estMean(backing.data(), affineDims, affineStrides);
    ShallowTensor<float> invVar(backing.data(), affineDims, affineStrides);

    EXPECT_THROW(
        GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, invVar, y),
        std::invalid_argument);
}

// --- Test 3D/4D/5D shapes ---

TEST(TestGpuBatchnormFwdInfVar3DShapes, Broadcast2D)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({3, 2, 4});
    Tensor<float> scale({1, 2});
    Tensor<float> bias({1, 2});
    Tensor<float> estMean({1, 2});
    Tensor<float> variance({1, 2});
    Tensor<float> yCpu({3, 2, 4});
    Tensor<float> yGpu({3, 2, 4});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(-fillRange, fillRange, seed++);
    scale.fillWithRandomValues(-fillRange, fillRange, seed++);
    bias.fillWithRandomValues(-fillRange, fillRange, seed++);
    estMean.fillWithRandomValues(-fillRange, fillRange, seed++);
    variance.fillWithRandomValues(0.1f, 1.f, seed++);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, variance, yCpu);
    GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, variance, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInferenceWithVariance<float>());
}

TEST(TestGpuBatchnormFwdInfVar3DShapes, MaterialEpsilon)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({3, 2, 4});
    Tensor<float> scale({1, 2, 1});
    Tensor<float> bias({1, 2, 1});
    Tensor<float> estMean({1, 2, 1});
    Tensor<float> variance({1, 2, 1});
    Tensor<float> yCpu({3, 2, 4});
    Tensor<float> yGpu({3, 2, 4});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(-fillRange, fillRange, seed++);
    scale.fillWithRandomValues(-fillRange, fillRange, seed++);
    bias.fillWithRandomValues(-fillRange, fillRange, seed++);
    estMean.fillWithRandomValues(-fillRange, fillRange, seed++);
    variance.fillWithRandomValues(0.1f, 1.f, seed++);

    const double epsilon = 0.1;
    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        x, scale, bias, estMean, variance, yCpu, epsilon);
    GpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        x, scale, bias, estMean, variance, yGpu, epsilon);

    assertAllClose(yCpu, yGpu, getToleranceInferenceWithVariance<float>());
}

TEST(TestGpuBatchnormFwdInfVar4DShapes, Broadcast2D)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({3, 2, 4, 4});
    Tensor<float> scale({1, 2});
    Tensor<float> bias({1, 2});
    Tensor<float> estMean({1, 2});
    Tensor<float> variance({1, 2});
    Tensor<float> yCpu({3, 2, 4, 4});
    Tensor<float> yGpu({3, 2, 4, 4});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(-fillRange, fillRange, seed++);
    scale.fillWithRandomValues(-fillRange, fillRange, seed++);
    bias.fillWithRandomValues(-fillRange, fillRange, seed++);
    estMean.fillWithRandomValues(-fillRange, fillRange, seed++);
    variance.fillWithRandomValues(0.1f, 1.0f, seed++);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, variance, yCpu);
    GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, variance, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInferenceWithVariance<float>());
}

TEST(TestGpuBatchnormFwdInfVar4DShapes, Broadcast3D)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({3, 2, 4, 4});
    Tensor<float> scale({1, 2, 1});
    Tensor<float> bias({1, 2, 1});
    Tensor<float> estMean({1, 2, 1});
    Tensor<float> variance({1, 2, 1});
    Tensor<float> yCpu({3, 2, 4, 4});
    Tensor<float> yGpu({3, 2, 4, 4});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(-fillRange, fillRange, seed++);
    scale.fillWithRandomValues(-fillRange, fillRange, seed++);
    bias.fillWithRandomValues(-fillRange, fillRange, seed++);
    estMean.fillWithRandomValues(-fillRange, fillRange, seed++);
    variance.fillWithRandomValues(0.1f, 1.0f, seed++);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, variance, yCpu);
    GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, variance, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInferenceWithVariance<float>());
}

TEST(TestGpuBatchnormFwdInfVar4DShapes, MaterialEpsilon)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({3, 2, 4, 4});
    Tensor<float> scale({1, 2, 1, 1});
    Tensor<float> bias({1, 2, 1, 1});
    Tensor<float> estMean({1, 2, 1, 1});
    Tensor<float> variance({1, 2, 1, 1});
    Tensor<float> yCpu({3, 2, 4, 4});
    Tensor<float> yGpu({3, 2, 4, 4});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(-fillRange, fillRange, seed++);
    scale.fillWithRandomValues(-fillRange, fillRange, seed++);
    bias.fillWithRandomValues(-fillRange, fillRange, seed++);
    estMean.fillWithRandomValues(-fillRange, fillRange, seed++);
    variance.fillWithRandomValues(0.1f, 1.0f, seed++);

    const double epsilon = 0.1;
    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        x, scale, bias, estMean, variance, yCpu, epsilon);
    GpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        x, scale, bias, estMean, variance, yGpu, epsilon);

    assertAllClose(yCpu, yGpu, getToleranceInferenceWithVariance<float>());
}

TEST(TestGpuBatchnormFwdInfVar5DShapes, Broadcast2D)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({3, 2, 4, 4, 2});
    Tensor<float> scale({1, 2});
    Tensor<float> bias({1, 2});
    Tensor<float> estMean({1, 2});
    Tensor<float> variance({1, 2});
    Tensor<float> yCpu({3, 2, 4, 4, 2});
    Tensor<float> yGpu({3, 2, 4, 4, 2});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(-fillRange, fillRange, seed++);
    scale.fillWithRandomValues(-fillRange, fillRange, seed++);
    bias.fillWithRandomValues(-fillRange, fillRange, seed++);
    estMean.fillWithRandomValues(-fillRange, fillRange, seed++);
    variance.fillWithRandomValues(0.1f, 1.0f, seed++);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, variance, yCpu);
    GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, variance, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInferenceWithVariance<float>());
}

TEST(TestGpuBatchnormFwdInfVar5DShapes, Broadcast3D)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({3, 2, 4, 4, 2});
    Tensor<float> scale({1, 2, 1});
    Tensor<float> bias({1, 2, 1});
    Tensor<float> estMean({1, 2, 1});
    Tensor<float> variance({1, 2, 1});
    Tensor<float> yCpu({3, 2, 4, 4, 2});
    Tensor<float> yGpu({3, 2, 4, 4, 2});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(-fillRange, fillRange, seed++);
    scale.fillWithRandomValues(-fillRange, fillRange, seed++);
    bias.fillWithRandomValues(-fillRange, fillRange, seed++);
    estMean.fillWithRandomValues(-fillRange, fillRange, seed++);
    variance.fillWithRandomValues(0.1f, 1.0f, seed++);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, variance, yCpu);
    GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, variance, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInferenceWithVariance<float>());
}

TEST(TestGpuBatchnormFwdInfVar5DShapes, Broadcast4D)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({3, 2, 4, 4, 2});
    Tensor<float> scale({1, 2, 1, 1});
    Tensor<float> bias({1, 2, 1, 1});
    Tensor<float> estMean({1, 2, 1, 1});
    Tensor<float> variance({1, 2, 1, 1});
    Tensor<float> yCpu({3, 2, 4, 4, 2});
    Tensor<float> yGpu({3, 2, 4, 4, 2});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(-fillRange, fillRange, seed++);
    scale.fillWithRandomValues(-fillRange, fillRange, seed++);
    bias.fillWithRandomValues(-fillRange, fillRange, seed++);
    estMean.fillWithRandomValues(-fillRange, fillRange, seed++);
    variance.fillWithRandomValues(0.1f, 1.0f, seed++);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, variance, yCpu);
    GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, variance, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInferenceWithVariance<float>());
}

TEST(TestGpuBatchnormFwdInfVar5DShapes, MaterialEpsilon)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({3, 2, 4, 4, 2});
    Tensor<float> scale({1, 2, 1, 1, 1});
    Tensor<float> bias({1, 2, 1, 1, 1});
    Tensor<float> estMean({1, 2, 1, 1, 1});
    Tensor<float> variance({1, 2, 1, 1, 1});
    Tensor<float> yCpu({3, 2, 4, 4, 2});
    Tensor<float> yGpu({3, 2, 4, 4, 2});

    unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    x.fillWithRandomValues(-fillRange, fillRange, seed++);
    scale.fillWithRandomValues(-fillRange, fillRange, seed++);
    bias.fillWithRandomValues(-fillRange, fillRange, seed++);
    estMean.fillWithRandomValues(-fillRange, fillRange, seed++);
    variance.fillWithRandomValues(0.1f, 1.0f, seed++);

    const double epsilon = 0.1;
    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        x, scale, bias, estMean, variance, yCpu, epsilon);
    GpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        x, scale, bias, estMean, variance, yGpu, epsilon);

    assertAllClose(yCpu, yGpu, getToleranceInferenceWithVariance<float>());
}

// Edge case tests with DISABLED_ prefix to avoid running in CI.
// Run the tests manually with --gtest_also_run_disabled_tests
// --gtest_filter=*ExceedsUInt32MaxElements* flags.
TEST(TestGpuBatchnormFwdInfVar5DShapes, DISABLED_ExceedsUInt32MaxElements)
{
    SKIP_IF_NO_DEVICES();
    // Test with 4,974,412,500 elements, which is greater than 4,294,967,295 UINT32_MAX
    Tensor<half> x({255, 255, 255, 50, 6});
    Tensor<half> scale({1, 255, 1, 1, 1});
    Tensor<half> bias({1, 255, 1, 1, 1});
    Tensor<half> estMean({1, 255, 1, 1, 1});
    Tensor<half> variance({1, 255, 1, 1, 1});
    Tensor<half> yCpu({255, 255, 255, 50, 6});
    Tensor<half> yGpu({255, 255, 255, 50, 6});

    unsigned int seed = getGlobalTestSeed();
    const half fillRange(1.0);
    x.fillWithRandomValues(-fillRange, fillRange, seed++);
    scale.fillWithRandomValues(-fillRange, fillRange, seed++);
    bias.fillWithRandomValues(-fillRange, fillRange, seed++);
    estMean.fillWithRandomValues(-fillRange, fillRange, seed++);
    variance.fillWithRandomValues(half(0.1f), half(1.0f), seed++);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, variance, yCpu);
    GpuFpReferenceBatchnorm::fwdInferenceWithVariance(x, scale, bias, estMean, variance, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInferenceWithVariance<half>());
}

// --- Test mixed precision ---

TEST(TestGpuBatchnormFwdInfVarMixedPrecision, UpcastX)
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
    Tensor<MeanVarType> variance({1, 2, 1, 1});
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
    variance.fillWithRandomValues(
        static_cast<MeanVarType>(0.1f), static_cast<MeanVarType>(1.0f), seed++);

    CpuFpReferenceBatchnorm::
        fwdInferenceWithVariance<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, variance, yCpu);
    GpuFpReferenceBatchnorm::
        fwdInferenceWithVariance<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, variance, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInferenceWithVariance<YDataType>());
}

TEST(TestGpuBatchnormFwdInfVarMixedPrecision, DowncastX)
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
    Tensor<MeanVarType> variance({1, 2, 1, 1});
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
    variance.fillWithRandomValues(
        static_cast<MeanVarType>(0.1f), static_cast<MeanVarType>(1.0f), seed++);

    CpuFpReferenceBatchnorm::
        fwdInferenceWithVariance<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, variance, yCpu);
    GpuFpReferenceBatchnorm::
        fwdInferenceWithVariance<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, variance, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInferenceWithVariance<YDataType>());
}

TEST(TestGpuBatchnormFwdInfVarMixedPrecision, UpcastY)
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
    Tensor<MeanVarType> variance({1, 2, 1, 1});
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
    variance.fillWithRandomValues(
        static_cast<MeanVarType>(0.1f), static_cast<MeanVarType>(1.0f), seed++);

    CpuFpReferenceBatchnorm::
        fwdInferenceWithVariance<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, variance, yCpu);
    GpuFpReferenceBatchnorm::
        fwdInferenceWithVariance<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, variance, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInferenceWithVariance<YDataType>());
}

TEST(TestGpuBatchnormFwdInfVarMixedPrecision, DowncastY)
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
    Tensor<MeanVarType> variance({1, 2, 1, 1});
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
    variance.fillWithRandomValues(
        static_cast<MeanVarType>(0.1f), static_cast<MeanVarType>(1.0f), seed++);

    CpuFpReferenceBatchnorm::
        fwdInferenceWithVariance<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, variance, yCpu);
    GpuFpReferenceBatchnorm::
        fwdInferenceWithVariance<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, variance, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInferenceWithVariance<YDataType>());
}

TEST(TestGpuBatchnormFwdInfVarMixedPrecision, UpcastAffine)
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
    Tensor<MeanVarType> variance({1, 2, 1, 1});
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
    variance.fillWithRandomValues(
        static_cast<MeanVarType>(0.1f), static_cast<MeanVarType>(1.0f), seed++);

    CpuFpReferenceBatchnorm::
        fwdInferenceWithVariance<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, variance, yCpu);
    GpuFpReferenceBatchnorm::
        fwdInferenceWithVariance<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, variance, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInferenceWithVariance<YDataType>());
}

TEST(TestGpuBatchnormFwdInfVarMixedPrecision, DowncastAffine)
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
    Tensor<MeanVarType> variance({1, 2, 1, 1});
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
    variance.fillWithRandomValues(
        static_cast<MeanVarType>(0.1f), static_cast<MeanVarType>(1.0f), seed++);

    CpuFpReferenceBatchnorm::
        fwdInferenceWithVariance<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, variance, yCpu);
    GpuFpReferenceBatchnorm::
        fwdInferenceWithVariance<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, variance, yGpu);

    assertAllClose(yCpu, yGpu, getToleranceInferenceWithVariance<YDataType>());
}

TEST(TestGpuBatchnormFwdInfVarMixedPrecision, DowncastComputeHalf)
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
    Tensor<MeanVarType> variance({1, 2, 1, 1});
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
    variance.fillWithRandomValues(
        static_cast<MeanVarType>(0.1f), static_cast<MeanVarType>(1.0f), seed++);

    CpuFpReferenceBatchnorm::
        fwdInferenceWithVariance<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, variance, yCpu);

    GpuFpReferenceBatchnorm::
        fwdInferenceWithVariance<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, variance, yGpu);

    // Use compute type tolerance since half operations may not have same implementation between host CPU
    // reference and device GPU reference, e.g scale * inhat + bias may get contracted into a more precise
    // fma on device but not host
    assertAllClose(yCpu, yGpu, getToleranceInferenceWithVariance<ComputeDataType>());
}

TEST(TestGpuBatchnormFwdInfVarMixedPrecision, DowncastComputeBfloat)
{
    SKIP_IF_NO_DEVICES();

    using XDataType = float;
    using ScaleBiasType = float;
    using MeanVarType = float;
    using YDataType = float;
    using ComputeDataType = bfloat16;

    Tensor<XDataType> x({1, 2, 2, 2});
    Tensor<ScaleBiasType> scale({1, 2, 1, 1});
    Tensor<ScaleBiasType> bias({1, 2, 1, 1});
    Tensor<MeanVarType> estMean({1, 2, 1, 1});
    Tensor<MeanVarType> variance({1, 2, 1, 1});
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
    variance.fillWithRandomValues(
        static_cast<MeanVarType>(0.1f), static_cast<MeanVarType>(1.0f), seed++);

    CpuFpReferenceBatchnorm::
        fwdInferenceWithVariance<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, variance, yCpu);

    GpuFpReferenceBatchnorm::
        fwdInferenceWithVariance<XDataType, ScaleBiasType, MeanVarType, YDataType, ComputeDataType>(
            x, scale, bias, estMean, variance, yGpu);

    // Use compute type tolerance since half operations may not have same implementation between host CPU
    // reference and device GPU reference, e.g scale * inhat + bias may get contracted into a more precise
    // fma on device but not host
    assertAllClose(yCpu, yGpu, getToleranceInferenceWithVariance<ComputeDataType>());
}

// --- Test suite instantiations ---

using TestGpuBatchnormFwdInfVarRef3DFp32 = BatchnormFwdInfVarTestSuite<float>;
using TestGpuBatchnormFwdInfVarRef3DFp16 = BatchnormFwdInfVarTestSuite<half>;
using TestGpuBatchnormFwdInfVarRef3DBfp16 = BatchnormFwdInfVarTestSuite<bfloat16>;
using TestGpuBatchnormFwdInfVarRef4DFp32 = BatchnormFwdInfVarTestSuite<float>;
using TestGpuBatchnormFwdInfVarRef4DFp16 = BatchnormFwdInfVarTestSuite<half>;
using TestGpuBatchnormFwdInfVarRef4DBfp16 = BatchnormFwdInfVarTestSuite<bfloat16>;
using TestGpuBatchnormFwdInfVarRef5DFp32 = BatchnormFwdInfVarTestSuite<float>;
using TestGpuBatchnormFwdInfVarRef5DFp16 = BatchnormFwdInfVarTestSuite<half>;
using TestGpuBatchnormFwdInfVarRef5DBfp16 = BatchnormFwdInfVarTestSuite<bfloat16>;

TEST_P(TestGpuBatchnormFwdInfVarRef3DFp32, MatchesCpuRef)
{
    this->runBatchnormFwdInfWithVarianceTest();
}
TEST_P(TestGpuBatchnormFwdInfVarRef3DFp16, MatchesCpuRef)
{
    this->runBatchnormFwdInfWithVarianceTest();
}
TEST_P(TestGpuBatchnormFwdInfVarRef3DBfp16, MatchesCpuRef)
{
    this->runBatchnormFwdInfWithVarianceTest();
}
TEST_P(TestGpuBatchnormFwdInfVarRef4DFp32, MatchesCpuRef)
{
    this->runBatchnormFwdInfWithVarianceTest();
}
TEST_P(TestGpuBatchnormFwdInfVarRef4DFp16, MatchesCpuRef)
{
    this->runBatchnormFwdInfWithVarianceTest();
}
TEST_P(TestGpuBatchnormFwdInfVarRef4DBfp16, MatchesCpuRef)
{
    this->runBatchnormFwdInfWithVarianceTest();
}
TEST_P(TestGpuBatchnormFwdInfVarRef5DFp32, MatchesCpuRef)
{
    this->runBatchnormFwdInfWithVarianceTest();
}
TEST_P(TestGpuBatchnormFwdInfVarRef5DFp16, MatchesCpuRef)
{
    this->runBatchnormFwdInfWithVarianceTest();
}
TEST_P(TestGpuBatchnormFwdInfVarRef5DBfp16, MatchesCpuRef)
{
    this->runBatchnormFwdInfWithVarianceTest();
}

// ============================================================================
// 3D (NCL/NLC) tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdInfVarRef3DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormSmall3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdInfVarRef3DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormSmall3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdInfVarRef3DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormSmall3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdInfVarRef3DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormMedium3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdInfVarRef3DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormMedium3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdInfVarRef3DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormMedium3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdInfVarRef3DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdInfVarRef3DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdInfVarRef3DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge3DTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdInfVarRef3DFp32,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     ::testing::ValuesIn(getBatchnormLargeStress3DTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdInfVarRef3DFp16,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     ::testing::ValuesIn(getBatchnormLargeStress3DTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdInfVarRef3DBfp16,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     ::testing::ValuesIn(getBatchnormLargeStress3DTestCases())));

// ============================================================================
// 4D (NCHW/NHWC) tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdInfVarRef4DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormSmall4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdInfVarRef4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormSmall4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdInfVarRef4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormSmall4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdInfVarRef4DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormMedium4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdInfVarRef4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormMedium4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdInfVarRef4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormMedium4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdInfVarRef4DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdInfVarRef4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdInfVarRef4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge4DTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdInfVarRef4DFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     ::testing::ValuesIn(getBatchnormLargeStress4DTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdInfVarRef4DFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     ::testing::ValuesIn(getBatchnormLargeStress4DTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdInfVarRef4DBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     ::testing::ValuesIn(getBatchnormLargeStress4DTestCases())));

// ============================================================================
// 5D (NCDHW/NDHWC) shape tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdInfVarRef5DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormSmall5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdInfVarRef5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormSmall5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdInfVarRef5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormSmall5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdInfVarRef5DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormMedium5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdInfVarRef5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormMedium5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdInfVarRef5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormMedium5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdInfVarRef5DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdInfVarRef5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdInfVarRef5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge5DTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdInfVarRef5DFp32,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     ::testing::ValuesIn(getBatchnormLargeStress5DTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdInfVarRef5DFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     ::testing::ValuesIn(getBatchnormLargeStress5DTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdInfVarRef5DBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     ::testing::ValuesIn(getBatchnormLargeStress5DTestCases())));
