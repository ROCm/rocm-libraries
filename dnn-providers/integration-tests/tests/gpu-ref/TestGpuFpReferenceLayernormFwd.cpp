// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GpuLayernormFwdRefTestFixture.hpp"
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <stdexcept>

// --- Valid configurations ---

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::layernorm;
using namespace hipdnn_gpu_ref;
using namespace gpu_layernorm_ref_test;
using namespace gpu_layernorm_fwd_ref_test;

TEST(TestGpuLayernormFwdRefValidation, AcceptsValidParams4D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> bias({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8});

    EXPECT_NO_THROW(
        GpuFpReferenceLayernorm::fprop<float>(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 3));
}

TEST(TestGpuLayernormFwdRefValidation, AcceptsValidParams5D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8, 8});
    Tensor<float> scale({1, 4, 8, 8, 8});
    Tensor<float> bias({1, 4, 8, 8, 8});
    Tensor<float> y({2, 4, 8, 8, 8});

    EXPECT_NO_THROW(
        GpuFpReferenceLayernorm::fprop<float>(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 4));
}

TEST(TestGpuLayernormFwdRefValidation, AcceptsValidParamsChannelLastLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> scale({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> bias({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> y({2, 4, 8, 8}, TensorLayout::NHWC);

    EXPECT_NO_THROW(
        GpuFpReferenceLayernorm::fprop<float>(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 3));
}

TEST(TestGpuLayernormFwdRefValidation, AcceptsValidParamsWithMeanAndRstd)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> bias({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8});
    Tensor<double> mean({2, 1, 1, 1});
    Tensor<double> rstd({2, 1, 1, 1});

    EXPECT_NO_THROW(GpuFpReferenceLayernorm::fprop<float>(
        x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 3, &mean, &rstd));
}

TEST(TestGpuLayernormFwdRefValidation, AcceptsValidParams4NormalizeDimTwoD)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 1, 8, 8});
    Tensor<float> bias({1, 1, 8, 8});
    Tensor<float> y({2, 4, 8, 8});
    Tensor<double> mean({2, 4, 1, 1});
    Tensor<double> rstd({2, 4, 1, 1});

    EXPECT_NO_THROW(GpuFpReferenceLayernorm::fprop<float>(
        x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 2, &mean, &rstd));
}

TEST(TestGpuLayernormFwdRefValidation, AcceptsValidParamsNormalizeDimThree4D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 1, 1, 8});
    Tensor<float> bias({1, 1, 1, 8});
    Tensor<float> y({2, 4, 8, 8});
    Tensor<double> mean({2, 4, 8, 1});
    Tensor<double> rstd({2, 4, 8, 1});

    EXPECT_NO_THROW(GpuFpReferenceLayernorm::fprop<float>(
        x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 1, &mean, &rstd));
}

TEST(TestGpuLayernormFwdRefValidation, AcceptsValidParamsNormalizeDimThree5D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8, 8});
    Tensor<float> scale({1, 1, 1, 8, 8});
    Tensor<float> bias({1, 1, 1, 8, 8});
    Tensor<float> y({2, 4, 8, 8, 8});
    Tensor<double> mean({2, 4, 8, 1, 1});
    Tensor<double> rstd({2, 4, 8, 1, 1});

    EXPECT_NO_THROW(GpuFpReferenceLayernorm::fprop<float>(
        x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 2, &mean, &rstd));
}

TEST(TestGpuLayernormFwdRefValidation, AcceptsValidParamsNormalizeDimFour5D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8, 8});
    Tensor<float> scale({1, 1, 1, 1, 8});
    Tensor<float> bias({1, 1, 1, 1, 8});
    Tensor<float> y({2, 4, 8, 8, 8});
    Tensor<double> mean({2, 4, 8, 8, 1});
    Tensor<double> rstd({2, 4, 8, 8, 1});

    EXPECT_NO_THROW(GpuFpReferenceLayernorm::fprop<float>(
        x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 1, &mean, &rstd));
}

// --- validateConsistentDimensions() throw paths ---

TEST(TestGpuLayernormFwdRefValidation, ThrowsOnInputRankTooSmall)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8});
    Tensor<float> scale({1, 4, 8});
    Tensor<float> bias({1, 4, 8});
    Tensor<float> y({2, 4, 8});

    EXPECT_THROW(
        GpuFpReferenceLayernorm::fprop<float>(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 2),
        std::invalid_argument);
}

TEST(TestGpuLayernormFwdRefValidation, ThrowsOnInputRankTooLarge)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8, 8, 8});
    Tensor<float> scale({1, 4, 8, 8, 8, 8});
    Tensor<float> bias({1, 4, 8, 8, 8, 8});
    Tensor<float> y({2, 4, 8, 8, 8, 8});

    EXPECT_THROW(
        GpuFpReferenceLayernorm::fprop<float>(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 5),
        std::invalid_argument);
}

TEST(TestGpuLayernormFwdRefValidation, ThrowsOnScaleMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 4, 8});
    Tensor<float> bias({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8});

    EXPECT_THROW(
        GpuFpReferenceLayernorm::fprop<float>(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 3),
        std::invalid_argument);
}

TEST(TestGpuLayernormFwdRefValidation, ThrowsOnBiasMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> bias({1, 4, 4, 8});
    Tensor<float> y({2, 4, 8, 8});

    EXPECT_THROW(
        GpuFpReferenceLayernorm::fprop<float>(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 3),
        std::invalid_argument);
}

TEST(TestGpuLayernormFwdRefValidation, ThrowsOnOutputMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> bias({1, 4, 8, 8});
    Tensor<float> y({2, 4, 4, 8});

    EXPECT_THROW(
        GpuFpReferenceLayernorm::fprop<float>(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 3),
        std::invalid_argument);
}

TEST(TestGpuLayernormFwdRefValidation, ThrowsOnMeanMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> bias({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8});
    Tensor<double> mean({4, 1, 1, 1});
    Tensor<double> rstd({2, 1, 1, 1});

    EXPECT_THROW(GpuFpReferenceLayernorm::fprop<float>(
                     x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 3, &mean, &rstd),
                 std::invalid_argument);
}

TEST(TestGpuLayernormFwdRefValidation, ThrowsOnRstdMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> bias({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8});
    Tensor<double> mean({2, 1, 1, 1});
    Tensor<double> rstd({4, 1, 1, 1});

    EXPECT_THROW(GpuFpReferenceLayernorm::fprop<float>(
                     x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 3, &mean, &rstd),
                 std::invalid_argument);
}

TEST(TestGpuLayernormFwdRefValidation, ThrowsOnNormalizedDimCountMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> bias({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8});

    EXPECT_THROW(
        GpuFpReferenceLayernorm::fprop<float>(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 2),
        std::invalid_argument);
}

// --- validateConsistentLayouts() throw paths ---

TEST(TestGpuLayernormFwdRefValidation, ThrowsOnInputLayoutNeitherChannelFirstNorLast)
{
    SKIP_IF_NO_DEVICES();
    // Random strides that don't correspond to either channel-first or channel-last layout
    Tensor<float> x({2, 4, 8, 8}, std::vector<int64_t>{1, 2, 3, 4});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> bias({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8});

    EXPECT_THROW(
        GpuFpReferenceLayernorm::fprop<float>(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 3),
        std::invalid_argument);
}

TEST(TestGpuLayernormFwdRefValidation, ThrowsOnOutputLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> bias({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8}, TensorLayout::NHWC);

    EXPECT_THROW(
        GpuFpReferenceLayernorm::fprop<float>(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 3),
        std::invalid_argument);
}

TEST(TestGpuLayernormFwdRefValidation, ThrowsOnScaleLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> bias({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8});

    EXPECT_THROW(
        GpuFpReferenceLayernorm::fprop<float>(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 3),
        std::invalid_argument);
}

TEST(TestGpuLayernormFwdRefValidation, ThrowsOnBiasLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> bias({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> y({2, 4, 8, 8}, TensorLayout::NHWC);

    EXPECT_THROW(
        GpuFpReferenceLayernorm::fprop<float>(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 3),
        std::invalid_argument);
}

TEST(TestGpuLayernormFwdRefValidation, ThrowsOnMeanLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 1, 8, 8});
    Tensor<float> bias({1, 1, 8, 8});
    Tensor<float> y({2, 4, 8, 8});
    Tensor<double> mean({2, 4, 1, 1}, TensorLayout::NHWC);
    Tensor<double> rstd({2, 4, 1, 1});

    EXPECT_THROW(GpuFpReferenceLayernorm::fprop<float>(
                     x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 2, &mean, &rstd),
                 std::invalid_argument);
}

TEST(TestGpuLayernormFwdRefValidation, ThrowsOnRstdLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 1, 8, 8});
    Tensor<float> bias({1, 1, 8, 8});
    Tensor<float> y({2, 4, 8, 8});
    Tensor<double> mean({2, 4, 1, 1});
    Tensor<double> rstd({2, 4, 1, 1}, TensorLayout::NHWC);

    EXPECT_THROW(GpuFpReferenceLayernorm::fprop<float>(
                     x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 2, &mean, &rstd),
                 std::invalid_argument);
}

// --- Mixed type tests ---

TEST(TestGpuLayernormFwdRefMixedType, FloatInputHalfScaleBias)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({2, 3, 4, 4});
    Tensor<half> scaleTensor({1, 3, 4, 4});
    Tensor<half> biasTensor({1, 3, 4, 4});
    Tensor<float> yCpu({2, 3, 4, 4});
    Tensor<float> yGpu({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    scaleTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 1);
    biasTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 2);

    CpuFpReferenceLayernorm::fprop<float, half, float, double>(
        xTensor, &scaleTensor, &biasTensor, yCpu, LAYERNORM_DEFAULT_EPSILON, 3);

    GpuFpReferenceLayernorm::fprop<float, half, float, double>(
        xTensor, &scaleTensor, &biasTensor, yGpu, LAYERNORM_DEFAULT_EPSILON, 3);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

TEST(TestGpuLayernormFwdRefMixedType, HalfInputFloatScaleBias)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> xTensor({2, 3, 4, 4});
    Tensor<float> scaleTensor({1, 3, 4, 4});
    Tensor<float> biasTensor({1, 3, 4, 4});
    Tensor<half> yCpu({2, 3, 4, 4});
    Tensor<half> yGpu({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed);
    scaleTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    biasTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 2);

    CpuFpReferenceLayernorm::fprop<half, float, half, double>(
        xTensor, &scaleTensor, &biasTensor, yCpu, LAYERNORM_DEFAULT_EPSILON, 3);

    GpuFpReferenceLayernorm::fprop<half, float, half, double>(
        xTensor, &scaleTensor, &biasTensor, yGpu, LAYERNORM_DEFAULT_EPSILON, 3);

    assertAllClose(yCpu, yGpu, getTolerance<half>());
}

TEST(TestGpuLayernormFwdRefMixedType, HalfInputHalfScaleBias)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> xTensor({2, 3, 4, 4});
    Tensor<half> scaleTensor({1, 3, 4, 4});
    Tensor<half> biasTensor({1, 3, 4, 4});
    Tensor<half> yCpu({2, 3, 4, 4});
    Tensor<half> yGpu({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed);
    scaleTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 1);
    biasTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 2);

    CpuFpReferenceLayernorm::fprop<half, half, half, double>(
        xTensor, &scaleTensor, &biasTensor, yCpu, LAYERNORM_DEFAULT_EPSILON, 3);

    GpuFpReferenceLayernorm::fprop<half, half, half, double>(
        xTensor, &scaleTensor, &biasTensor, yGpu, LAYERNORM_DEFAULT_EPSILON, 3);

    assertAllClose(yCpu, yGpu, getTolerance<half>());
}

TEST(TestGpuLayernormFwdRefMixedType, BfloatInputFloatOutput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<bfloat16> xTensor({2, 3, 4, 4});
    Tensor<bfloat16> scaleTensor({1, 3, 4, 4});
    Tensor<bfloat16> biasTensor({1, 3, 4, 4});
    Tensor<float> yCpu({2, 3, 4, 4});
    Tensor<float> yGpu({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed);
    scaleTensor.fillWithRandomValues(
        static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed + 1);
    biasTensor.fillWithRandomValues(
        static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed + 2);

    CpuFpReferenceLayernorm::fprop<bfloat16, bfloat16, float, double>(
        xTensor, &scaleTensor, &biasTensor, yCpu, LAYERNORM_DEFAULT_EPSILON, 3);

    GpuFpReferenceLayernorm::fprop<bfloat16, bfloat16, float, double>(
        xTensor, &scaleTensor, &biasTensor, yGpu, LAYERNORM_DEFAULT_EPSILON, 3);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

TEST(TestGpuLayernormFwdRefMixedType, BfloatInputHalfScaleBias)
{
    SKIP_IF_NO_DEVICES();

    Tensor<bfloat16> xTensor({2, 3, 4, 4});
    Tensor<half> scaleTensor({1, 3, 4, 4});
    Tensor<half> biasTensor({1, 3, 4, 4});
    Tensor<bfloat16> yCpu({2, 3, 4, 4});
    Tensor<bfloat16> yGpu({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed);
    scaleTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 1);
    biasTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 2);

    CpuFpReferenceLayernorm::fprop<bfloat16, half, bfloat16, double>(
        xTensor, &scaleTensor, &biasTensor, yCpu, LAYERNORM_DEFAULT_EPSILON, 3);

    GpuFpReferenceLayernorm::fprop<bfloat16, half, bfloat16, double>(
        xTensor, &scaleTensor, &biasTensor, yGpu, LAYERNORM_DEFAULT_EPSILON, 3);

    assertAllClose(yCpu, yGpu, getTolerance<bfloat16>());
}

// --- Optional argument tests ---

TEST(TestGpuLayernormFwdRefOptionalArgs, WithMeanAndRstd)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({2, 4, 8, 8});
    Tensor<float> scaleTensor({1, 4, 8, 8});
    Tensor<float> biasTensor({1, 4, 8, 8});
    Tensor<float> yCpu({2, 4, 8, 8});
    Tensor<float> yGpu({2, 4, 8, 8});
    Tensor<double> meanCpu({2, 1, 1, 1});
    Tensor<double> meanGpu({2, 1, 1, 1});
    Tensor<double> rstdCpu({2, 1, 1, 1});
    Tensor<double> rstdGpu({2, 1, 1, 1});

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    scaleTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    biasTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 2);

    CpuFpReferenceLayernorm::fprop<float, float, float, double>(
        xTensor, &scaleTensor, &biasTensor, yCpu, LAYERNORM_DEFAULT_EPSILON, 3, &meanCpu, &rstdCpu);

    GpuFpReferenceLayernorm::fprop<float, float, float, double>(
        xTensor, &scaleTensor, &biasTensor, yGpu, LAYERNORM_DEFAULT_EPSILON, 3, &meanGpu, &rstdGpu);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
    assertAllClose(meanCpu, meanGpu, getTolerance<double>());
    assertAllClose(rstdCpu, rstdGpu, getTolerance<double>());
}

// -- Channel-last layout tests ---

TEST(TestGpuLayernormFwdRefChannelLast, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> scaleTensor({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> biasTensor({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> yCpu({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> yGpu({2, 4, 8, 8}, TensorLayout::NHWC);

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    scaleTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    biasTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 2);

    CpuFpReferenceLayernorm::fprop<float, float, float, double>(
        xTensor, &scaleTensor, &biasTensor, yCpu, LAYERNORM_DEFAULT_EPSILON, 3);

    GpuFpReferenceLayernorm::fprop<float, float, float, double>(
        xTensor, &scaleTensor, &biasTensor, yGpu, LAYERNORM_DEFAULT_EPSILON, 3);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

TEST(TestGpuLayernormFwdRefChannelLast, MatchesCpuRefWithMeanAndRstd)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> scaleTensor({1, 1, 8, 8}, TensorLayout::NHWC);
    Tensor<float> biasTensor({1, 1, 8, 8}, TensorLayout::NHWC);
    Tensor<float> yCpu({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> yGpu({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<double> meanCpu({2, 4, 1, 1}, TensorLayout::NHWC);
    Tensor<double> meanGpu({2, 4, 1, 1}, TensorLayout::NHWC);
    Tensor<double> rstdCpu({2, 4, 1, 1}, TensorLayout::NHWC);
    Tensor<double> rstdGpu({2, 4, 1, 1}, TensorLayout::NHWC);

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    scaleTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    biasTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 2);

    CpuFpReferenceLayernorm::fprop<float, float, float, double>(
        xTensor, &scaleTensor, &biasTensor, yCpu, LAYERNORM_DEFAULT_EPSILON, 2, &meanCpu, &rstdCpu);

    GpuFpReferenceLayernorm::fprop<float, float, float, double>(
        xTensor, &scaleTensor, &biasTensor, yGpu, LAYERNORM_DEFAULT_EPSILON, 2, &meanGpu, &rstdGpu);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
    assertAllClose(meanCpu, meanGpu, getTolerance<double>());
    assertAllClose(rstdCpu, rstdGpu, getTolerance<double>());
}

// --- Test suite instantiations ---

using TestGpuLayernormFwdRef4DFp32 = LayernormFwdTestSuite<float, float, float, double>;
using TestGpuLayernormFwdRef4DFp16 = LayernormFwdTestSuite<half, half, half, double>;
using TestGpuLayernormFwdRef4DBfp16 = LayernormFwdTestSuite<bfloat16, bfloat16, bfloat16, double>;
using TestGpuLayernormFwdRef5DFp32 = LayernormFwdTestSuite<float, float, float, double>;
using TestGpuLayernormFwdRef5DFp16 = LayernormFwdTestSuite<half, half, half, double>;
using TestGpuLayernormFwdRef5DBfp16 = LayernormFwdTestSuite<bfloat16, bfloat16, bfloat16, double>;

TEST_P(TestGpuLayernormFwdRef4DFp32, MatchesCpuRef)
{
    this->runLayernormFwdTest();
}
TEST_P(TestGpuLayernormFwdRef4DFp16, MatchesCpuRef)
{
    this->runLayernormFwdTest();
}
TEST_P(TestGpuLayernormFwdRef4DBfp16, MatchesCpuRef)
{
    this->runLayernormFwdTest();
}
TEST_P(TestGpuLayernormFwdRef5DFp32, MatchesCpuRef)
{
    this->runLayernormFwdTest();
}
TEST_P(TestGpuLayernormFwdRef5DFp16, MatchesCpuRef)
{
    this->runLayernormFwdTest();
}
TEST_P(TestGpuLayernormFwdRef5DBfp16, MatchesCpuRef)
{
    this->runLayernormFwdTest();
}

// ============================================================================
// 4D (NCHW/NHWC) tests
// ============================================================================

// --- Quick tests ---

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuLayernormFwdRef4DFp32,
                         ::testing::ValuesIn(getLayernormSmall4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuLayernormFwdRef4DFp16,
                         ::testing::ValuesIn(getLayernormSmall4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuLayernormFwdRef4DBfp16,
                         ::testing::ValuesIn(getLayernormSmall4DTestCases()));

INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuLayernormFwdRef4DFp32,
                         ::testing::ValuesIn(getLayernormMedium4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuLayernormFwdRef4DFp16,
                         ::testing::ValuesIn(getLayernormMedium4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuLayernormFwdRef4DBfp16,
                         ::testing::ValuesIn(getLayernormMedium4DTestCases()));

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuLayernormFwdRef4DFp32,
                         ::testing::ValuesIn(getLayernormLarge4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuLayernormFwdRef4DFp16,
                         ::testing::ValuesIn(getLayernormLarge4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuLayernormFwdRef4DBfp16,
                         ::testing::ValuesIn(getLayernormLarge4DTestCases()));

INSTANTIATE_TEST_SUITE_P(Full, TestGpuLayernormFwdRef4DFp32, ::testing::ValuesIn([]() {
                             auto v = getLayernormSmall4DTestCases();
                             auto m = getLayernormMedium4DTestCases();
                             auto l = getLayernormLarge4DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuLayernormFwdRef4DFp16, ::testing::ValuesIn([]() {
                             auto v = getLayernormSmall4DTestCases();
                             auto m = getLayernormMedium4DTestCases();
                             auto l = getLayernormLarge4DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuLayernormFwdRef4DBfp16, ::testing::ValuesIn([]() {
                             auto v = getLayernormSmall4DTestCases();
                             auto m = getLayernormMedium4DTestCases();
                             auto l = getLayernormLarge4DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));

// ============================================================================
// 5D (NCDHW/NDHWC) shape tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuLayernormFwdRef5DFp32,
                         ::testing::ValuesIn(getLayernormSmall5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuLayernormFwdRef5DFp16,
                         ::testing::ValuesIn(getLayernormSmall5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuLayernormFwdRef5DBfp16,
                         ::testing::ValuesIn(getLayernormSmall5DTestCases()));

INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuLayernormFwdRef5DFp32,
                         ::testing::ValuesIn(getLayernormMedium5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuLayernormFwdRef5DFp16,
                         ::testing::ValuesIn(getLayernormMedium5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuLayernormFwdRef5DBfp16,
                         ::testing::ValuesIn(getLayernormMedium5DTestCases()));

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuLayernormFwdRef5DFp32,
                         ::testing::ValuesIn(getLayernormLarge5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuLayernormFwdRef5DFp16,
                         ::testing::ValuesIn(getLayernormLarge5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuLayernormFwdRef5DBfp16,
                         ::testing::ValuesIn(getLayernormLarge5DTestCases()));

INSTANTIATE_TEST_SUITE_P(Full, TestGpuLayernormFwdRef5DFp32, ::testing::ValuesIn([]() {
                             auto v = getLayernormSmall5DTestCases();
                             auto m = getLayernormMedium5DTestCases();
                             auto l = getLayernormLarge5DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuLayernormFwdRef5DFp16, ::testing::ValuesIn([]() {
                             auto v = getLayernormSmall5DTestCases();
                             auto m = getLayernormMedium5DTestCases();
                             auto l = getLayernormLarge5DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuLayernormFwdRef5DBfp16, ::testing::ValuesIn([]() {
                             auto v = getLayernormSmall5DTestCases();
                             auto m = getLayernormMedium5DTestCases();
                             auto l = getLayernormLarge5DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));

// ============================================================================
// Edge case tests with DISABLED_ prefix to avoid running in CI.
// Run the tests manually with --gtest_also_run_disabled_tests
// --gtest_filter=*TestGpuLayernormFwdRefEdgeCaseValidation* flags.
// ============================================================================

namespace
{

int64_t getMaxOuterSizeForCurrentDevice()
{
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
    {
        // No devices available, return a default value to skip the tests.
        return 1;
    }

    int deviceId = 0;
    EXPECT_EQ(hipGetDevice(&deviceId), hipSuccess);

    hipDeviceProp_t props{};
    EXPECT_EQ(hipGetDeviceProperties(&props, deviceId), hipSuccess);

    return static_cast<int64_t>(props.maxGridSize[0]);
}

} // namespace

TEST(TestGpuLayernormFwdRefEdgeCaseValidation, DISABLED_OuterSizeAtMaxBlocksMinusOneSucceeds)
{
    SKIP_IF_NO_DEVICES();
    const int64_t outerSize = getMaxOuterSizeForCurrentDevice() - 1;
    Tensor<float> x({outerSize, 1, 1, 1});
    Tensor<float> scale({1, 1, 1, 1});
    Tensor<float> bias({1, 1, 1, 1});
    Tensor<float> y({outerSize, 1, 1, 1});

    EXPECT_NO_THROW(
        GpuFpReferenceLayernorm::fprop<float>(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 3));
}

TEST(TestGpuLayernormFwdRefEdgeCaseValidation, DISABLED_OuterSizeAtMaxBlocksSucceeds)
{
    SKIP_IF_NO_DEVICES();
    const int64_t outerSize = getMaxOuterSizeForCurrentDevice();
    Tensor<float> x({outerSize, 1, 1, 1});
    Tensor<float> scale({1, 1, 1, 1});
    Tensor<float> bias({1, 1, 1, 1});
    Tensor<float> y({outerSize, 1, 1, 1});

    EXPECT_NO_THROW(
        GpuFpReferenceLayernorm::fprop<float>(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 3));
}

TEST(TestGpuLayernormFwdRefEdgeCaseValidation, DISABLED_OuterSizeAboveMaxBlocksThrows)
{
    SKIP_IF_NO_DEVICES();
    const int64_t outerSize = getMaxOuterSizeForCurrentDevice() + 1;
    Tensor<float> x({outerSize, 1, 1, 1});
    Tensor<float> scale({1, 1, 1, 1});
    Tensor<float> bias({1, 1, 1, 1});
    Tensor<float> y({outerSize, 1, 1, 1});

    EXPECT_THROW(
        GpuFpReferenceLayernorm::fprop<float>(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 3),
        std::runtime_error);
}

TEST(TestGpuLayernormFwdRefEdgeCaseValidation, DISABLED_BeyondInt32InnerSizeIfMemoryAllows)
{
    SKIP_IF_NO_DEVICES();

    size_t freeBytes = 0;
    size_t totalBytes = 0;
    ASSERT_EQ(hipMemGetInfo(&freeBytes, &totalBytes), hipSuccess);

    constexpr int64_t OUTER_SIZE = 1;
    // NOTE: INNER_SIZE in this test should be 2^32+1, but is reduced here due to
    // slow CPU fill/reference functions. Revisit once rocRAND-based GPU fill and
    // golden references for large tensors are available.
    constexpr int64_t INNER_SIZE = 100000000; // 100 million elements

    Tensor<float> x({OUTER_SIZE, 1, 1, INNER_SIZE});
    Tensor<float> scale({1, 1, 1, INNER_SIZE});
    Tensor<float> bias({1, 1, 1, INNER_SIZE});
    Tensor<float> yCpu({OUTER_SIZE, 1, 1, INNER_SIZE});
    Tensor<float> yGpu({OUTER_SIZE, 1, 1, INNER_SIZE});

    // Calculate the required memory for input, scale, and output tensors
    const size_t requiredBytes
        = (x.elementCount() + scale.elementCount() + bias.elementCount() + yGpu.elementCount())
          * sizeof(float);
    if(requiredBytes > freeBytes)
    {
        GTEST_SKIP() << "Insufficient GPU memory for the test. Required: " << requiredBytes
                     << " bytes, Free: " << freeBytes << " bytes.";
    }

    const unsigned int seed = getGlobalTestSeed();
    x.fillWithRandomValues(-1.0f, 1.0f, seed);
    scale.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    bias.fillWithRandomValues(-1.0f, 1.0f, seed + 2);

    CpuFpReferenceLayernorm::fprop<float, float, float, double>(
        x, &scale, &bias, yCpu, LAYERNORM_DEFAULT_EPSILON, 3);
    GpuFpReferenceLayernorm::fprop<float, float, float, double>(
        x, &scale, &bias, yGpu, LAYERNORM_DEFAULT_EPSILON, 3);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

using TestGpuLayernormFwdRefEdgeCaseValidationFp32
    = LayernormFwdTestSuite<float, float, float, double>;

TEST_P(TestGpuLayernormFwdRefEdgeCaseValidationFp32, DISABLED_MatchesCpuRef)
{
    this->runLayernormFwdTest();
}

INSTANTIATE_TEST_SUITE_P(SkinnyModerate,
                         TestGpuLayernormFwdRefEdgeCaseValidationFp32,
                         ::testing::ValuesIn(getLayernormSkinnyModerateTestCases()));

INSTANTIATE_TEST_SUITE_P(PowerOfTwo,
                         TestGpuLayernormFwdRefEdgeCaseValidationFp32,
                         ::testing::ValuesIn(getLayernormPowerOfTwoTestCases()));

INSTANTIATE_TEST_SUITE_P(
    SkinnyInt32Scale, TestGpuLayernormFwdRefEdgeCaseValidationFp32, ::testing::ValuesIn([]() {
        return getLayernormSkinnyInt32ScaleTestCases(getMaxOuterSizeForCurrentDevice());
    }()));

INSTANTIATE_TEST_SUITE_P(InnerSizeInt32Boundary,
                         TestGpuLayernormFwdRefEdgeCaseValidationFp32,
                         ::testing::ValuesIn(getLayernormInnerSizeInt32BoundaryTestCases()));
