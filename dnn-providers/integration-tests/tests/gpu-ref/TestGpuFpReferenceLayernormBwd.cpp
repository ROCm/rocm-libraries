// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GpuLayernormBwdRefTestFixture.hpp"
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <stdexcept>

// --- Valid configurations ---

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::layernorm;
using namespace hipdnn_gpu_ref;
using namespace gpu_layernorm_ref_test;
using namespace gpu_layernorm_bwd_ref_test;

TEST(TestGpuLayernormBwdRefValidation, AcceptsValidParams4D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});
    Tensor<float> dbias({1, 4, 8, 8});

    EXPECT_NO_THROW((GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
        dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 3)));
}

TEST(TestGpuLayernormBwdRefValidation, AcceptsValidParams5D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8, 8});
    Tensor<float> x({2, 4, 8, 8, 8});
    Tensor<float> scale({1, 4, 8, 8, 8});
    Tensor<float> dx({2, 4, 8, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8, 8});
    Tensor<float> dbias({1, 4, 8, 8, 8});

    EXPECT_NO_THROW((GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
        dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 4)));
}

TEST(TestGpuLayernormBwdRefValidation, AcceptsValidParamsChannelLastLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> x({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> scale({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dx({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dscale({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dbias({1, 4, 8, 8}, TensorLayout::NHWC);

    EXPECT_NO_THROW((GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
        dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 3)));
}

TEST(TestGpuLayernormBwdRefValidation, AcceptsValidParamsWithMeanAndRstd)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});
    Tensor<float> dbias({1, 4, 8, 8});
    Tensor<double> mean({2, 1, 1, 1});
    Tensor<double> rstd({2, 1, 1, 1});

    EXPECT_NO_THROW((GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
        dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, &mean, &rstd, 3)));
}

TEST(TestGpuLayernormBwdRefValidation, AcceptsValidParams4NormalizeDimTwoD)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 1, 8, 8});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 1, 8, 8});
    Tensor<float> dbias({1, 1, 8, 8});

    EXPECT_NO_THROW((GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
        dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 2)));
}

TEST(TestGpuLayernormBwdRefValidation, AcceptsValidParamsNormalizeDimThree4D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 1, 1, 8});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 1, 1, 8});
    Tensor<float> dbias({1, 1, 1, 8});

    EXPECT_NO_THROW((GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
        dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 1)));
}

TEST(TestGpuLayernormBwdRefValidation, AcceptsValidParamsNormalizeDimThree5D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8, 8});
    Tensor<float> x({2, 4, 8, 8, 8});
    Tensor<float> scale({1, 1, 1, 8, 8});
    Tensor<float> dx({2, 4, 8, 8, 8});
    Tensor<float> dscale({1, 1, 1, 8, 8});
    Tensor<float> dbias({1, 1, 1, 8, 8});

    EXPECT_NO_THROW((GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
        dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 2)));
}

TEST(TestGpuLayernormBwdRefValidation, AcceptsValidParamsNormalizeDimFour5D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8, 8});
    Tensor<float> x({2, 4, 8, 8, 8});
    Tensor<float> scale({1, 1, 1, 1, 8});
    Tensor<float> dx({2, 4, 8, 8, 8});
    Tensor<float> dscale({1, 1, 1, 1, 8});
    Tensor<float> dbias({1, 1, 1, 1, 8});

    EXPECT_NO_THROW((GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
        dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 1)));
}

// --- validateConsistentDimensions() throw paths ---

TEST(TestGpuLayernormBwdRefValidation, ThrowsOnInputRankTooSmall)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8});
    Tensor<float> x({2, 4, 8});
    Tensor<float> scale({1, 4, 8});
    Tensor<float> dx({2, 4, 8});
    Tensor<float> dscale({1, 4, 8});
    Tensor<float> dbias({1, 4, 8});

    EXPECT_THROW(
        (GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
            dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 2)),
        std::invalid_argument);
}

TEST(TestGpuLayernormBwdRefValidation, ThrowsOnInputRankTooLarge)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8, 8, 8});
    Tensor<float> x({2, 4, 8, 8, 8, 8});
    Tensor<float> scale({1, 4, 8, 8, 8, 8});
    Tensor<float> dx({2, 4, 8, 8, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8, 8, 8});
    Tensor<float> dbias({1, 4, 8, 8, 8, 8});

    EXPECT_THROW(
        (GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
            dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 5)),
        std::invalid_argument);
}

TEST(TestGpuLayernormBwdRefValidation, ThrowsOnScaleMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 4, 8});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});
    Tensor<float> dbias({1, 4, 8, 8});

    EXPECT_THROW(
        (GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
            dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 3)),
        std::invalid_argument);
}

TEST(TestGpuLayernormBwdRefValidation, ThrowsOnDscaleMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 4, 8});
    Tensor<float> dbias({1, 4, 8, 8});

    EXPECT_THROW(
        (GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
            dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 3)),
        std::invalid_argument);
}

TEST(TestGpuLayernormBwdRefValidation, ThrowsOnDbiasMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});
    Tensor<float> dbias({1, 4, 4, 8});

    EXPECT_THROW(
        (GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
            dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 3)),
        std::invalid_argument);
}

TEST(TestGpuLayernormBwdRefValidation, ThrowsOnOutputMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> dx({2, 4, 4, 8});
    Tensor<float> dscale({1, 4, 8, 8});
    Tensor<float> dbias({1, 4, 8, 8});

    EXPECT_THROW(
        (GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
            dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 3)),
        std::invalid_argument);
}

TEST(TestGpuLayernormBwdRefValidation, ThrowsOnMeanMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});
    Tensor<float> dbias({1, 4, 8, 8});
    Tensor<double> mean({4, 1, 1, 1});
    Tensor<double> rstd({2, 1, 1, 1});

    EXPECT_THROW((GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
                     dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, &mean, &rstd, 3)),
                 std::invalid_argument);
}

TEST(TestGpuLayernormBwdRefValidation, ThrowsOnRstdMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});
    Tensor<float> dbias({1, 4, 8, 8});
    Tensor<double> mean({2, 1, 1, 1});
    Tensor<double> rstd({4, 1, 1, 1});

    EXPECT_THROW((GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
                     dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, &mean, &rstd, 3)),
                 std::invalid_argument);
}

TEST(TestGpuLayernormBwdRefValidation, ThrowsOnNormalizedDimCountMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});
    Tensor<float> dbias({1, 4, 8, 8});

    EXPECT_THROW(
        (GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
            dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 2)),
        std::invalid_argument);
}

// --- validateConsistentLayouts() throw paths ---

TEST(TestGpuLayernormBwdRefValidation, ThrowsOnInputLayoutNeitherChannelFirstNorLast)
{
    SKIP_IF_NO_DEVICES();
    // Random strides that don't correspond to either channel-first or channel-last layout
    Tensor<float> dy({2, 4, 8, 8}, std::vector<int64_t>{1, 2, 3, 4});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});
    Tensor<float> dbias({1, 4, 8, 8});

    EXPECT_THROW(
        (GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
            dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 3)),
        std::invalid_argument);
}

TEST(TestGpuLayernormBwdRefValidation, ThrowsOnOutputLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> dx({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dscale({1, 4, 8, 8});
    Tensor<float> dbias({1, 4, 8, 8});

    EXPECT_THROW(
        (GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
            dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 3)),
        std::invalid_argument);
}

TEST(TestGpuLayernormBwdRefValidation, ThrowsOnXLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});
    Tensor<float> dbias({1, 4, 8, 8});

    EXPECT_THROW(
        (GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
            dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 3)),
        std::invalid_argument);
}

TEST(TestGpuLayernormBwdRefValidation, ThrowsOnScaleLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});
    Tensor<float> dbias({1, 4, 8, 8});

    EXPECT_THROW(
        (GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
            dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 3)),
        std::invalid_argument);
}

TEST(TestGpuLayernormBwdRefValidation, ThrowsOnDscaleLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dbias({1, 4, 8, 8});

    EXPECT_THROW(
        (GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
            dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 3)),
        std::invalid_argument);
}

TEST(TestGpuLayernormBwdRefValidation, ThrowsOnDbiasLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});
    Tensor<float> dbias({1, 4, 8, 8}, TensorLayout::NHWC);

    EXPECT_THROW(
        (GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
            dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 3)),
        std::invalid_argument);
}

TEST(TestGpuLayernormBwdRefValidation, ThrowsOnMeanLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 1, 8, 8});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 1, 8, 8});
    Tensor<float> dbias({1, 1, 8, 8});
    Tensor<double> mean({2, 4, 1, 1}, TensorLayout::NHWC);
    Tensor<double> rstd({2, 4, 1, 1});

    EXPECT_THROW((GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
                     dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, &mean, &rstd, 2)),
                 std::invalid_argument);
}

TEST(TestGpuLayernormBwdRefValidation, ThrowsOnRstdLayoutInconsistentWithInput)
{
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 1, 8, 8});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 1, 8, 8});
    Tensor<float> dbias({1, 1, 8, 8});
    Tensor<double> mean({2, 4, 1, 1});
    Tensor<double> rstd({2, 4, 1, 1}, TensorLayout::NHWC);

    EXPECT_THROW((GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
                     dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, &mean, &rstd, 2)),
                 std::invalid_argument);
}

// --- Mixed type tests ---

TEST(TestGpuLayernormBwdRefMixedType, FloatInputHalfScaleBias)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> dyTensor({2, 3, 4, 4});
    Tensor<float> xTensor({2, 3, 4, 4});
    Tensor<half> scaleTensor({1, 3, 4, 4});
    Tensor<float> dxCpu({2, 3, 4, 4});
    Tensor<float> dxGpu({2, 3, 4, 4});
    Tensor<half> dscaleCpu({1, 3, 4, 4});
    Tensor<half> dscaleGpu({1, 3, 4, 4});
    Tensor<half> dbiasCpu({1, 3, 4, 4});
    Tensor<half> dbiasGpu({1, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    scaleTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 2);

    CpuFpReferenceLayernorm::bprop<float, half, float, double, double>(dyTensor,
                                                                       xTensor,
                                                                       scaleTensor,
                                                                       dxCpu,
                                                                       dscaleCpu,
                                                                       dbiasCpu,
                                                                       LAYERNORM_DEFAULT_EPSILON,
                                                                       nullptr,
                                                                       nullptr,
                                                                       3);

    GpuFpReferenceLayernorm::bprop<float, half, float, double, double>(dyTensor,
                                                                       xTensor,
                                                                       scaleTensor,
                                                                       dxGpu,
                                                                       dscaleGpu,
                                                                       dbiasGpu,
                                                                       LAYERNORM_DEFAULT_EPSILON,
                                                                       nullptr,
                                                                       nullptr,
                                                                       3);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<half>());
    assertAllClose(dbiasCpu, dbiasGpu, getTolerance<half>());
}

TEST(TestGpuLayernormBwdRefMixedType, HalfInputFloatScaleBias)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> dyTensor({2, 3, 4, 4});
    Tensor<half> xTensor({2, 3, 4, 4});
    Tensor<float> scaleTensor({1, 3, 4, 4});
    Tensor<half> dxCpu({2, 3, 4, 4});
    Tensor<half> dxGpu({2, 3, 4, 4});
    Tensor<float> dscaleCpu({1, 3, 4, 4});
    Tensor<float> dscaleGpu({1, 3, 4, 4});
    Tensor<float> dbiasCpu({1, 3, 4, 4});
    Tensor<float> dbiasGpu({1, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    dyTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed);
    xTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 1);
    scaleTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 2);

    CpuFpReferenceLayernorm::bprop<half, float, half, double, double>(dyTensor,
                                                                      xTensor,
                                                                      scaleTensor,
                                                                      dxCpu,
                                                                      dscaleCpu,
                                                                      dbiasCpu,
                                                                      LAYERNORM_DEFAULT_EPSILON,
                                                                      nullptr,
                                                                      nullptr,
                                                                      3);

    GpuFpReferenceLayernorm::bprop<half, float, half, double, double>(dyTensor,
                                                                      xTensor,
                                                                      scaleTensor,
                                                                      dxGpu,
                                                                      dscaleGpu,
                                                                      dbiasGpu,
                                                                      LAYERNORM_DEFAULT_EPSILON,
                                                                      nullptr,
                                                                      nullptr,
                                                                      3);

    assertAllClose(dxCpu, dxGpu, getTolerance<half>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<float>());
    assertAllClose(dbiasCpu, dbiasGpu, getTolerance<float>());
}

TEST(TestGpuLayernormBwdRefMixedType, HalfInputHalfScaleBias)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> dyTensor({2, 3, 4, 4});
    Tensor<half> xTensor({2, 3, 4, 4});
    Tensor<half> scaleTensor({1, 3, 4, 4});
    Tensor<half> dxCpu({2, 3, 4, 4});
    Tensor<half> dxGpu({2, 3, 4, 4});
    Tensor<half> dscaleCpu({1, 3, 4, 4});
    Tensor<half> dscaleGpu({1, 3, 4, 4});
    Tensor<half> dbiasCpu({1, 3, 4, 4});
    Tensor<half> dbiasGpu({1, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    dyTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed);
    xTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 1);
    scaleTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 2);

    CpuFpReferenceLayernorm::bprop<half, half, half, double, double>(dyTensor,
                                                                     xTensor,
                                                                     scaleTensor,
                                                                     dxCpu,
                                                                     dscaleCpu,
                                                                     dbiasCpu,
                                                                     LAYERNORM_DEFAULT_EPSILON,
                                                                     nullptr,
                                                                     nullptr,
                                                                     3);

    GpuFpReferenceLayernorm::bprop<half, half, half, double, double>(dyTensor,
                                                                     xTensor,
                                                                     scaleTensor,
                                                                     dxGpu,
                                                                     dscaleGpu,
                                                                     dbiasGpu,
                                                                     LAYERNORM_DEFAULT_EPSILON,
                                                                     nullptr,
                                                                     nullptr,
                                                                     3);

    assertAllClose(dxCpu, dxGpu, getTolerance<half>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<half>());
    assertAllClose(dbiasCpu, dbiasGpu, getTolerance<half>());
}

TEST(TestGpuLayernormBwdRefMixedType, BfloatInputFloatOutput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> dyTensor({2, 3, 4, 4});
    Tensor<bfloat16> xTensor({2, 3, 4, 4});
    Tensor<bfloat16> scaleTensor({1, 3, 4, 4});
    Tensor<bfloat16> dxCpu({2, 3, 4, 4});
    Tensor<bfloat16> dxGpu({2, 3, 4, 4});
    Tensor<bfloat16> dscaleCpu({1, 3, 4, 4});
    Tensor<bfloat16> dscaleGpu({1, 3, 4, 4});
    Tensor<bfloat16> dbiasCpu({1, 3, 4, 4});
    Tensor<bfloat16> dbiasGpu({1, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    xTensor.fillWithRandomValues(
        static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed + 1);
    scaleTensor.fillWithRandomValues(
        static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed + 2);

    CpuFpReferenceLayernorm::bprop<float, bfloat16, bfloat16, double, double>(
        dyTensor,
        xTensor,
        scaleTensor,
        dxCpu,
        dscaleCpu,
        dbiasCpu,
        LAYERNORM_DEFAULT_EPSILON,
        nullptr,
        nullptr,
        3);

    GpuFpReferenceLayernorm::bprop<float, bfloat16, bfloat16, double, double>(
        dyTensor,
        xTensor,
        scaleTensor,
        dxGpu,
        dscaleGpu,
        dbiasGpu,
        LAYERNORM_DEFAULT_EPSILON,
        nullptr,
        nullptr,
        3);

    assertAllClose(dxCpu, dxGpu, getTolerance<bfloat16>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<bfloat16>());
    assertAllClose(dbiasCpu, dbiasGpu, getTolerance<bfloat16>());
}

TEST(TestGpuLayernormBwdRefMixedType, BfloatInputHalfScaleBias)
{
    SKIP_IF_NO_DEVICES();

    Tensor<bfloat16> dyTensor({2, 3, 4, 4});
    Tensor<bfloat16> xTensor({2, 3, 4, 4});
    Tensor<half> scaleTensor({1, 3, 4, 4});
    Tensor<bfloat16> dxCpu({2, 3, 4, 4});
    Tensor<bfloat16> dxGpu({2, 3, 4, 4});
    Tensor<half> dscaleCpu({1, 3, 4, 4});
    Tensor<half> dscaleGpu({1, 3, 4, 4});
    Tensor<half> dbiasCpu({1, 3, 4, 4});
    Tensor<half> dbiasGpu({1, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    dyTensor.fillWithRandomValues(static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed);
    xTensor.fillWithRandomValues(
        static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed + 1);
    scaleTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 2);

    CpuFpReferenceLayernorm::bprop<bfloat16, half, bfloat16, double, double>(
        dyTensor,
        xTensor,
        scaleTensor,
        dxCpu,
        dscaleCpu,
        dbiasCpu,
        LAYERNORM_DEFAULT_EPSILON,
        nullptr,
        nullptr,
        3);

    GpuFpReferenceLayernorm::bprop<bfloat16, half, bfloat16, double, double>(
        dyTensor,
        xTensor,
        scaleTensor,
        dxGpu,
        dscaleGpu,
        dbiasGpu,
        LAYERNORM_DEFAULT_EPSILON,
        nullptr,
        nullptr,
        3);

    assertAllClose(dxCpu, dxGpu, getTolerance<bfloat16>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<half>());
    assertAllClose(dbiasCpu, dbiasGpu, getTolerance<half>());
}

// --- Optional argument tests ---

TEST(TestGpuLayernormBwdRefOptionalArgs, WithMeanAndRstd)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> dyTensor({2, 3, 4, 4});
    Tensor<float> xTensor({2, 3, 4, 4});
    Tensor<float> scaleTensor({1, 3, 4, 4});
    Tensor<double> meanTensor({2, 1, 1, 1});
    Tensor<double> rstdTensor({2, 1, 1, 1});
    Tensor<float> dxCpu({2, 3, 4, 4});
    Tensor<float> dxGpu({2, 3, 4, 4});
    Tensor<float> dscaleCpu({1, 3, 4, 4});
    Tensor<float> dscaleGpu({1, 3, 4, 4});
    Tensor<float> dbiasCpu({1, 3, 4, 4});
    Tensor<float> dbiasGpu({1, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    scaleTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 2);
    meanTensor.fillWithRandomValues(
        static_cast<double>(-1.0f), static_cast<double>(1.0f), seed + 3);
    rstdTensor.fillWithRandomValues(
        static_cast<double>(-1.0f), static_cast<double>(1.0f), seed + 4);

    CpuFpReferenceLayernorm::bprop<float, float, float, double, double>(dyTensor,
                                                                        xTensor,
                                                                        scaleTensor,
                                                                        dxCpu,
                                                                        dscaleCpu,
                                                                        dbiasCpu,
                                                                        LAYERNORM_DEFAULT_EPSILON,
                                                                        &meanTensor,
                                                                        &rstdTensor,
                                                                        3);

    GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(dyTensor,
                                                                        xTensor,
                                                                        scaleTensor,
                                                                        dxGpu,
                                                                        dscaleGpu,
                                                                        dbiasGpu,
                                                                        LAYERNORM_DEFAULT_EPSILON,
                                                                        &meanTensor,
                                                                        &rstdTensor,
                                                                        3);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<float>());
    assertAllClose(dbiasCpu, dbiasGpu, getTolerance<float>());
}

// -- Channel-last layout tests ---

TEST(TestGpuLayernormBwdRefChannelLast, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> dyTensor({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> xTensor({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> scaleTensor({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dxCpu({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dxGpu({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dscaleCpu({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dscaleGpu({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dbiasCpu({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dbiasGpu({1, 4, 8, 8}, TensorLayout::NHWC);

    const unsigned int seed = getGlobalTestSeed();
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    scaleTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 2);

    CpuFpReferenceLayernorm::bprop<float, float, float, double, double>(dyTensor,
                                                                        xTensor,
                                                                        scaleTensor,
                                                                        dxCpu,
                                                                        dscaleCpu,
                                                                        dbiasCpu,
                                                                        LAYERNORM_DEFAULT_EPSILON,
                                                                        nullptr,
                                                                        nullptr,
                                                                        3);

    GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(dyTensor,
                                                                        xTensor,
                                                                        scaleTensor,
                                                                        dxGpu,
                                                                        dscaleGpu,
                                                                        dbiasGpu,
                                                                        LAYERNORM_DEFAULT_EPSILON,
                                                                        nullptr,
                                                                        nullptr,
                                                                        3);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<float>());
    assertAllClose(dbiasCpu, dbiasGpu, getTolerance<float>());
}

TEST(TestGpuLayernormBwdRefChannelLast, MatchesCpuRefWithMeanAndRstd)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> dyTensor({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> xTensor({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> scaleTensor({1, 1, 8, 8}, TensorLayout::NHWC);
    Tensor<double> meanTensor({2, 4, 1, 1}, TensorLayout::NHWC);
    Tensor<double> rstdTensor({2, 4, 1, 1}, TensorLayout::NHWC);
    Tensor<float> dxCpu({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dxGpu({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dscaleCpu({1, 1, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dscaleGpu({1, 1, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dbiasCpu({1, 1, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dbiasGpu({1, 1, 8, 8}, TensorLayout::NHWC);

    const unsigned int seed = getGlobalTestSeed();
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    scaleTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 2);
    meanTensor.fillWithRandomValues(
        static_cast<double>(-1.0f), static_cast<double>(1.0f), seed + 3);
    rstdTensor.fillWithRandomValues(
        static_cast<double>(-1.0f), static_cast<double>(1.0f), seed + 4);

    CpuFpReferenceLayernorm::bprop<float, float, float, double, double>(dyTensor,
                                                                        xTensor,
                                                                        scaleTensor,
                                                                        dxCpu,
                                                                        dscaleCpu,
                                                                        dbiasCpu,
                                                                        LAYERNORM_DEFAULT_EPSILON,
                                                                        &meanTensor,
                                                                        &rstdTensor,
                                                                        2);

    GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(dyTensor,
                                                                        xTensor,
                                                                        scaleTensor,
                                                                        dxGpu,
                                                                        dscaleGpu,
                                                                        dbiasGpu,
                                                                        LAYERNORM_DEFAULT_EPSILON,
                                                                        &meanTensor,
                                                                        &rstdTensor,
                                                                        2);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<float>());
    assertAllClose(dbiasCpu, dbiasGpu, getTolerance<float>());
}

// --- Test suite instantiations ---

using TestGpuLayernormBwdRef4DFp32 = LayernormBwdTestSuite<float, float, float, double>;
using TestGpuLayernormBwdRef4DFp16 = LayernormBwdTestSuite<half, half, half, double>;
using TestGpuLayernormBwdRef4DBfp16 = LayernormBwdTestSuite<bfloat16, bfloat16, bfloat16, double>;
using TestGpuLayernormBwdRef5DFp32 = LayernormBwdTestSuite<float, float, float, double>;
using TestGpuLayernormBwdRef5DFp16 = LayernormBwdTestSuite<half, half, half, double>;
using TestGpuLayernormBwdRef5DBfp16 = LayernormBwdTestSuite<bfloat16, bfloat16, bfloat16, double>;

TEST_P(TestGpuLayernormBwdRef4DFp32, MatchesCpuRef)
{
    this->runLayernormBwdTest();
}
TEST_P(TestGpuLayernormBwdRef4DFp16, MatchesCpuRef)
{
    this->runLayernormBwdTest();
}
TEST_P(TestGpuLayernormBwdRef4DBfp16, MatchesCpuRef)
{
    this->runLayernormBwdTest();
}
TEST_P(TestGpuLayernormBwdRef5DFp32, MatchesCpuRef)
{
    this->runLayernormBwdTest();
}
TEST_P(TestGpuLayernormBwdRef5DFp16, MatchesCpuRef)
{
    this->runLayernormBwdTest();
}
TEST_P(TestGpuLayernormBwdRef5DBfp16, MatchesCpuRef)
{
    this->runLayernormBwdTest();
}

// ============================================================================
// 4D (NCHW/NHWC) tests
// ============================================================================

// --- Quick tests ---

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuLayernormBwdRef4DFp32,
                         ::testing::ValuesIn(getLayernormSmall4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuLayernormBwdRef4DFp16,
                         ::testing::ValuesIn(getLayernormSmall4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuLayernormBwdRef4DBfp16,
                         ::testing::ValuesIn(getLayernormSmall4DTestCases()));

INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuLayernormBwdRef4DFp32,
                         ::testing::ValuesIn(getLayernormMedium4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuLayernormBwdRef4DFp16,
                         ::testing::ValuesIn(getLayernormMedium4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuLayernormBwdRef4DBfp16,
                         ::testing::ValuesIn(getLayernormMedium4DTestCases()));

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuLayernormBwdRef4DFp32,
                         ::testing::ValuesIn(getLayernormLarge4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuLayernormBwdRef4DFp16,
                         ::testing::ValuesIn(getLayernormLarge4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuLayernormBwdRef4DBfp16,
                         ::testing::ValuesIn(getLayernormLarge4DTestCases()));

INSTANTIATE_TEST_SUITE_P(Full, TestGpuLayernormBwdRef4DFp32, ::testing::ValuesIn([]() {
                             auto v = getLayernormSmall4DTestCases();
                             auto m = getLayernormMedium4DTestCases();
                             auto l = getLayernormLarge4DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuLayernormBwdRef4DFp16, ::testing::ValuesIn([]() {
                             auto v = getLayernormSmall4DTestCases();
                             auto m = getLayernormMedium4DTestCases();
                             auto l = getLayernormLarge4DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuLayernormBwdRef4DBfp16, ::testing::ValuesIn([]() {
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
                         TestGpuLayernormBwdRef5DFp32,
                         ::testing::ValuesIn(getLayernormSmall5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuLayernormBwdRef5DFp16,
                         ::testing::ValuesIn(getLayernormSmall5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuLayernormBwdRef5DBfp16,
                         ::testing::ValuesIn(getLayernormSmall5DTestCases()));

INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuLayernormBwdRef5DFp32,
                         ::testing::ValuesIn(getLayernormMedium5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuLayernormBwdRef5DFp16,
                         ::testing::ValuesIn(getLayernormMedium5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuLayernormBwdRef5DBfp16,
                         ::testing::ValuesIn(getLayernormMedium5DTestCases()));

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuLayernormBwdRef5DFp32,
                         ::testing::ValuesIn(getLayernormLarge5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuLayernormBwdRef5DFp16,
                         ::testing::ValuesIn(getLayernormLarge5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuLayernormBwdRef5DBfp16,
                         ::testing::ValuesIn(getLayernormLarge5DTestCases()));

INSTANTIATE_TEST_SUITE_P(Full, TestGpuLayernormBwdRef5DFp32, ::testing::ValuesIn([]() {
                             auto v = getLayernormSmall5DTestCases();
                             auto m = getLayernormMedium5DTestCases();
                             auto l = getLayernormLarge5DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuLayernormBwdRef5DFp16, ::testing::ValuesIn([]() {
                             auto v = getLayernormSmall5DTestCases();
                             auto m = getLayernormMedium5DTestCases();
                             auto l = getLayernormLarge5DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuLayernormBwdRef5DBfp16, ::testing::ValuesIn([]() {
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
// --gtest_filter=*TestGpuLayernormBwdRefEdgeCaseValidation* flags.
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

TEST(TestGpuLayernormBwdRefEdgeCaseValidation, DISABLED_OuterSizeAtMaxBlocksMinusOneSucceeds)
{
    SKIP_IF_NO_DEVICES();
    const int64_t outerSize = getMaxOuterSizeForCurrentDevice() - 1;
    Tensor<float> dy({outerSize, 1, 1, 1});
    Tensor<float> x({outerSize, 1, 1, 1});
    Tensor<float> scale({1, 1, 1, 1});
    Tensor<float> dx({outerSize, 1, 1, 1});
    Tensor<float> dscale({1, 1, 1, 1});
    Tensor<float> dbias({1, 1, 1, 1});

    EXPECT_NO_THROW((GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
        dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 3)));
}

TEST(TestGpuLayernormBwdRefEdgeCaseValidation, DISABLED_OuterSizeAtMaxBlocksSucceeds)
{
    SKIP_IF_NO_DEVICES();
    const int64_t outerSize = getMaxOuterSizeForCurrentDevice();
    Tensor<float> dy({outerSize, 1, 1, 1});
    Tensor<float> x({outerSize, 1, 1, 1});
    Tensor<float> scale({1, 1, 1, 1});
    Tensor<float> dx({outerSize, 1, 1, 1});
    Tensor<float> dscale({1, 1, 1, 1});
    Tensor<float> dbias({1, 1, 1, 1});

    EXPECT_NO_THROW((GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
        dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 3)));
}

TEST(TestGpuLayernormBwdRefEdgeCaseValidation, DISABLED_OuterSizeAboveMaxBlocksThrows)
{
    SKIP_IF_NO_DEVICES();
    const int64_t outerSize = getMaxOuterSizeForCurrentDevice() + 1;
    Tensor<float> dy({outerSize, 1, 1, 1});
    Tensor<float> x({outerSize, 1, 1, 1});
    Tensor<float> scale({1, 1, 1, 1});
    Tensor<float> dx({outerSize, 1, 1, 1});
    Tensor<float> dscale({1, 1, 1, 1});
    Tensor<float> dbias({1, 1, 1, 1});

    EXPECT_THROW(
        (GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
            dy, x, scale, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 3)),
        std::runtime_error);
}

TEST(TestGpuLayernormBwdRefEdgeCaseValidation, DISABLED_BeyondInt32InnerSizeIfMemoryAllows)
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

    Tensor<float> dy({OUTER_SIZE, 1, 1, INNER_SIZE});
    Tensor<float> x({OUTER_SIZE, 1, 1, INNER_SIZE});
    Tensor<float> scale({1, 1, 1, INNER_SIZE});
    Tensor<float> dxCpu({OUTER_SIZE, 1, 1, INNER_SIZE});
    Tensor<float> dxGpu({OUTER_SIZE, 1, 1, INNER_SIZE});
    Tensor<float> dscaleCpu({1, 1, 1, INNER_SIZE});
    Tensor<float> dscaleGpu({1, 1, 1, INNER_SIZE});
    Tensor<float> dbiasCpu({1, 1, 1, INNER_SIZE});
    Tensor<float> dbiasGpu({1, 1, 1, INNER_SIZE});

    // Calculate the required memory for input, scale, and output tensors
    const size_t requiredBytes
        = (dy.elementCount() + x.elementCount() + scale.elementCount() + dxGpu.elementCount()
           + dscaleGpu.elementCount() + dbiasGpu.elementCount())
          * sizeof(float);
    if(requiredBytes > freeBytes)
    {
        GTEST_SKIP() << "Insufficient GPU memory for the test. Required: " << requiredBytes
                     << " bytes, Free: " << freeBytes << " bytes.";
    }

    const unsigned int seed = getGlobalTestSeed();
    dy.fillWithRandomValues(-1.0f, 1.0f, seed);
    x.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    scale.fillWithRandomValues(-1.0f, 1.0f, seed + 2);

    CpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
        dy, x, scale, dxCpu, dscaleCpu, dbiasCpu, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 3);
    GpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
        dy, x, scale, dxGpu, dscaleGpu, dbiasGpu, LAYERNORM_DEFAULT_EPSILON, nullptr, nullptr, 3);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<float>());
    assertAllClose(dbiasCpu, dbiasGpu, getTolerance<float>());
}

using TestGpuLayernormBwdRefEdgeCaseValidationFp32
    = LayernormBwdTestSuite<float, float, float, double>;

TEST_P(TestGpuLayernormBwdRefEdgeCaseValidationFp32, DISABLED_MatchesCpuRef)
{
    this->runLayernormBwdTest();
}

INSTANTIATE_TEST_SUITE_P(SkinnyModerate,
                         TestGpuLayernormBwdRefEdgeCaseValidationFp32,
                         ::testing::ValuesIn(getLayernormSkinnyModerateTestCases()));

INSTANTIATE_TEST_SUITE_P(PowerOfTwo,
                         TestGpuLayernormBwdRefEdgeCaseValidationFp32,
                         ::testing::ValuesIn(getLayernormPowerOfTwoTestCases()));

INSTANTIATE_TEST_SUITE_P(
    SkinnyInt32Scale, TestGpuLayernormBwdRefEdgeCaseValidationFp32, ::testing::ValuesIn([]() {
        return getLayernormSkinnyInt32ScaleTestCases(getMaxOuterSizeForCurrentDevice());
    }()));

INSTANTIATE_TEST_SUITE_P(InnerSizeInt32Boundary,
                         TestGpuLayernormBwdRefEdgeCaseValidationFp32,
                         ::testing::ValuesIn(getLayernormInnerSizeInt32BoundaryTestCases()));
