// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "GpuRMSNormFwdRefTestFixture.hpp"

// --- Valid configurations ---

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::rmsnorm;
using namespace hipdnn_gpu_ref;
using namespace gpu_rmsnorm_ref_test;
using namespace gpu_rmsnorm_fwd_ref_test;

TEST(TestGpuRMSNormFwdRefValidation, AcceptsValidParams3D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8});
    Tensor<float> scale({1, 4, 8});
    Tensor<float> y({2, 4, 8});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y));
}

TEST(TestGpuRMSNormFwdRefValidation, AcceptsValidParams4D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y));
}

TEST(TestGpuRMSNormFwdRefValidation, AcceptsValidParams5D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8, 8});
    Tensor<float> scale({1, 4, 8, 8, 8});
    Tensor<float> y({2, 4, 8, 8, 8});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y));
}

TEST(TestGpuRMSNormFwdRefValidation, AcceptsValidParamsChannelLastLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> scale({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> y({2, 4, 8, 8}, TensorLayout::NHWC);

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y));
}

TEST(TestGpuRMSNormFwdRefValidation, AcceptsValidParamsWithBiasAndInvRms)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8});
    Tensor<float> bias({1, 4, 8, 8});
    Tensor<double> invRms({2, 1, 1, 1});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y, 1.0e-5, &bias, &invRms));
}

TEST(TestGpuRMSNormFwdRefValidation, AcceptsValidParamsNormalizeDimTwo4D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 1, 8, 8});
    Tensor<float> y({2, 4, 8, 8});
    Tensor<float> bias({1, 1, 8, 8});
    Tensor<double> invRms({2, 4, 1, 1});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y, 1.0e-5, &bias, &invRms));
}

TEST(TestGpuRMSNormFwdRefValidation, AcceptsValidParamsNormalizeDimThree4D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 1, 1, 8});
    Tensor<float> y({2, 4, 8, 8});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y));
}

TEST(TestGpuRMSNormFwdRefValidation, AcceptsValidParamsNormalizeDimThree5D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8, 8});
    Tensor<float> scale({1, 1, 1, 8, 8});
    Tensor<float> y({2, 4, 8, 8, 8});
    Tensor<float> bias({1, 1, 1, 8, 8});
    Tensor<double> invRms({2, 4, 8, 1, 1});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y, 1.0e-5, &bias, &invRms));
}

// --- validateConsistentDimensions() throw paths ---

TEST(TestGpuRMSNormFwdRefValidation, ThrowsOnInputRankTooSmall)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8});
    Tensor<float> scale({1, 8});
    Tensor<float> y({4, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y), std::invalid_argument);
}

TEST(TestGpuRMSNormFwdRefValidation, ThrowsOnScaleRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({4, 8});
    Tensor<float> y({2, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y), std::invalid_argument);
}

TEST(TestGpuRMSNormFwdRefValidation, ThrowsOnOutputRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y), std::invalid_argument);
}

TEST(TestGpuRMSNormFwdRefValidation, ThrowsOnInvRmsRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8});
    Tensor<double> invRms({2, 1, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(
                     x,
                     scale,
                     y,
                     1.0e-5,
                     static_cast<hipdnn_data_sdk::utilities::TensorBase<float>*>(nullptr),
                     &invRms),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormFwdRefValidation, ThrowsOnBiasRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8});
    Tensor<float> bias({4, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y, 1.0e-5, &bias),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormFwdRefValidation, ThrowsOnInputOutputShapeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 4});

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y), std::invalid_argument);
}

TEST(TestGpuRMSNormFwdRefValidation, ThrowsOnScaleBiasShapeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8});
    Tensor<float> bias({1, 1, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y, 1.0e-5, &bias),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormFwdRefValidation, ThrowsOnAffineLeadingDimsNotOne)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({2, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y), std::invalid_argument);
}

TEST(TestGpuRMSNormFwdRefValidation, ThrowsOnInvRmsDimsNotDerivedFromInputAndScale)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8});
    Tensor<double> invRms({2, 1, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(
                     x,
                     scale,
                     y,
                     1.0e-5,
                     static_cast<hipdnn_data_sdk::utilities::TensorBase<float>*>(nullptr),
                     &invRms),
                 std::invalid_argument);
}

// --- validateConsistentLayouts() throw paths ---

TEST(TestGpuRMSNormFwdRefValidation, ThrowsOnInputRankNotSupportedByLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8, 8, 8});
    Tensor<float> scale({1, 4, 8, 8, 8, 8});
    Tensor<float> y({2, 4, 8, 8, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y), std::invalid_argument);
}

TEST(TestGpuRMSNormFwdRefValidation, ThrowsOnInputLayoutNeitherChannelFirstNorLast)
{
    SKIP_IF_NO_DEVICES();
    // Random strides that don't correspond to either channel-first or channel-last layout
    Tensor<float> x({2, 4, 8, 8}, std::vector<int64_t>{1, 2, 3, 4});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y), std::invalid_argument);
}

TEST(TestGpuRMSNormFwdRefValidation, ThrowsOnOutputLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8}, TensorLayout::NHWC);

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y), std::invalid_argument);
}

TEST(TestGpuRMSNormFwdRefValidation, ThrowsOnScaleLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> y({2, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y), std::invalid_argument);
}

TEST(TestGpuRMSNormFwdRefValidation, ThrowsOnBiasLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8});
    Tensor<float> bias({1, 4, 8, 8}, TensorLayout::NHWC);

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y, 1.0e-5, &bias),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormFwdRefValidation, ThrowsOnInvRmsLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 1, 8, 8});
    Tensor<float> y({2, 4, 8, 8});
    Tensor<double> invRms({2, 4, 1, 1}, TensorLayout::NHWC);

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(
                     x,
                     scale,
                     y,
                     1.0e-5,
                     static_cast<hipdnn_data_sdk::utilities::TensorBase<float>*>(nullptr),
                     &invRms),
                 std::invalid_argument);
}

// --- Mixed type tests ---

TEST(TestGpuRMSNormFwdRefMixedType, FloatInputHalfScale)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({2, 3, 4, 4});
    Tensor<half> scaleTensor({1, 3, 4, 4});
    Tensor<float> yCpu({2, 3, 4, 4});
    Tensor<float> yGpu({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    scaleTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 1);

    CpuFpReferenceRMSNorm::forward<float, half, float, double>(xTensor, scaleTensor, yCpu, 1e-5);

    GpuFpReferenceRMSNorm::fprop<float, half, float, double>(xTensor, scaleTensor, yGpu, 1e-5);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

TEST(TestGpuRMSNormFwdRefMixedType, HalfInputFloatScale)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> xTensor({2, 3, 4, 4});
    Tensor<float> scaleTensor({1, 3, 4, 4});
    Tensor<half> yCpu({2, 3, 4, 4});
    Tensor<half> yGpu({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed);
    scaleTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceRMSNorm::forward<half, float, half, double>(xTensor, scaleTensor, yCpu, 1e-5);

    GpuFpReferenceRMSNorm::fprop<half, float, half, double>(xTensor, scaleTensor, yGpu, 1e-5);

    assertAllClose(yCpu, yGpu, getTolerance<half>());
}

TEST(TestGpuRMSNormFwdRefMixedType, HalfInputHalfScale)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> xTensor({2, 3, 4, 4});
    Tensor<half> scaleTensor({1, 3, 4, 4});
    Tensor<half> yCpu({2, 3, 4, 4});
    Tensor<half> yGpu({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed);
    scaleTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 1);

    CpuFpReferenceRMSNorm::forward<half, half, half>(xTensor, scaleTensor, yCpu, 1e-5);
    GpuFpReferenceRMSNorm::fprop<half, half, half>(xTensor, scaleTensor, yGpu, 1e-5);

    assertAllClose(yCpu, yGpu, getTolerance<half>());
}

TEST(TestGpuRMSNormFwdRefMixedType, BfloatInputFloatOutput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<bfloat16> xTensor({2, 3, 4, 4});
    Tensor<bfloat16> scaleTensor({1, 3, 4, 4});
    Tensor<float> yCpu({2, 3, 4, 4});
    Tensor<float> yGpu({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed);
    scaleTensor.fillWithRandomValues(
        static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed + 1);

    CpuFpReferenceRMSNorm::forward<bfloat16, bfloat16, float, double>(
        xTensor, scaleTensor, yCpu, 1e-5);

    GpuFpReferenceRMSNorm::fprop<bfloat16, bfloat16, float, double>(
        xTensor, scaleTensor, yGpu, 1e-5);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

TEST(TestGpuRMSNormFwdRefMixedType, BfloatInputHalfScale)
{
    SKIP_IF_NO_DEVICES();

    Tensor<bfloat16> xTensor({2, 3, 4, 4});
    Tensor<half> scaleTensor({1, 3, 4, 4});
    Tensor<bfloat16> yCpu({2, 3, 4, 4});
    Tensor<bfloat16> yGpu({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed);
    scaleTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 1);

    CpuFpReferenceRMSNorm::forward<bfloat16, half, bfloat16, double>(
        xTensor, scaleTensor, yCpu, 1e-5);
    GpuFpReferenceRMSNorm::fprop<bfloat16, half, bfloat16, double>(
        xTensor, scaleTensor, yGpu, 1e-5);

    assertAllClose(yCpu, yGpu, getTolerance<bfloat16>());
}

// --- Optional argument tests ---

TEST(TestGpuRMSNormFwdRefOptionalArgs, WithBias)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({2, 3, 4, 4});
    Tensor<float> scaleTensor({1, 3, 4, 4});
    Tensor<float> biasTensor({1, 3, 4, 4});
    Tensor<float> yCpu({2, 3, 4, 4});
    Tensor<float> yGpu({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    scaleTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    biasTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 2);

    CpuFpReferenceRMSNorm::forward<float, float, float>(
        xTensor,
        scaleTensor,
        yCpu,
        1e-5,
        static_cast<hipdnn_data_sdk::utilities::TensorBase<double>*>(nullptr),
        &biasTensor);
    GpuFpReferenceRMSNorm::fprop<float, float, float>(
        xTensor,
        scaleTensor,
        yGpu,
        1e-5,
        &biasTensor,
        static_cast<hipdnn_data_sdk::utilities::TensorBase<double>*>(nullptr));

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

TEST(TestGpuRMSNormFwdRefOptionalArgs, WithInvRms)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({2, 3, 4, 4});
    Tensor<float> scaleTensor({1, 3, 4, 4});
    Tensor<float> yCpu({2, 3, 4, 4});
    Tensor<float> yGpu({2, 3, 4, 4});
    Tensor<double> invRmsCpu({2, 1, 1, 1});
    Tensor<double> invRmsGpu({2, 1, 1, 1});

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    scaleTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceRMSNorm::forward<float, float, float>(
        xTensor, scaleTensor, yCpu, 1e-5, &invRmsCpu, nullptr);
    GpuFpReferenceRMSNorm::fprop<float, float, float>(
        xTensor, scaleTensor, yGpu, 1e-5, nullptr, &invRmsGpu);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
    assertAllClose(invRmsCpu, invRmsGpu, getTolerance<double>());
}

// -- Channel-last layout tests ---

TEST(TestGpuRMSNormFwdRefChannelLast, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> xTensor({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> scaleTensor({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> yCpu({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> yGpu({2, 4, 8, 8}, TensorLayout::NHWC);

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    scaleTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceRMSNorm::forward<float, float, float>(xTensor, scaleTensor, yCpu, 1e-5);
    GpuFpReferenceRMSNorm::fprop<float, float, float>(xTensor, scaleTensor, yGpu, 1e-5);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

TEST(TestGpuRMSNormFwdRefChannelLast, MatchesCpuRefWithBiasAndInvRms)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> xTensor({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> scaleTensor({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> biasTensor({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> yCpu({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> yGpu({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<double> invRmsCpu({2, 1, 1, 1}, TensorLayout::NHWC);
    Tensor<double> invRmsGpu({2, 1, 1, 1}, TensorLayout::NHWC);

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    scaleTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    biasTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 2);

    CpuFpReferenceRMSNorm::forward<float, float, float>(
        xTensor, scaleTensor, yCpu, 1e-5, &invRmsCpu, &biasTensor);
    GpuFpReferenceRMSNorm::fprop<float, float, float>(
        xTensor, scaleTensor, yGpu, 1e-5, &biasTensor, &invRmsGpu);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
    assertAllClose(invRmsCpu, invRmsGpu, getTolerance<double>());
}

// --- Test suite instantiations ---

using TestGpuRMSNormFwdRefFp32 = RMSNormFwdTestSuite<float>;
using TestGpuRMSNormFwdRefFp16 = RMSNormFwdTestSuite<half>;
using TestGpuRMSNormFwdRefBfp16 = RMSNormFwdTestSuite<bfloat16>;

TEST_P(TestGpuRMSNormFwdRefFp32, MatchesCpuRef)
{
    this->runRMSNormFwdTest();
}
TEST_P(TestGpuRMSNormFwdRefFp16, MatchesCpuRef)
{
    this->runRMSNormFwdTest();
}
TEST_P(TestGpuRMSNormFwdRefBfp16, MatchesCpuRef)
{
    this->runRMSNormFwdTest();
}

// ============================================================================
// 4D (NCHW/NHWC) shape tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuRMSNormFwdRefFp32,
                         ::testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuRMSNormFwdRefFp16,
                         ::testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuRMSNormFwdRefBfp16,
                         ::testing::ValuesIn(getRMSnormTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuRMSNormFwdRefFp32,
                         ::testing::ValuesIn(getRMSnormFullTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuRMSNormFwdRefFp16,
                         ::testing::ValuesIn(getRMSnormFullTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuRMSNormFwdRefBfp16,
                         ::testing::ValuesIn(getRMSnormFullTestCases()));

// ============================================================================
// 5D (NCDHW/NDHWC) shape tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Smoke3d,
                         TestGpuRMSNormFwdRefFp32,
                         ::testing::ValuesIn(getRMSnorm3dTestCases()));
INSTANTIATE_TEST_SUITE_P(Smoke3d,
                         TestGpuRMSNormFwdRefFp16,
                         ::testing::ValuesIn(getRMSnorm3dTestCases()));
INSTANTIATE_TEST_SUITE_P(Smoke3d,
                         TestGpuRMSNormFwdRefBfp16,
                         ::testing::ValuesIn(getRMSnorm3dTestCases()));
