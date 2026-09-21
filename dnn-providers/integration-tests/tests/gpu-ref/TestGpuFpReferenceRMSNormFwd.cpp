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

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y, 1.0e-5, &invRms, &bias));
}

TEST(TestGpuRMSNormFwdRefValidation, AcceptsValidParamsNormalizeDimTwo4D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 1, 8, 8});
    Tensor<float> y({2, 4, 8, 8});
    Tensor<float> bias({1, 1, 8, 8});
    Tensor<double> invRms({2, 4, 1, 1});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y, 1.0e-5, &invRms, &bias));
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

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y, 1.0e-5, &invRms, &bias));
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

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y, 1.0e-5, &invRms),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormFwdRefValidation, ThrowsOnBiasRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 8});
    Tensor<float> bias({4, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(
                     x,
                     scale,
                     y,
                     1.0e-5,
                     static_cast<hipdnn_data_sdk::utilities::TensorBase<double>*>(nullptr),
                     &bias),
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

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(
                     x,
                     scale,
                     y,
                     1.0e-5,
                     static_cast<hipdnn_data_sdk::utilities::TensorBase<double>*>(nullptr),
                     &bias),
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
                     &invRms,
                     static_cast<hipdnn_data_sdk::utilities::TensorBase<float>*>(nullptr)),
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

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(
                     x,
                     scale,
                     y,
                     1.0e-5,
                     static_cast<hipdnn_data_sdk::utilities::TensorBase<double>*>(nullptr),
                     &bias),
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
                     &invRms,
                     static_cast<hipdnn_data_sdk::utilities::TensorBase<float>*>(nullptr)),
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
        static_cast<hipdnn_data_sdk::utilities::TensorBase<double>*>(nullptr),
        &biasTensor);

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
        xTensor, scaleTensor, yGpu, 1e-5, &invRmsGpu, nullptr);

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
        xTensor, scaleTensor, yGpu, 1e-5, &invRmsGpu, &biasTensor);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
    assertAllClose(invRmsCpu, invRmsGpu, getTolerance<double>());
}

// --- Test suite instantiations ---

using TestGpuRMSNormFwdRef4DFp32 = RMSNormFwdTestSuite<float>;
using TestGpuRMSNormFwdRef4DFp16 = RMSNormFwdTestSuite<half>;
using TestGpuRMSNormFwdRef4DBfp16 = RMSNormFwdTestSuite<bfloat16>;
using TestGpuRMSNormFwdRef5DFp32 = RMSNormFwdTestSuite<float>;
using TestGpuRMSNormFwdRef5DFp16 = RMSNormFwdTestSuite<half>;
using TestGpuRMSNormFwdRef5DBfp16 = RMSNormFwdTestSuite<bfloat16>;

TEST_P(TestGpuRMSNormFwdRef4DFp32, MatchesCpuRef)
{
    this->runRMSNormFwdTest();
}
TEST_P(TestGpuRMSNormFwdRef4DFp16, MatchesCpuRef)
{
    this->runRMSNormFwdTest();
}
TEST_P(TestGpuRMSNormFwdRef4DBfp16, MatchesCpuRef)
{
    this->runRMSNormFwdTest();
}
TEST_P(TestGpuRMSNormFwdRef5DFp32, MatchesCpuRef)
{
    this->runRMSNormFwdTest();
}
TEST_P(TestGpuRMSNormFwdRef5DFp16, MatchesCpuRef)
{
    this->runRMSNormFwdTest();
}
TEST_P(TestGpuRMSNormFwdRef5DBfp16, MatchesCpuRef)
{
    this->runRMSNormFwdTest();
}

// ============================================================================
// 4D (NCHW/NHWC) tests
// ============================================================================

// --- Quick tests ---

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuRMSNormFwdRef4DFp32,
                         ::testing::ValuesIn(getRMSnormSmall4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuRMSNormFwdRef4DFp16,
                         ::testing::ValuesIn(getRMSnormSmall4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuRMSNormFwdRef4DBfp16,
                         ::testing::ValuesIn(getRMSnormSmall4DTestCases()));

INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuRMSNormFwdRef4DFp32,
                         ::testing::ValuesIn(getRMSnormMedium4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuRMSNormFwdRef4DFp16,
                         ::testing::ValuesIn(getRMSnormMedium4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuRMSNormFwdRef4DBfp16,
                         ::testing::ValuesIn(getRMSnormMedium4DTestCases()));

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuRMSNormFwdRef4DFp32,
                         ::testing::ValuesIn(getRMSnormLarge4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuRMSNormFwdRef4DFp16,
                         ::testing::ValuesIn(getRMSnormLarge4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuRMSNormFwdRef4DBfp16,
                         ::testing::ValuesIn(getRMSnormLarge4DTestCases()));

INSTANTIATE_TEST_SUITE_P(Full, TestGpuRMSNormFwdRef4DFp32, ::testing::ValuesIn([]() {
                             auto v = getRMSnormSmall4DTestCases();
                             auto m = getRMSnormMedium4DTestCases();
                             auto l = getRMSnormLarge4DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuRMSNormFwdRef4DFp16, ::testing::ValuesIn([]() {
                             auto v = getRMSnormSmall4DTestCases();
                             auto m = getRMSnormMedium4DTestCases();
                             auto l = getRMSnormLarge4DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuRMSNormFwdRef4DBfp16, ::testing::ValuesIn([]() {
                             auto v = getRMSnormSmall4DTestCases();
                             auto m = getRMSnormMedium4DTestCases();
                             auto l = getRMSnormLarge4DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));

// ============================================================================
// 5D (NCDHW/NDHWC) shape tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuRMSNormFwdRef5DFp32,
                         ::testing::ValuesIn(getRMSnormSmall5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuRMSNormFwdRef5DFp16,
                         ::testing::ValuesIn(getRMSnormSmall5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuRMSNormFwdRef5DBfp16,
                         ::testing::ValuesIn(getRMSnormSmall5DTestCases()));

INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuRMSNormFwdRef5DFp32,
                         ::testing::ValuesIn(getRMSnormMedium5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuRMSNormFwdRef5DFp16,
                         ::testing::ValuesIn(getRMSnormMedium5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuRMSNormFwdRef5DBfp16,
                         ::testing::ValuesIn(getRMSnormMedium5DTestCases()));

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuRMSNormFwdRef5DFp32,
                         ::testing::ValuesIn(getRMSnormLarge5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuRMSNormFwdRef5DFp16,
                         ::testing::ValuesIn(getRMSnormLarge5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuRMSNormFwdRef5DBfp16,
                         ::testing::ValuesIn(getRMSnormLarge5DTestCases()));

INSTANTIATE_TEST_SUITE_P(Full, TestGpuRMSNormFwdRef5DFp32, ::testing::ValuesIn([]() {
                             auto v = getRMSnormSmall5DTestCases();
                             auto m = getRMSnormMedium5DTestCases();
                             auto l = getRMSnormLarge5DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuRMSNormFwdRef5DFp16, ::testing::ValuesIn([]() {
                             auto v = getRMSnormSmall5DTestCases();
                             auto m = getRMSnormMedium5DTestCases();
                             auto l = getRMSnormLarge5DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuRMSNormFwdRef5DBfp16, ::testing::ValuesIn([]() {
                             auto v = getRMSnormSmall5DTestCases();
                             auto m = getRMSnormMedium5DTestCases();
                             auto l = getRMSnormLarge5DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));

// ============================================================================
// Edge case tests with DISABLED_ prefix to avoid running in CI.
// Run the tests manually with --gtest_also_run_disabled_tests
// --gtest_filter=*TestGpuRMSNormFwdRefEdgeCaseValidation* flags.
// ============================================================================

TEST(TestGpuRMSNormFwdRefEdgeCaseValidation, DISABLED_OuterSizeAtMaxBlocksMinusOneSucceeds)
{
    SKIP_IF_NO_DEVICES();
    const int64_t outerSize
        = getMaxGridSizeForCurrentDevice<GpuFpReferenceRMSNorm::BLOCK_SIZE>() - 1;
    Tensor<float> x({outerSize, 1, 1, 1});
    Tensor<float> scale({1, 1, 1, 1});
    Tensor<float> y({outerSize, 1, 1, 1});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y));
}

TEST(TestGpuRMSNormFwdRefEdgeCaseValidation, DISABLED_OuterSizeAtMaxBlocksSucceeds)
{
    SKIP_IF_NO_DEVICES();
    const int64_t outerSize = getMaxGridSizeForCurrentDevice<GpuFpReferenceRMSNorm::BLOCK_SIZE>();
    Tensor<float> x({outerSize, 1, 1, 1});
    Tensor<float> scale({1, 1, 1, 1});
    Tensor<float> y({outerSize, 1, 1, 1});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y));
}

TEST(TestGpuRMSNormFwdRefEdgeCaseValidation, DISABLED_OuterSizeAboveMaxBlocksThrows)
{
    SKIP_IF_NO_DEVICES();
    const int64_t outerSize
        = getMaxGridSizeForCurrentDevice<GpuFpReferenceRMSNorm::BLOCK_SIZE>() + 1;
    Tensor<float> x({outerSize, 1, 1, 1});
    Tensor<float> scale({1, 1, 1, 1});
    Tensor<float> y({outerSize, 1, 1, 1});

    EXPECT_THROW(GpuFpReferenceRMSNorm::fprop<float>(x, scale, y), std::runtime_error);
}

TEST(TestGpuRMSNormFwdRefEdgeCaseValidation, DISABLED_BeyondInt32InnerSizeIfMemoryAllows)
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
    Tensor<float> yCpu({OUTER_SIZE, 1, 1, INNER_SIZE});
    Tensor<float> yGpu({OUTER_SIZE, 1, 1, INNER_SIZE});

    // Calculate the required memory for input, scale, and output tensors
    const size_t requiredBytes
        = (x.elementCount() + scale.elementCount() + yGpu.elementCount()) * sizeof(float);
    if(requiredBytes > freeBytes)
    {
        GTEST_SKIP() << "Insufficient GPU memory for the test. Required: " << requiredBytes
                     << " bytes, Free: " << freeBytes << " bytes.";
    }

    const unsigned int seed = getGlobalTestSeed();
    x.fillWithRandomValues(-1.0f, 1.0f, seed);
    scale.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceRMSNorm::forward<float, float, float, double>(x, scale, yCpu, 1e-5);
    GpuFpReferenceRMSNorm::fprop<float, float, float, double>(x, scale, yGpu, 1e-5);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

using TestGpuRMSNormFwdRefEdgeCaseValidationFp32 = RMSNormFwdTestSuite<float>;

TEST_P(TestGpuRMSNormFwdRefEdgeCaseValidationFp32, DISABLED_MatchesCpuRef)
{
    this->runRMSNormFwdTest();
}

INSTANTIATE_TEST_SUITE_P(SkinnyModerate,
                         TestGpuRMSNormFwdRefEdgeCaseValidationFp32,
                         ::testing::ValuesIn(getRMSnormSkinnyModerateTestCases()));

INSTANTIATE_TEST_SUITE_P(PowerOfTwo,
                         TestGpuRMSNormFwdRefEdgeCaseValidationFp32,
                         ::testing::ValuesIn(getRMSnormPowerOfTwoTestCases()));

INSTANTIATE_TEST_SUITE_P(
    SkinnyInt32Scale, TestGpuRMSNormFwdRefEdgeCaseValidationFp32, ::testing::ValuesIn([]() {
        return getRMSnormSkinnyInt32ScaleTestCases(
            getMaxGridSizeForCurrentDevice<GpuFpReferenceRMSNorm::BLOCK_SIZE>());
    }()));

INSTANTIATE_TEST_SUITE_P(InnerSizeInt32Boundary,
                         TestGpuRMSNormFwdRefEdgeCaseValidationFp32,
                         ::testing::ValuesIn(getRMSnormInnerSizeInt32BoundaryTestCases()));
