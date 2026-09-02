// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "GpuRMSNormBwdRefTestFixture.hpp"

// --- Valid configurations ---

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::rmsnorm;
using namespace hipdnn_gpu_ref;
using namespace gpu_rmsnorm_ref_test;
using namespace gpu_rmsnorm_bwd_ref_test;

TEST(TestGpuRMSNormBwdRefValidation, AcceptsValidParams3D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8});
    Tensor<float> x({2, 4, 8});
    Tensor<float> scale({1, 4, 8});
    Tensor<double> invRms({2, 1, 1});
    Tensor<float> dx({2, 4, 8});
    Tensor<float> dscale({1, 4, 8});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale));
}

TEST(TestGpuRMSNormBwdRefValidation, AcceptsValidParams4D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<double> invRms({2, 1, 1, 1});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale));
}

TEST(TestGpuRMSNormBwdRefValidation, AcceptsValidParams5D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8, 8});
    Tensor<float> x({2, 4, 8, 8, 8});
    Tensor<float> scale({1, 4, 8, 8, 8});
    Tensor<double> invRms({2, 1, 1, 1, 1});
    Tensor<float> dx({2, 4, 8, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8, 8});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale));
}

TEST(TestGpuRMSNormBwdRefValidation, AcceptsValidParamsChannelLastLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> x({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> scale({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<double> invRms({2, 1, 1, 1}, TensorLayout::NHWC);
    Tensor<float> dx({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dscale({1, 4, 8, 8}, TensorLayout::NHWC);

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale));
}

TEST(TestGpuRMSNormBwdRefValidation, AcceptsValidParamsWithBias)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<double> invRms({2, 1, 1, 1});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});
    Tensor<float> dbias({1, 4, 8, 8});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale, &dbias));
}

TEST(TestGpuRMSNormBwdRefValidation, AcceptsValidParamsNormalizeDimTwo4D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 1, 8, 8});
    Tensor<double> invRms({2, 4, 1, 1});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 1, 8, 8});
    Tensor<float> dbias({1, 1, 8, 8});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale, &dbias));
}

TEST(TestGpuRMSNormBwdRefValidation, AcceptsValidParamsNormalizeDimThree4D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 1, 1, 8});
    Tensor<double> invRms({2, 4, 8, 1});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 1, 1, 8});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale));
}

TEST(TestGpuRMSNormBwdRefValidation, AcceptsValidParamsNormalizeDimThree5D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8, 8});
    Tensor<float> x({2, 4, 8, 8, 8});
    Tensor<float> scale({1, 1, 1, 8, 8});
    Tensor<double> invRms({2, 4, 8, 1, 1});
    Tensor<float> dx({2, 4, 8, 8, 8});
    Tensor<float> dscale({1, 1, 1, 8, 8});
    Tensor<float> dbias({1, 1, 1, 8, 8});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale, &dbias));
}

// --- validateConsistentDimensions() throw paths ---

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnInputRankTooSmall)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({4, 8});
    Tensor<float> x({4, 8});
    Tensor<float> scale({1, 8});
    Tensor<double> invRms({4, 1});
    Tensor<float> dx({4, 8});
    Tensor<float> dscale({1, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnScaleRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({4, 8});
    Tensor<double> invRms({2, 1, 1, 1});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnGradOutputRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<double> invRms({2, 1, 1, 1});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnGradInputRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<double> invRms({2, 1, 1, 1});
    Tensor<float> dx({2, 4, 8});
    Tensor<float> dscale({1, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnInvRmsRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<double> invRms({2, 1, 8});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnDscaleRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<double> invRms({2, 1, 1, 1});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({4, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnDbiasRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<double> invRms({2, 1, 1, 1});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});
    Tensor<float> dbias({4, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale, &dbias),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnInputGradOutputShapeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 4});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<double> invRms({2, 1, 1, 1});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnInputGradInputShapeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<double> invRms({2, 1, 1, 1});
    Tensor<float> dx({2, 4, 8, 4});
    Tensor<float> dscale({1, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnScaleDscaleShapeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<double> invRms({2, 1, 1, 1});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 1, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnScaleDbiasShapeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<double> invRms({2, 1, 1, 1});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});
    Tensor<float> dbias({1, 1, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale, &dbias),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnAffineLeadingDimsNotOne)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({2, 4, 8, 8});
    Tensor<double> invRms({2, 1, 1, 1});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({2, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnInvRmsDimsNotDerivedFromInputAndScale)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<double> invRms({2, 1, 8, 8});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::invalid_argument);
}

// --- validateConsistentLayouts() throw paths ---

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnInputRankNotSupportedByLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8, 8, 8});
    Tensor<float> x({2, 4, 8, 8, 8, 8});
    Tensor<float> scale({1, 4, 8, 8, 8, 8});
    Tensor<double> invRms({2, 1, 1, 1, 1, 1});
    Tensor<float> dx({2, 4, 8, 8, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnInputLayoutNeitherChannelFirstNorLast)
{
    SKIP_IF_NO_DEVICES();
    // Random strides that don't correspond to either channel-first or channel-last layout
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8}, std::vector<int64_t>{1, 2, 3, 4});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<double> invRms({2, 1, 1, 1});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnGradOutputLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<double> invRms({2, 1, 1, 1});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnGradInputLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<double> invRms({2, 1, 1, 1});
    Tensor<float> dx({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dscale({1, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnScaleLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<double> invRms({2, 1, 1, 1});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnDscaleLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<double> invRms({2, 1, 1, 1});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8}, TensorLayout::NHWC);

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnDbiasLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 8, 8});
    Tensor<double> invRms({2, 1, 1, 1});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 4, 8, 8});
    Tensor<float> dbias({1, 4, 8, 8}, TensorLayout::NHWC);

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale, &dbias),
                 std::invalid_argument);
}

TEST(TestGpuRMSNormBwdRefValidation, ThrowsOnInvRmsLayoutInconsistentWithInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 8, 8});
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 1, 8, 8});
    Tensor<double> invRms({2, 4, 1, 1}, TensorLayout::NHWC);
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<float> dscale({1, 1, 8, 8});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::invalid_argument);
}

// --- Mixed type tests ---

TEST(TestGpuRMSNormBwdRefMixedType, FloatInputHalfScale)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> dyTensor({2, 3, 4, 4});
    Tensor<float> xTensor({2, 3, 4, 4});
    Tensor<half> scaleTensor({1, 3, 4, 4});
    Tensor<double> invRmsTensor({2, 1, 1, 1});
    Tensor<float> dxCpu({2, 3, 4, 4});
    Tensor<float> dxGpu({2, 3, 4, 4});
    Tensor<half> dscaleCpu({1, 3, 4, 4});
    Tensor<half> dscaleGpu({1, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    scaleTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 2);
    invRmsTensor.fillWithRandomValues(0.1, 2.0, seed + 3);

    CpuFpReferenceRMSNorm::backward<float, float, half, float, double>(
        dyTensor, xTensor, scaleTensor, invRmsTensor, dxCpu, dscaleCpu);

    GpuFpReferenceRMSNorm::bprop<float, float, half, float, double>(
        dyTensor, xTensor, scaleTensor, invRmsTensor, dxGpu, dscaleGpu);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<half>());
}

TEST(TestGpuRMSNormBwdRefMixedType, HalfInputFloatScale)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> dyTensor({2, 3, 4, 4});
    Tensor<half> xTensor({2, 3, 4, 4});
    Tensor<float> scaleTensor({1, 3, 4, 4});
    Tensor<double> invRmsTensor({2, 1, 1, 1});
    Tensor<half> dxCpu({2, 3, 4, 4});
    Tensor<half> dxGpu({2, 3, 4, 4});
    Tensor<float> dscaleCpu({1, 3, 4, 4});
    Tensor<float> dscaleGpu({1, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    dyTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed);
    xTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 1);
    scaleTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 2);
    invRmsTensor.fillWithRandomValues(0.1, 2.0, seed + 3);

    CpuFpReferenceRMSNorm::backward<half, half, float, half, double>(
        dyTensor, xTensor, scaleTensor, invRmsTensor, dxCpu, dscaleCpu);

    GpuFpReferenceRMSNorm::bprop<half, half, float, half, double>(
        dyTensor, xTensor, scaleTensor, invRmsTensor, dxGpu, dscaleGpu);

    assertAllClose(dxCpu, dxGpu, getTolerance<half>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<float>());
}

TEST(TestGpuRMSNormBwdRefMixedType, HalfInputHalfScale)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> dyTensor({2, 3, 4, 4});
    Tensor<half> xTensor({2, 3, 4, 4});
    Tensor<half> scaleTensor({1, 3, 4, 4});
    Tensor<double> invRmsTensor({2, 1, 1, 1});
    Tensor<half> dxCpu({2, 3, 4, 4});
    Tensor<half> dxGpu({2, 3, 4, 4});
    Tensor<half> dscaleCpu({1, 3, 4, 4});
    Tensor<half> dscaleGpu({1, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    dyTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed);
    xTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 1);
    scaleTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 2);
    invRmsTensor.fillWithRandomValues(0.1, 2.0, seed + 3);

    CpuFpReferenceRMSNorm::backward<half>(
        dyTensor, xTensor, scaleTensor, invRmsTensor, dxCpu, dscaleCpu);
    GpuFpReferenceRMSNorm::bprop<half>(
        dyTensor, xTensor, scaleTensor, invRmsTensor, dxGpu, dscaleGpu);

    assertAllClose(dxCpu, dxGpu, getTolerance<half>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<half>());
}

TEST(TestGpuRMSNormBwdRefMixedType, BfloatInputFloatGradInput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<bfloat16> dyTensor({2, 3, 4, 4});
    Tensor<bfloat16> xTensor({2, 3, 4, 4});
    Tensor<bfloat16> scaleTensor({1, 3, 4, 4});
    Tensor<double> invRmsTensor({2, 1, 1, 1});
    Tensor<float> dxCpu({2, 3, 4, 4});
    Tensor<float> dxGpu({2, 3, 4, 4});
    Tensor<bfloat16> dscaleCpu({1, 3, 4, 4});
    Tensor<bfloat16> dscaleGpu({1, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    dyTensor.fillWithRandomValues(static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed);
    xTensor.fillWithRandomValues(
        static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed + 1);
    scaleTensor.fillWithRandomValues(
        static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed + 2);
    invRmsTensor.fillWithRandomValues(0.1, 2.0, seed + 3);

    CpuFpReferenceRMSNorm::backward<bfloat16, bfloat16, bfloat16, float, double>(
        dyTensor, xTensor, scaleTensor, invRmsTensor, dxCpu, dscaleCpu);

    GpuFpReferenceRMSNorm::bprop<bfloat16, bfloat16, bfloat16, float, double>(
        dyTensor, xTensor, scaleTensor, invRmsTensor, dxGpu, dscaleGpu);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<bfloat16>());
}

TEST(TestGpuRMSNormBwdRefMixedType, BfloatInputHalfScale)
{
    SKIP_IF_NO_DEVICES();

    Tensor<bfloat16> dyTensor({2, 3, 4, 4});
    Tensor<bfloat16> xTensor({2, 3, 4, 4});
    Tensor<half> scaleTensor({1, 3, 4, 4});
    Tensor<double> invRmsTensor({2, 1, 1, 1});
    Tensor<bfloat16> dxCpu({2, 3, 4, 4});
    Tensor<bfloat16> dxGpu({2, 3, 4, 4});
    Tensor<half> dscaleCpu({1, 3, 4, 4});
    Tensor<half> dscaleGpu({1, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    dyTensor.fillWithRandomValues(static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed);
    xTensor.fillWithRandomValues(
        static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed + 1);
    scaleTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 2);
    invRmsTensor.fillWithRandomValues(0.1, 2.0, seed + 3);

    CpuFpReferenceRMSNorm::backward<bfloat16, bfloat16, half, bfloat16, double>(
        dyTensor, xTensor, scaleTensor, invRmsTensor, dxCpu, dscaleCpu);
    GpuFpReferenceRMSNorm::bprop<bfloat16, bfloat16, half, bfloat16, double>(
        dyTensor, xTensor, scaleTensor, invRmsTensor, dxGpu, dscaleGpu);

    assertAllClose(dxCpu, dxGpu, getTolerance<bfloat16>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<half>());
}

// --- Optional argument tests ---

TEST(TestGpuRMSNormBwdRefOptionalArgs, WithBias)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> dyTensor({2, 3, 4, 4});
    Tensor<float> xTensor({2, 3, 4, 4});
    Tensor<float> scaleTensor({1, 3, 4, 4});
    Tensor<double> invRmsTensor({2, 1, 1, 1});
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
    invRmsTensor.fillWithRandomValues(0.1, 2.0, seed + 3);

    CpuFpReferenceRMSNorm::backward<float>(
        dyTensor, xTensor, scaleTensor, invRmsTensor, dxCpu, dscaleCpu, &dbiasCpu);
    GpuFpReferenceRMSNorm::bprop<float>(
        dyTensor, xTensor, scaleTensor, invRmsTensor, dxGpu, dscaleGpu, &dbiasGpu);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<float>());
    assertAllClose(dbiasCpu, dbiasGpu, getTolerance<float>());
}

TEST(TestGpuRMSNormBwdRefOptionalArgs, WithoutBias)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> dyTensor({2, 3, 4, 4});
    Tensor<float> xTensor({2, 3, 4, 4});
    Tensor<float> scaleTensor({1, 3, 4, 4});
    Tensor<double> invRmsTensor({2, 1, 1, 1});
    Tensor<float> dxCpu({2, 3, 4, 4});
    Tensor<float> dxGpu({2, 3, 4, 4});
    Tensor<float> dscaleCpu({1, 3, 4, 4});
    Tensor<float> dscaleGpu({1, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    scaleTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 2);
    invRmsTensor.fillWithRandomValues(0.1, 2.0, seed + 3);

    CpuFpReferenceRMSNorm::backward<float>(
        dyTensor, xTensor, scaleTensor, invRmsTensor, dxCpu, dscaleCpu);
    GpuFpReferenceRMSNorm::bprop<float>(
        dyTensor, xTensor, scaleTensor, invRmsTensor, dxGpu, dscaleGpu);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<float>());
}

// -- Channel-last layout tests ---

TEST(TestGpuRMSNormBwdRefChannelLast, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dyTensor({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> xTensor({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> scaleTensor({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<double> invRmsTensor({2, 1, 1, 1}, TensorLayout::NHWC);
    Tensor<float> dxCpu({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dxGpu({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dscaleCpu({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dscaleGpu({1, 4, 8, 8}, TensorLayout::NHWC);

    const unsigned int seed = getGlobalTestSeed();
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    scaleTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 2);
    invRmsTensor.fillWithRandomValues(0.1, 2.0, seed + 3);

    CpuFpReferenceRMSNorm::backward<float>(
        dyTensor, xTensor, scaleTensor, invRmsTensor, dxCpu, dscaleCpu);
    GpuFpReferenceRMSNorm::bprop<float>(
        dyTensor, xTensor, scaleTensor, invRmsTensor, dxGpu, dscaleGpu);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<float>());
}

TEST(TestGpuRMSNormBwdRefChannelLast, MatchesCpuRefWithBias)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dyTensor({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> xTensor({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> scaleTensor({1, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<double> invRmsTensor({2, 1, 1, 1}, TensorLayout::NHWC);
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
    invRmsTensor.fillWithRandomValues(0.1, 2.0, seed + 3);

    CpuFpReferenceRMSNorm::backward<float>(
        dyTensor, xTensor, scaleTensor, invRmsTensor, dxCpu, dscaleCpu, &dbiasCpu);
    GpuFpReferenceRMSNorm::bprop<float>(
        dyTensor, xTensor, scaleTensor, invRmsTensor, dxGpu, dscaleGpu, &dbiasGpu);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<float>());
    assertAllClose(dbiasCpu, dbiasGpu, getTolerance<float>());
}

// --- Test suite instantiations ---

using TestGpuRMSNormBwdRef4DFp32 = RMSNormBwdShapeSuite<float>;
using TestGpuRMSNormBwdRef4DFp16 = RMSNormBwdShapeSuite<half>;
using TestGpuRMSNormBwdRef4DBfp16 = RMSNormBwdShapeSuite<bfloat16>;
using TestGpuRMSNormBwdRef5DFp32 = RMSNormBwdShapeSuite<float>;
using TestGpuRMSNormBwdRef5DFp16 = RMSNormBwdShapeSuite<half>;
using TestGpuRMSNormBwdRef5DBfp16 = RMSNormBwdShapeSuite<bfloat16>;

TEST_P(TestGpuRMSNormBwdRef4DFp32, MatchesCpuRef)
{
    this->runRMSNormBwdShapeTest();
}
TEST_P(TestGpuRMSNormBwdRef4DFp16, MatchesCpuRef)
{
    this->runRMSNormBwdShapeTest();
}
TEST_P(TestGpuRMSNormBwdRef4DBfp16, MatchesCpuRef)
{
    this->runRMSNormBwdShapeTest();
}
TEST_P(TestGpuRMSNormBwdRef5DFp32, MatchesCpuRef)
{
    this->runRMSNormBwdShapeTest();
}
TEST_P(TestGpuRMSNormBwdRef5DFp16, MatchesCpuRef)
{
    this->runRMSNormBwdShapeTest();
}
TEST_P(TestGpuRMSNormBwdRef5DBfp16, MatchesCpuRef)
{
    this->runRMSNormBwdShapeTest();
}

// ============================================================================
// 4D (NCHW/NHWC) tests
// ============================================================================

// --- Quick tests ---

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuRMSNormBwdRef4DFp32,
                         ::testing::ValuesIn(getRMSnormSmall4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuRMSNormBwdRef4DFp16,
                         ::testing::ValuesIn(getRMSnormSmall4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuRMSNormBwdRef4DBfp16,
                         ::testing::ValuesIn(getRMSnormSmall4DTestCases()));

INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuRMSNormBwdRef4DFp32,
                         ::testing::ValuesIn(getRMSnormMedium4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuRMSNormBwdRef4DFp16,
                         ::testing::ValuesIn(getRMSnormMedium4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuRMSNormBwdRef4DBfp16,
                         ::testing::ValuesIn(getRMSnormMedium4DTestCases()));

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuRMSNormBwdRef4DFp32,
                         ::testing::ValuesIn(getRMSnormLarge4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuRMSNormBwdRef4DFp16,
                         ::testing::ValuesIn(getRMSnormLarge4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuRMSNormBwdRef4DBfp16,
                         ::testing::ValuesIn(getRMSnormLarge4DTestCases()));

INSTANTIATE_TEST_SUITE_P(Full, TestGpuRMSNormBwdRef4DFp32, ::testing::ValuesIn([]() {
                             auto v = getRMSnormSmall4DTestCases();
                             auto m = getRMSnormMedium4DTestCases();
                             auto l = getRMSnormLarge4DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuRMSNormBwdRef4DFp16, ::testing::ValuesIn([]() {
                             auto v = getRMSnormSmall4DTestCases();
                             auto m = getRMSnormMedium4DTestCases();
                             auto l = getRMSnormLarge4DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuRMSNormBwdRef4DBfp16, ::testing::ValuesIn([]() {
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
                         TestGpuRMSNormBwdRef5DFp32,
                         ::testing::ValuesIn(getRMSnormSmall5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuRMSNormBwdRef5DFp16,
                         ::testing::ValuesIn(getRMSnormSmall5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuRMSNormBwdRef5DBfp16,
                         ::testing::ValuesIn(getRMSnormSmall5DTestCases()));

INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuRMSNormBwdRef5DFp32,
                         ::testing::ValuesIn(getRMSnormMedium5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuRMSNormBwdRef5DFp16,
                         ::testing::ValuesIn(getRMSnormMedium5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuRMSNormBwdRef5DBfp16,
                         ::testing::ValuesIn(getRMSnormMedium5DTestCases()));

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuRMSNormBwdRef5DFp32,
                         ::testing::ValuesIn(getRMSnormLarge5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuRMSNormBwdRef5DFp16,
                         ::testing::ValuesIn(getRMSnormLarge5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuRMSNormBwdRef5DBfp16,
                         ::testing::ValuesIn(getRMSnormLarge5DTestCases()));

INSTANTIATE_TEST_SUITE_P(Full, TestGpuRMSNormBwdRef5DFp32, ::testing::ValuesIn([]() {
                             auto v = getRMSnormSmall5DTestCases();
                             auto m = getRMSnormMedium5DTestCases();
                             auto l = getRMSnormLarge5DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuRMSNormBwdRef5DFp16, ::testing::ValuesIn([]() {
                             auto v = getRMSnormSmall5DTestCases();
                             auto m = getRMSnormMedium5DTestCases();
                             auto l = getRMSnormLarge5DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuRMSNormBwdRef5DBfp16, ::testing::ValuesIn([]() {
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
// --gtest_filter=*TestGpuRMSNormBwdRefEdgeCaseValidation* flags.
// ============================================================================

// Dgrad: grid dim = outerSize * stride. Using stride = 1 (NCHW)
// so outerSize alone determines the grid dim.

TEST(TestGpuRMSNormBwdRefEdgeCaseValidation, DISABLED_DgradOuterSizeAtMaxBlocksMinusOneSucceeds)
{
    SKIP_IF_NO_DEVICES();
    const int64_t outerSize
        = getMaxGridSizeForCurrentDevice<GpuFpReferenceRMSNorm::BLOCK_SIZE>() - 1;

    Tensor<float> dy({outerSize, 1, 1, 1});
    Tensor<float> x({outerSize, 1, 1, 1});
    Tensor<float> scale({1, 1, 1, 1});
    Tensor<double> invRms({outerSize, 1, 1, 1});
    Tensor<float> dx({outerSize, 1, 1, 1});
    Tensor<float> dscale({1, 1, 1, 1});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale));
}

TEST(TestGpuRMSNormBwdRefEdgeCaseValidation, DISABLED_DgradOuterSizeAtMaxBlocksSucceeds)
{
    SKIP_IF_NO_DEVICES();
    const int64_t outerSize = getMaxGridSizeForCurrentDevice<GpuFpReferenceRMSNorm::BLOCK_SIZE>();

    Tensor<float> dy({outerSize, 1, 1, 1});
    Tensor<float> x({outerSize, 1, 1, 1});
    Tensor<float> scale({1, 1, 1, 1});
    Tensor<double> invRms({outerSize, 1, 1, 1});
    Tensor<float> dx({outerSize, 1, 1, 1});
    Tensor<float> dscale({1, 1, 1, 1});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale));
}

TEST(TestGpuRMSNormBwdRefEdgeCaseValidation, DISABLED_DgradOuterSizeAboveMaxBlocksThrows)
{
    SKIP_IF_NO_DEVICES();
    const int64_t outerSize
        = getMaxGridSizeForCurrentDevice<GpuFpReferenceRMSNorm::BLOCK_SIZE>() + 1;

    Tensor<float> dy({outerSize, 1, 1, 1});
    Tensor<float> x({outerSize, 1, 1, 1});
    Tensor<float> scale({1, 1, 1, 1});
    Tensor<double> invRms({outerSize, 1, 1, 1});
    Tensor<float> dx({outerSize, 1, 1, 1});
    Tensor<float> dscale({1, 1, 1, 1});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::runtime_error);
}

// Wgrad: grid dim = ceil(innerSize / BLOCK_SIZE). Picking innerSize
// such that ceil(innerSize / BLOCK_SIZE) = maxGridSize.

namespace
{

int64_t getInnerSizeForWgrad(int64_t numBlocks)
{
    return numBlocks * static_cast<int64_t>(GpuFpReferenceRMSNorm::BLOCK_SIZE);
}

} // namespace

TEST(TestGpuRMSNormBwdRefEdgeCaseValidation, DISABLED_WgradInnerSizeAtMaxBlocksMinusOneSucceeds)
{
    SKIP_IF_NO_DEVICES();
    const int64_t numBlocks
        = getMaxGridSizeForCurrentDevice<GpuFpReferenceRMSNorm::BLOCK_SIZE>() - 1;
    const int64_t innerSize = getInnerSizeForWgrad(numBlocks);

    Tensor<float> dy({1, 1, 1, innerSize});
    Tensor<float> x({1, 1, 1, innerSize});
    Tensor<float> scale({1, 1, 1, innerSize});
    Tensor<double> invRms({1, 1, 1, 1});
    Tensor<float> dx({1, 1, 1, innerSize});
    Tensor<float> dscale({1, 1, 1, innerSize});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale));
}

TEST(TestGpuRMSNormBwdRefEdgeCaseValidation, DISABLED_WgradInnerSizeAtMaxBlocksSucceeds)
{
    SKIP_IF_NO_DEVICES();
    const int64_t numBlocks = getMaxGridSizeForCurrentDevice<GpuFpReferenceRMSNorm::BLOCK_SIZE>();
    const int64_t innerSize = getInnerSizeForWgrad(numBlocks);

    Tensor<float> dy({1, 1, 1, innerSize});
    Tensor<float> x({1, 1, 1, innerSize});
    Tensor<float> scale({1, 1, 1, innerSize});
    Tensor<double> invRms({1, 1, 1, 1});
    Tensor<float> dx({1, 1, 1, innerSize});
    Tensor<float> dscale({1, 1, 1, innerSize});

    EXPECT_NO_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale));
}

TEST(TestGpuRMSNormBwdRefEdgeCaseValidation, DISABLED_WgradInnerSizeAboveMaxBlocksThrows)
{
    SKIP_IF_NO_DEVICES();
    const int64_t numBlocks
        = getMaxGridSizeForCurrentDevice<GpuFpReferenceRMSNorm::BLOCK_SIZE>() + 1;
    const int64_t innerSize = getInnerSizeForWgrad(numBlocks);

    Tensor<float> dy({1, 1, 1, innerSize});
    Tensor<float> x({1, 1, 1, innerSize});
    Tensor<float> scale({1, 1, 1, innerSize});
    Tensor<double> invRms({1, 1, 1, 1});
    Tensor<float> dx({1, 1, 1, innerSize});
    Tensor<float> dscale({1, 1, 1, innerSize});

    EXPECT_THROW(GpuFpReferenceRMSNorm::bprop<float>(dy, x, scale, invRms, dx, dscale),
                 std::runtime_error);
}

TEST(TestGpuRMSNormBwdRefEdgeCaseValidation, DISABLED_BeyondInt32InnerSizeIfMemoryAllows)
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
    Tensor<double> invRms({OUTER_SIZE, 1, 1, 1});
    Tensor<float> dxCpu({OUTER_SIZE, 1, 1, INNER_SIZE});
    Tensor<float> dxGpu({OUTER_SIZE, 1, 1, INNER_SIZE});
    Tensor<float> dscaleCpu({1, 1, 1, INNER_SIZE});
    Tensor<float> dscaleGpu({1, 1, 1, INNER_SIZE});

    const size_t requiredBytes
        = (dy.elementCount() + x.elementCount() + scale.elementCount() + dxCpu.elementCount()
           + dxGpu.elementCount() + dscaleCpu.elementCount() + dscaleGpu.elementCount())
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
    invRms.fillWithRandomValues(1e-5, 1.0, seed + 3);

    CpuFpReferenceRMSNorm::backward<float, float, float, float, double>(
        dy, x, scale, invRms, dxCpu, dscaleCpu, nullptr);
    GpuFpReferenceRMSNorm::bprop<float, float, float, float, double>(
        dy, x, scale, invRms, dxGpu, dscaleGpu, nullptr);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
    assertAllClose(dscaleCpu, dscaleGpu, getTolerance<float>());
}

using TestGpuRMSNormBwdRefEdgeCaseValidationFp32 = RMSNormBwdShapeSuite<float>;

TEST_P(TestGpuRMSNormBwdRefEdgeCaseValidationFp32, DISABLED_MatchesCpuRef)
{
    this->runRMSNormBwdShapeTest();
}

INSTANTIATE_TEST_SUITE_P(SkinnyModerate,
                         TestGpuRMSNormBwdRefEdgeCaseValidationFp32,
                         ::testing::ValuesIn(getRMSnormSkinnyModerateTestCases()));

INSTANTIATE_TEST_SUITE_P(PowerOfTwo,
                         TestGpuRMSNormBwdRefEdgeCaseValidationFp32,
                         ::testing::ValuesIn(getRMSnormPowerOfTwoTestCases()));

INSTANTIATE_TEST_SUITE_P(
    SkinnyInt32Scale, TestGpuRMSNormBwdRefEdgeCaseValidationFp32, ::testing::ValuesIn([]() {
        return getRMSnormSkinnyInt32ScaleTestCases(
            getMaxGridSizeForCurrentDevice<GpuFpReferenceRMSNorm::BLOCK_SIZE>());
    }()));

INSTANTIATE_TEST_SUITE_P(InnerSizeInt32Boundary,
                         TestGpuRMSNormBwdRefEdgeCaseValidationFp32,
                         ::testing::ValuesIn(getRMSnormInnerSizeInt32BoundaryTestCases()));
