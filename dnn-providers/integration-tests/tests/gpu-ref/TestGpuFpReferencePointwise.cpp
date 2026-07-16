// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GpuPointwiseRefTestFixture.hpp"
#include "PointwiseShapeCatalog.hpp"
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

using namespace gpu_pointwise_ref_test;

// --- Test mixed layouts ---

TEST(TestGpuPointwiseMixedLayouts, Unary)
{
    SKIP_IF_NO_DEVICES();

    // Input and output tensor have different layout
    Tensor<float> inputTensor({1, 3, 2, 2}, TensorLayout::NHWC);
    Tensor<float> outputCpuTensor({1, 3, 2, 2}, TensorLayout::NCHW);
    Tensor<float> outputGpuTensor({1, 3, 2, 2}, TensorLayout::NCHW);

    const unsigned int seed = getGlobalTestSeed();
    inputTensor.fillWithRandomValues(-1.0f, 1.0f, seed);

    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        PointwiseMode::ABS, outputCpuTensor, inputTensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        PointwiseMode::ABS, outputGpuTensor, inputTensor);

    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<float>());
}

TEST(TestGpuPointwiseMixedLayouts, Binary)
{
    SKIP_IF_NO_DEVICES();

    // Input tensors have different layouts
    Tensor<float> input0Tensor({1, 3, 2, 2}, TensorLayout::NHWC);
    Tensor<float> input1Tensor({1, 3, 2, 2}, TensorLayout::NCHW);
    Tensor<float> outputCpuTensor({1, 3, 2, 2}, TensorLayout::NHWC);
    Tensor<float> outputGpuTensor({1, 3, 2, 2}, TensorLayout::NHWC);

    const unsigned int seed = getGlobalTestSeed();
    input0Tensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    input1Tensor.fillWithRandomValues(-1.0f, 1.0f, seed);

    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        PointwiseMode::ADD, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        PointwiseMode::ADD, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<float>());
}

// --- Test mixed types ---

TEST(TestGpuPointwiseMixedTypes, UnaryUpcast)
{
    SKIP_IF_NO_DEVICES();

    // Input and output tensor have different types
    Tensor<half> inputTensor({1, 3, 2, 2});
    Tensor<float> outputCpuTensor({1, 3, 2, 2});
    Tensor<float> outputGpuTensor({1, 3, 2, 2});

    const unsigned int seed = getGlobalTestSeed();
    inputTensor.fillWithRandomValues(half{-1.0}, half{1.0}, seed);

    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        PointwiseMode::ABS, outputCpuTensor, inputTensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        PointwiseMode::ABS, outputGpuTensor, inputTensor);

    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<float>());
}

TEST(TestGpuPointwiseMixedTypes, UnaryDowncast)
{
    SKIP_IF_NO_DEVICES();

    // Input and output tensor have different types
    Tensor<half> inputTensor({1, 3, 2, 2});
    Tensor<bfloat16> outputCpuTensor({1, 3, 2, 2});
    Tensor<bfloat16> outputGpuTensor({1, 3, 2, 2});

    const unsigned int seed = getGlobalTestSeed();
    inputTensor.fillWithRandomValues(half{-1.0}, half{1.0}, seed);

    CpuReferencePointwiseImpl<bfloat16>::pointwiseCompute(
        PointwiseMode::ABS, outputCpuTensor, inputTensor);

    GpuReferencePointwise::pointwiseCompute<bfloat16>(
        PointwiseMode::ABS, outputGpuTensor, inputTensor);

    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<bfloat16>());
}

TEST(TestGpuPointwiseMixedTypes, BinaryMixedInputs)
{
    SKIP_IF_NO_DEVICES();

    // Input tensors have different types
    Tensor<half> input0Tensor({1, 3, 2, 2});
    Tensor<bfloat16> input1Tensor({1, 3, 2, 2});
    Tensor<float> outputCpuTensor({1, 3, 2, 2});
    Tensor<float> outputGpuTensor({1, 3, 2, 2});

    const unsigned int seed = getGlobalTestSeed();
    input0Tensor.fillWithRandomValues(half{-1.0f}, half{1.0f}, seed);
    input1Tensor.fillWithRandomValues(bfloat16{-1.0f}, bfloat16{1.0f}, seed);

    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        PointwiseMode::ADD, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        PointwiseMode::ADD, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<float>());
}

// --- Test 1D/2D/3D shapes ---

TEST(TestGpuPointwise1DShapes, Unary)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> inputTensor({3});
    Tensor<float> outputCpuTensor({3});
    Tensor<float> outputGpuTensor({3});

    const unsigned int seed = getGlobalTestSeed();
    inputTensor.fillWithRandomValues(1.0f, 1.0f, seed);

    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        PointwiseMode::ABS, outputCpuTensor, inputTensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        PointwiseMode::ABS, outputGpuTensor, inputTensor);

    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<float>());
}

TEST(TestGpuPointwise1DShapes, Binary)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> input0Tensor({3});
    Tensor<float> input1Tensor({3});
    Tensor<float> outputCpuTensor({3});
    Tensor<float> outputGpuTensor({3});

    const unsigned int seed = getGlobalTestSeed();
    input0Tensor.fillWithRandomValues(1.0f, 1.0f, seed);
    input1Tensor.fillWithRandomValues(1.0f, 1.0f, seed);

    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        PointwiseMode::ADD, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        PointwiseMode::ADD, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<float>());
}

TEST(TestGpuPointwise2DShapes, Unary)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> inputTensor({3, 2});
    Tensor<float> outputCpuTensor({3, 2});
    Tensor<float> outputGpuTensor({3, 2});

    const unsigned int seed = getGlobalTestSeed();
    inputTensor.fillWithRandomValues(1.0f, 1.0f, seed);

    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        PointwiseMode::ABS, outputCpuTensor, inputTensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        PointwiseMode::ABS, outputGpuTensor, inputTensor);

    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<float>());
}

TEST(TestGpuPointwise2DShapes, Binary)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> input0Tensor({3, 2});
    Tensor<float> input1Tensor({3, 2});
    Tensor<float> outputCpuTensor({3, 2});
    Tensor<float> outputGpuTensor({3, 2});

    const unsigned int seed = getGlobalTestSeed();
    input0Tensor.fillWithRandomValues(1.0f, 1.0f, seed);
    input1Tensor.fillWithRandomValues(1.0f, 1.0f, seed);

    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        PointwiseMode::ADD, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        PointwiseMode::ADD, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<float>());
}

TEST(TestGpuPointwise3DShapes, Unary)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> inputTensor({3, 2, 4});
    Tensor<float> outputCpuTensor({3, 2, 4});
    Tensor<float> outputGpuTensor({3, 2, 4});

    const unsigned int seed = getGlobalTestSeed();
    inputTensor.fillWithRandomValues(1.0f, 1.0f, seed);

    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        PointwiseMode::ABS, outputCpuTensor, inputTensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        PointwiseMode::ABS, outputGpuTensor, inputTensor);

    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<float>());
}

TEST(TestGpuPointwise3DShapes, Binary)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> input0Tensor({3, 2, 4});
    Tensor<float> input1Tensor({3, 2, 4});
    Tensor<float> outputCpuTensor({3, 2, 4});
    Tensor<float> outputGpuTensor({3, 2, 4});

    const unsigned int seed = getGlobalTestSeed();
    input0Tensor.fillWithRandomValues(1.0f, 1.0f, seed);
    input1Tensor.fillWithRandomValues(1.0f, 1.0f, seed);

    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        PointwiseMode::ADD, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        PointwiseMode::ADD, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<float>());
}

// --- Test validation ---

TEST(TestGpuPointwiseValidation, UnaryNotBroadcastable)
{
    SKIP_IF_NO_DEVICES();

    // Not broacastable if output tensor has less dimensions than input
    {
        Tensor<float> inputTensor({1, 3, 2, 4});
        Tensor<float> outputTensor({3, 2, 4});
        EXPECT_THROW(
            GpuReferencePointwise::pointwiseCompute(PointwiseMode::ABS, outputTensor, inputTensor),
            std::invalid_argument);
    }

    // Not broadcastable if dims mismatch and input is not 1
    {
        Tensor<float> inputTensor({1, 3, 2, 2});
        Tensor<float> outputTensor({1, 3, 2, 4});
        EXPECT_THROW(
            GpuReferencePointwise::pointwiseCompute(PointwiseMode::ABS, outputTensor, inputTensor),
            std::invalid_argument);
    }
}

TEST(TestGpuPointwiseValidation, ExceedsMaxDim)
{
    SKIP_IF_NO_DEVICES();

    // Max dimension of pointwise is 5
    Tensor<float> inputTensor({3, 1, 2, 3, 2, 4});
    Tensor<float> outputTensor({3, 1, 2, 3, 2, 4});
    EXPECT_THROW(
        GpuReferencePointwise::pointwiseCompute(PointwiseMode::ABS, outputTensor, inputTensor),
        std::invalid_argument);
}

// --- Test broadcasting ---

TEST(TestGpuPointwiseBroadcast, 2D)
{
    // broadcast [3, 1] + [3,4] -> [3,4]
    Tensor<float> inputTensor({3, 1});
    Tensor<float> outputGpuTensor({3, 4});
    Tensor<float> outputCpuTensor({3, 4});

    const unsigned int seed = getGlobalTestSeed();
    inputTensor.fillWithRandomValues(-1.0f, 1.0f, seed);

    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        PointwiseMode::ABS, outputCpuTensor, inputTensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        PointwiseMode::ABS, outputGpuTensor, inputTensor);

    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<float>());
}

TEST(TestGpuPointwiseBroadcast, 2DImplicitLeading)
{
    // broadcast [4] + [3,4] -> [3,4]
    Tensor<float> input1Tensor({4});
    Tensor<float> input2Tensor({3, 4});
    Tensor<float> outputGpuTensor({3, 4});
    Tensor<float> outputCpuTensor({3, 4});

    const unsigned int seed = getGlobalTestSeed();
    input1Tensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    input2Tensor.fillWithRandomValues(-1.0f, 1.0f, seed);

    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        PointwiseMode::MUL, outputCpuTensor, input1Tensor, input2Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        PointwiseMode::MUL, outputGpuTensor, input1Tensor, input2Tensor);

    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<float>());
}

TEST(TestGpuPointwiseBroadcast, 3D)
{
    // broadcast [2,3,4] + [1,3,1] -> [2,3,4]
    Tensor<float> input1Tensor({2, 3, 4});
    Tensor<float> input2Tensor({1, 3, 1});
    Tensor<float> outputGpuTensor({2, 3, 4});
    Tensor<float> outputCpuTensor({2, 3, 4});

    const unsigned int seed = getGlobalTestSeed();
    input1Tensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    input2Tensor.fillWithRandomValues(-1.0f, 1.0f, seed);

    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        PointwiseMode::MUL, outputCpuTensor, input1Tensor, input2Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        PointwiseMode::MUL, outputGpuTensor, input1Tensor, input2Tensor);

    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<float>());
}

TEST(TestGpuPointwiseBroadcast, 3DImplicitLeading)
{
    // broadcast [2,3,4] + [3,1] -> [2,3,4]
    Tensor<float> input1Tensor({2, 3, 4});
    Tensor<float> input2Tensor({3, 1});
    Tensor<float> outputGpuTensor({2, 3, 4});
    Tensor<float> outputCpuTensor({2, 3, 4});

    const unsigned int seed = getGlobalTestSeed();
    input1Tensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    input2Tensor.fillWithRandomValues(-1.0f, 1.0f, seed);

    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        PointwiseMode::MUL, outputCpuTensor, input1Tensor, input2Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        PointwiseMode::MUL, outputGpuTensor, input1Tensor, input2Tensor);

    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<float>());
}

TEST(TestGpuPointwiseBroadcast, 4D)
{
    // broadcast [2,1,3,1] + [1,2,1,4] -> [2,2,3,4]
    Tensor<float> input1Tensor({2, 1, 3, 1});
    Tensor<float> input2Tensor({1, 2, 1, 4});
    Tensor<float> outputGpuTensor({2, 2, 3, 4});
    Tensor<float> outputCpuTensor({2, 2, 3, 4});

    const unsigned int seed = getGlobalTestSeed();
    input1Tensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    input2Tensor.fillWithRandomValues(-1.0f, 1.0f, seed);

    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        PointwiseMode::MUL, outputCpuTensor, input1Tensor, input2Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        PointwiseMode::MUL, outputGpuTensor, input1Tensor, input2Tensor);

    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<float>());
}

TEST(TestGpuPointwiseBroadcast, 5D)
{
    // broadcast [2,3,2,2,2] + [1,3,1,1,1] -> [2,3,2,2,2]
    Tensor<float> input1Tensor({2, 3, 2, 2, 2}, TensorLayout::NDHWC);
    Tensor<float> input2Tensor({1, 3, 1, 1, 1}, TensorLayout::NDHWC);
    Tensor<float> outputGpuTensor({2, 3, 2, 2, 2}, TensorLayout::NDHWC);
    Tensor<float> outputCpuTensor({2, 3, 2, 2, 2}, TensorLayout::NDHWC);

    const unsigned int seed = getGlobalTestSeed();
    input1Tensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    input2Tensor.fillWithRandomValues(-1.0f, 1.0f, seed);

    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        PointwiseMode::MUL, outputCpuTensor, input1Tensor, input2Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        PointwiseMode::MUL, outputGpuTensor, input1Tensor, input2Tensor);

    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<float>());
}

// --- Test activation parameters ---

TEST(TestGpuPointwiseRefSwishForward, WithBetaVal)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> inputTensor({1, 1, 4, 4});
    Tensor<float> outputCpuTensor({1, 1, 4, 4});
    Tensor<float> outputGpuTensor({1, 1, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    inputTensor.fillWithRandomValues(-1.0f, 1.0f, seed);

    const float beta = 2.0f;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        PointwiseMode::SWISH_FWD, outputCpuTensor, inputTensor, 0.f, 0.f, 0.f, beta);

    GpuReferencePointwise::pointwiseCompute<float>(
        PointwiseMode::SWISH_FWD, outputGpuTensor, inputTensor, 0.f, 0.f, 0.f, beta);

    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<float>());
}

TEST(TestGpuPointwiseRefRELuForward, WithVals)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> inputTensor({1, 2, 2, 2});
    Tensor<float> outputCpuTensor({1, 2, 2, 2});
    Tensor<float> outputGpuTensor({1, 2, 2, 2});

    // Fill with values that will test all three regions: below lower_clip, between clips, above upper_clip
    inputTensor.setHostValue(-5.0f, 0, 0, 0, 0); // below lower_clip
    inputTensor.setHostValue(-1.0f, 0, 0, 0, 1); // between clips
    inputTensor.setHostValue(2.0f, 0, 0, 1, 0); // between clips
    inputTensor.setHostValue(5.0f, 0, 0, 1, 1); // above upper_clip
    inputTensor.setHostValue(-2.0f, 0, 1, 0, 0); // at lower_clip
    inputTensor.setHostValue(0.0f, 0, 1, 0, 1); // between clips
    inputTensor.setHostValue(4.0f, 0, 1, 1, 0); // at upper_clip
    inputTensor.setHostValue(1.0f, 0, 1, 1, 1); // between clips

    const float lowerClip = -2.0f;
    const float upperClip = 4.0f;
    const float lowerSlope = 0.1f;

    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        PointwiseMode::SWISH_FWD, outputCpuTensor, inputTensor, lowerClip, upperClip, lowerSlope);

    GpuReferencePointwise::pointwiseCompute<float>(
        PointwiseMode::SWISH_FWD, outputGpuTensor, inputTensor, lowerClip, upperClip, lowerSlope);

    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<float>());
}

TEST(TestGpuPointwiseRefRELuBackward, WithVals)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> input0Tensor({1, 2, 2, 2});
    Tensor<float> input1Tensor({1, 2, 2, 2});
    Tensor<float> outputCpuTensor({1, 2, 2, 2});
    Tensor<float> outputGpuTensor({1, 2, 2, 2});

    // Fill forward input with values that will test all three regions: below lower_clip, between clips, above upper_clip
    input0Tensor.setHostValue(-5.0f, 0, 0, 0, 0); // below lower_clip
    input0Tensor.setHostValue(-1.0f, 0, 0, 0, 1); // between clips
    input0Tensor.setHostValue(2.0f, 0, 0, 1, 0); // between clips
    input0Tensor.setHostValue(5.0f, 0, 0, 1, 1); // above upper_clip
    input0Tensor.setHostValue(-2.0f, 0, 1, 0, 0); // at lower_clip
    input0Tensor.setHostValue(0.0f, 0, 1, 0, 1); // between clips
    input0Tensor.setHostValue(4.0f, 0, 1, 1, 0); // at upper_clip
    input0Tensor.setHostValue(1.0f, 0, 1, 1, 1); // between clips

    input1Tensor.setHostValue(2.0f, 0, 0, 0, 0);
    input1Tensor.setHostValue(1.5f, 0, 0, 0, 1);
    input1Tensor.setHostValue(3.0f, 0, 0, 1, 0);
    input1Tensor.setHostValue(1.0f, 0, 0, 1, 1);
    input1Tensor.setHostValue(2.5f, 0, 1, 0, 0);
    input1Tensor.setHostValue(4.0f, 0, 1, 0, 1);
    input1Tensor.setHostValue(1.5f, 0, 1, 1, 0);
    input1Tensor.setHostValue(3.0f, 0, 1, 1, 1);

    const float lowerClip = -2.f;
    const float upperClip = 4.f;
    const float lowerSlope = 0.1f;

    CpuReferencePointwiseImpl<float>::pointwiseCompute(PointwiseMode::RELU_BWD,
                                                       outputCpuTensor,
                                                       input0Tensor,
                                                       input1Tensor,
                                                       lowerClip,
                                                       upperClip,
                                                       lowerSlope);

    GpuReferencePointwise::pointwiseCompute<float>(PointwiseMode::RELU_BWD,
                                                   outputGpuTensor,
                                                   input0Tensor,
                                                   input1Tensor,
                                                   lowerClip,
                                                   upperClip,
                                                   lowerSlope);
    assertAllClose(outputCpuTensor, outputGpuTensor, pointwise::getTolerance<float>());
}

// --- Test suite instantiations ---

using TestGpuPointwiseUnaryRefFp164D = PointwiseTestSuite<half>;
using TestGpuPointwiseUnaryRefFp165D = PointwiseTestSuite<half>;
using TestGpuPointwiseBinaryRefFp164D = PointwiseTestSuite<half>;
using TestGpuPointwiseBinaryRefFp165D = PointwiseTestSuite<half>;
using TestGpuPointwiseUnaryRefBfp164D = PointwiseTestSuite<bfloat16>;
using TestGpuPointwiseUnaryRefBfp165D = PointwiseTestSuite<bfloat16>;
using TestGpuPointwiseBinaryRefBfp164D = PointwiseTestSuite<bfloat16>;
using TestGpuPointwiseBinaryRefBfp165D = PointwiseTestSuite<bfloat16>;
using TestGpuPointwiseUnaryRefFp324D = PointwiseTestSuite<float>;
using TestGpuPointwiseUnaryRefFp325D = PointwiseTestSuite<float>;
using TestGpuPointwiseBinaryRefFp324D = PointwiseTestSuite<float>;
using TestGpuPointwiseBinaryRefFp325D = PointwiseTestSuite<float>;

TEST_P(TestGpuPointwiseUnaryRefFp164D, MatchesCpuRef)
{
    this->runPointwiseUnaryTest();
}

TEST_P(TestGpuPointwiseUnaryRefFp165D, MatchesCpuRef)
{
    this->runPointwiseUnaryTest();
}

TEST_P(TestGpuPointwiseBinaryRefFp164D, MatchesCpuRef)
{
    this->runPointwiseBinaryTest();
}

TEST_P(TestGpuPointwiseBinaryRefFp165D, MatchesCpuRef)
{
    this->runPointwiseBinaryTest();
}

TEST_P(TestGpuPointwiseUnaryRefBfp164D, MatchesCpuRef)
{
    this->runPointwiseUnaryTest();
}

TEST_P(TestGpuPointwiseUnaryRefBfp165D, MatchesCpuRef)
{
    this->runPointwiseUnaryTest();
}

TEST_P(TestGpuPointwiseBinaryRefBfp164D, MatchesCpuRef)
{
    this->runPointwiseBinaryTest();
}

TEST_P(TestGpuPointwiseBinaryRefBfp165D, MatchesCpuRef)
{
    this->runPointwiseBinaryTest();
}

TEST_P(TestGpuPointwiseUnaryRefFp324D, MatchesCpuRef)
{
    this->runPointwiseUnaryTest();
}

TEST_P(TestGpuPointwiseUnaryRefFp325D, MatchesCpuRef)
{
    this->runPointwiseUnaryTest();
}

TEST_P(TestGpuPointwiseBinaryRefFp324D, MatchesCpuRef)
{
    this->runPointwiseBinaryTest();
}

TEST_P(TestGpuPointwiseBinaryRefFp325D, MatchesCpuRef)
{
    this->runPointwiseBinaryTest();
}

// ============================================================================
// 4D tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseUnaryRefFp324D,
                         ::testing::ValuesIn(getSmall4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseBinaryRefFp324D,
                         ::testing::ValuesIn(getSmall4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseUnaryRefFp164D,
                         ::testing::ValuesIn(getSmall4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseBinaryRefFp164D,
                         ::testing::ValuesIn(getSmall4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseUnaryRefBfp164D,
                         ::testing::ValuesIn(getSmall4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseBinaryRefBfp164D,
                         ::testing::ValuesIn(getSmall4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnaryRefFp324D,
                         ::testing::ValuesIn(getMedium4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseBinaryRefFp324D,
                         ::testing::ValuesIn(getMedium4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnaryRefFp164D,
                         ::testing::ValuesIn(getMedium4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseBinaryRefFp164D,
                         ::testing::ValuesIn(getMedium4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnaryRefBfp164D,
                         ::testing::ValuesIn(getMedium4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseBinaryRefBfp164D,
                         ::testing::ValuesIn(getMedium4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseUnaryRefFp324D,
                         ::testing::ValuesIn(getLarge4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseBinaryRefFp324D,
                         ::testing::ValuesIn(getLarge4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseUnaryRefFp164D,
                         ::testing::ValuesIn(getLarge4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseBinaryRefFp164D,
                         ::testing::ValuesIn(getLarge4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseUnaryRefBfp164D,
                         ::testing::ValuesIn(getLarge4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseBinaryRefBfp164D,
                         ::testing::ValuesIn(getLarge4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseUnaryRefFp324D, ::testing::ValuesIn([]() {
                             auto v = getSmall4dUnaryPointwiseCases();
                             auto m = getMedium4dUnaryPointwiseCases();
                             auto l = getLarge4dUnaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseBinaryRefFp324D, ::testing::ValuesIn([]() {
                             auto v = getSmall4dBinaryPointwiseCases();
                             auto m = getMedium4dBinaryPointwiseCases();
                             auto l = getLarge4dBinaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseUnaryRefFp164D, ::testing::ValuesIn([]() {
                             auto v = getSmall4dUnaryPointwiseCases();
                             auto m = getMedium4dUnaryPointwiseCases();
                             auto l = getLarge4dUnaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseBinaryRefFp164D, ::testing::ValuesIn([]() {
                             auto v = getSmall4dBinaryPointwiseCases();
                             auto m = getMedium4dBinaryPointwiseCases();
                             auto l = getLarge4dBinaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseUnaryRefBfp164D, ::testing::ValuesIn([]() {
                             auto v = getSmall4dUnaryPointwiseCases();
                             auto m = getMedium4dUnaryPointwiseCases();
                             auto l = getLarge4dUnaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseBinaryRefBfp164D, ::testing::ValuesIn([]() {
                             auto v = getSmall4dBinaryPointwiseCases();
                             auto m = getMedium4dBinaryPointwiseCases();
                             auto l = getLarge4dBinaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
// ============================================================================
// 5D tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseUnaryRefFp325D,
                         ::testing::ValuesIn(getSmall5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseBinaryRefFp325D,
                         ::testing::ValuesIn(getSmall5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseUnaryRefFp165D,
                         ::testing::ValuesIn(getSmall5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseBinaryRefFp165D,
                         ::testing::ValuesIn(getSmall5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseUnaryRefBfp165D,
                         ::testing::ValuesIn(getSmall5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseBinaryRefBfp165D,
                         ::testing::ValuesIn(getSmall5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnaryRefFp325D,
                         ::testing::ValuesIn(getMedium5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseBinaryRefFp325D,
                         ::testing::ValuesIn(getMedium5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnaryRefFp165D,
                         ::testing::ValuesIn(getMedium5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseBinaryRefFp165D,
                         ::testing::ValuesIn(getMedium5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnaryRefBfp165D,
                         ::testing::ValuesIn(getMedium5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseBinaryRefBfp165D,
                         ::testing::ValuesIn(getMedium5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseUnaryRefFp325D,
                         ::testing::ValuesIn(getLarge5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseBinaryRefFp165D,
                         ::testing::ValuesIn(getLarge5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseUnaryRefFp165D,
                         ::testing::ValuesIn(getLarge5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseBinaryRefFp325D,
                         ::testing::ValuesIn(getLarge5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseUnaryRefBfp165D,
                         ::testing::ValuesIn(getLarge5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseBinaryRefBfp165D,
                         ::testing::ValuesIn(getLarge5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseUnaryRefFp325D, ::testing::ValuesIn([]() {
                             auto v = getSmall5dUnaryPointwiseCases();
                             auto m = getMedium5dUnaryPointwiseCases();
                             auto l = getLarge5dUnaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseBinaryRefFp325D, ::testing::ValuesIn([]() {
                             auto v = getSmall5dBinaryPointwiseCases();
                             auto m = getMedium5dBinaryPointwiseCases();
                             auto l = getLarge5dBinaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseUnaryRefFp165D, ::testing::ValuesIn([]() {
                             auto v = getSmall5dUnaryPointwiseCases();
                             auto m = getMedium5dUnaryPointwiseCases();
                             auto l = getLarge5dUnaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseBinaryRefFp165D, ::testing::ValuesIn([]() {
                             auto v = getSmall5dBinaryPointwiseCases();
                             auto m = getMedium5dBinaryPointwiseCases();
                             auto l = getLarge5dBinaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseUnaryRefBfp165D, ::testing::ValuesIn([]() {
                             auto v = getSmall5dUnaryPointwiseCases();
                             auto m = getMedium5dUnaryPointwiseCases();
                             auto l = getLarge5dUnaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseBinaryRefBfp165D, ::testing::ValuesIn([]() {
                             auto v = getSmall5dBinaryPointwiseCases();
                             auto m = getMedium5dBinaryPointwiseCases();
                             auto l = getLarge5dBinaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
