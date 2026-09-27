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
    const float fillRange = 1.0f;
    inputTensor.fillWithRandomValues(-fillRange, fillRange, seed);

    auto operation = PointwiseMode::ABS;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(operation, outputCpuTensor, inputTensor);

    GpuReferencePointwise::pointwiseCompute<float>(operation, outputGpuTensor, inputTensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
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
    const float fillRange = 1.0f;
    input0Tensor.fillWithRandomValues(-fillRange, fillRange, seed);
    input1Tensor.fillWithRandomValues(-fillRange, fillRange, seed);

    auto operation = PointwiseMode::ADD;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
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
    const float fillRange = 1.0f;
    inputTensor.fillWithRandomValues(half{-fillRange}, half{fillRange}, seed);

    auto operation = PointwiseMode::ABS;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(operation, outputCpuTensor, inputTensor);

    GpuReferencePointwise::pointwiseCompute<float>(operation, outputGpuTensor, inputTensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float, half>(operation, fillRange));
}

TEST(TestGpuPointwiseMixedTypes, UnaryDowncast)
{
    SKIP_IF_NO_DEVICES();

    // Input and output tensor have different types
    Tensor<half> inputTensor({1, 3, 2, 2});
    Tensor<bfloat16> outputCpuTensor({1, 3, 2, 2});
    Tensor<bfloat16> outputGpuTensor({1, 3, 2, 2});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    inputTensor.fillWithRandomValues(half{-fillRange}, half{fillRange}, seed);

    auto operation = PointwiseMode::ABS;
    CpuReferencePointwiseImpl<bfloat16>::pointwiseCompute(operation, outputCpuTensor, inputTensor);

    GpuReferencePointwise::pointwiseCompute<bfloat16>(operation, outputGpuTensor, inputTensor);

    assertAllClose(outputCpuTensor,
                   outputGpuTensor,
                   getDynamicTolerance<bfloat16, half>(operation, fillRange));
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
    const float fillRange = 1.0f;
    input0Tensor.fillWithRandomValues(half{-fillRange}, half{fillRange}, seed);
    input1Tensor.fillWithRandomValues(bfloat16{-fillRange}, bfloat16{fillRange}, seed);

    auto operation = PointwiseMode::ADD;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float, half>(operation, fillRange));
}

// --- Test 1D/2D/3D shapes ---

TEST(TestGpuPointwise1DShapes, Unary)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> inputTensor({3});
    Tensor<float> outputCpuTensor({3});
    Tensor<float> outputGpuTensor({3});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    inputTensor.fillWithRandomValues(-fillRange, fillRange, seed);

    auto operation = PointwiseMode::ABS;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(operation, outputCpuTensor, inputTensor);

    GpuReferencePointwise::pointwiseCompute<float>(operation, outputGpuTensor, inputTensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

TEST(TestGpuPointwise1DShapes, Binary)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> input0Tensor({3});
    Tensor<float> input1Tensor({3});
    Tensor<float> outputCpuTensor({3});
    Tensor<float> outputGpuTensor({3});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    input0Tensor.fillWithRandomValues(-fillRange, fillRange, seed);
    input1Tensor.fillWithRandomValues(-fillRange, fillRange, seed);

    auto operation = PointwiseMode::ADD;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

TEST(TestGpuPointwise2DShapes, Unary)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> inputTensor({3, 2});
    Tensor<float> outputCpuTensor({3, 2});
    Tensor<float> outputGpuTensor({3, 2});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    inputTensor.fillWithRandomValues(-fillRange, fillRange, seed);

    auto operation = PointwiseMode::ABS;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(operation, outputCpuTensor, inputTensor);

    GpuReferencePointwise::pointwiseCompute<float>(operation, outputGpuTensor, inputTensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

TEST(TestGpuPointwise2DShapes, Binary)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> input0Tensor({3, 2});
    Tensor<float> input1Tensor({3, 2});
    Tensor<float> outputCpuTensor({3, 2});
    Tensor<float> outputGpuTensor({3, 2});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    input0Tensor.fillWithRandomValues(-fillRange, fillRange, seed);
    input1Tensor.fillWithRandomValues(-fillRange, fillRange, seed);

    auto operation = PointwiseMode::ADD;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

TEST(TestGpuPointwise3DShapes, Unary)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> inputTensor({3, 2, 4});
    Tensor<float> outputCpuTensor({3, 2, 4});
    Tensor<float> outputGpuTensor({3, 2, 4});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    inputTensor.fillWithRandomValues(-fillRange, fillRange, seed);

    auto operation = PointwiseMode::ABS;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(operation, outputCpuTensor, inputTensor);

    GpuReferencePointwise::pointwiseCompute<float>(operation, outputGpuTensor, inputTensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

TEST(TestGpuPointwise3DShapes, Binary)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> input0Tensor({3, 2, 4});
    Tensor<float> input1Tensor({3, 2, 4});
    Tensor<float> outputCpuTensor({3, 2, 4});
    Tensor<float> outputGpuTensor({3, 2, 4});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    input0Tensor.fillWithRandomValues(-fillRange, fillRange, seed);
    input1Tensor.fillWithRandomValues(-fillRange, fillRange, seed);

    auto operation = PointwiseMode::ADD;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

// Edge case tests with DISABLED_ prefix to avoid running in CI.
// Run the tests manually with --gtest_also_run_disabled_tests
// --gtest_filter=*ExceedsInt32MaxElements* flags.
TEST(TestGpuPointwise5DShapes, DISABLED_ExceedsInt32MaxElements)
{
    SKIP_IF_NO_DEVICES();
    // Test with 2,487,206,250 elements, which is greater than 2,147,483,647 INT32_MAX
    Tensor<float> inputTensor({255, 255, 255, 50, 3});
    Tensor<float> outputCpuTensor({255, 255, 255, 50, 3});
    Tensor<float> outputGpuTensor({255, 255, 255, 50, 3});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    inputTensor.fillWithRandomValues(-fillRange, fillRange, seed);

    auto operation = PointwiseMode::ABS;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(operation, outputCpuTensor, inputTensor);

    GpuReferencePointwise::pointwiseCompute<float>(operation, outputGpuTensor, inputTensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

// --- Test validation ---

TEST(TestGpuPointwiseValidation, UnaryNotBroadcastable)
{
    SKIP_IF_NO_DEVICES();

    auto operation = PointwiseMode::ABS;

    // Not broadcastable if output tensor has less dimensions than input
    {
        Tensor<float> inputTensor({1, 3, 2, 4});
        Tensor<float> outputTensor({3, 2, 4});
        EXPECT_THROW(GpuReferencePointwise::pointwiseCompute(operation, outputTensor, inputTensor),
                     std::invalid_argument);
    }

    // Not broadcastable if dims mismatch and input is not 1
    {
        Tensor<float> inputTensor({1, 3, 2, 2});
        Tensor<float> outputTensor({1, 3, 2, 4});
        EXPECT_THROW(GpuReferencePointwise::pointwiseCompute(operation, outputTensor, inputTensor),
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

TEST(TestGpuPointwiseValidation, InvalidUnaryOp)
{
    SKIP_IF_NO_DEVICES();

    // MUL is a binary operation rather than a unary operation
    Tensor<float> inputTensor({2, 3, 2, 4});
    Tensor<float> outputTensor({2, 3, 2, 4});
    EXPECT_THROW(
        GpuReferencePointwise::pointwiseCompute(PointwiseMode::MUL, outputTensor, inputTensor),
        std::invalid_argument);
}

TEST(TestGpuPointwiseValidation, InvalidBinaryOp)
{
    SKIP_IF_NO_DEVICES();

    // ABS is a unary operation rather than a binary operation
    Tensor<float> input1Tensor({2, 3, 2, 4});
    Tensor<float> input2Tensor({2, 3, 2, 4});
    Tensor<float> outputTensor({2, 3, 2, 4});
    EXPECT_THROW(GpuReferencePointwise::pointwiseCompute(
                     PointwiseMode::ABS, outputTensor, input1Tensor, input2Tensor),
                 std::invalid_argument);
}

// --- Test broadcasting ---

TEST(TestGpuPointwiseBroadcast, Broadcast2D)
{
    SKIP_IF_NO_DEVICES();

    // broadcast [3, 1] -> [3,4]
    Tensor<float> inputTensor({3, 1});
    Tensor<float> outputGpuTensor({3, 4});
    Tensor<float> outputCpuTensor({3, 4});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    inputTensor.fillWithRandomValues(-fillRange, fillRange, seed);

    auto operation = PointwiseMode::ABS;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(operation, outputCpuTensor, inputTensor);

    GpuReferencePointwise::pointwiseCompute<float>(operation, outputGpuTensor, inputTensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

TEST(TestGpuPointwiseBroadcast, Broadcast2DImplicitLeading)
{
    SKIP_IF_NO_DEVICES();

    // broadcast [4] + [3,4] -> [3,4]
    Tensor<float> input1Tensor({4});
    Tensor<float> input2Tensor({3, 4});
    Tensor<float> outputGpuTensor({3, 4});
    Tensor<float> outputCpuTensor({3, 4});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    input1Tensor.fillWithRandomValues(-fillRange, fillRange, seed);
    input2Tensor.fillWithRandomValues(-fillRange, fillRange, seed);

    auto operation = PointwiseMode::MUL;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        operation, outputCpuTensor, input1Tensor, input2Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input1Tensor, input2Tensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

TEST(TestGpuPointwiseBroadcast, Broadcast3D)
{
    SKIP_IF_NO_DEVICES();

    // broadcast [2,3,4] + [1,3,1] -> [2,3,4]
    Tensor<float> input1Tensor({2, 3, 4});
    Tensor<float> input2Tensor({1, 3, 1});
    Tensor<float> outputGpuTensor({2, 3, 4});
    Tensor<float> outputCpuTensor({2, 3, 4});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    input1Tensor.fillWithRandomValues(-fillRange, fillRange, seed);
    input2Tensor.fillWithRandomValues(-fillRange, fillRange, seed);

    auto operation = PointwiseMode::MUL;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        operation, outputCpuTensor, input1Tensor, input2Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input1Tensor, input2Tensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

TEST(TestGpuPointwiseBroadcast, Broadcast3DImplicitLeading)
{
    SKIP_IF_NO_DEVICES();

    // broadcast [2,3,4] + [3,1] -> [2,3,4]
    Tensor<float> input1Tensor({2, 3, 4});
    Tensor<float> input2Tensor({3, 1});
    Tensor<float> outputGpuTensor({2, 3, 4});
    Tensor<float> outputCpuTensor({2, 3, 4});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    input1Tensor.fillWithRandomValues(-fillRange, fillRange, seed);
    input2Tensor.fillWithRandomValues(-fillRange, fillRange, seed);

    auto operation = PointwiseMode::MUL;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        operation, outputCpuTensor, input1Tensor, input2Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input1Tensor, input2Tensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

TEST(TestGpuPointwiseBroadcast, Broadcast4D)
{
    SKIP_IF_NO_DEVICES();

    // broadcast [2,1,3,1] + [1,2,1,4] -> [2,2,3,4]
    Tensor<float> input1Tensor({2, 1, 3, 1});
    Tensor<float> input2Tensor({1, 2, 1, 4});
    Tensor<float> outputGpuTensor({2, 2, 3, 4});
    Tensor<float> outputCpuTensor({2, 2, 3, 4});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    input1Tensor.fillWithRandomValues(-fillRange, fillRange, seed);
    input2Tensor.fillWithRandomValues(-fillRange, fillRange, seed);

    auto operation = PointwiseMode::MUL;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        operation, outputCpuTensor, input1Tensor, input2Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input1Tensor, input2Tensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

TEST(TestGpuPointwiseBroadcast, Broadcast5D)
{
    SKIP_IF_NO_DEVICES();

    // broadcast [2,3,2,2,2] + [1,3,1,1,1] -> [2,3,2,2,2]
    Tensor<float> input1Tensor({2, 3, 2, 2, 2}, TensorLayout::NDHWC);
    Tensor<float> input2Tensor({1, 3, 1, 1, 1}, TensorLayout::NDHWC);
    Tensor<float> outputGpuTensor({2, 3, 2, 2, 2}, TensorLayout::NDHWC);
    Tensor<float> outputCpuTensor({2, 3, 2, 2, 2}, TensorLayout::NDHWC);

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    input1Tensor.fillWithRandomValues(-fillRange, fillRange, seed);
    input2Tensor.fillWithRandomValues(-fillRange, fillRange, seed);

    auto operation = PointwiseMode::MUL;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        operation, outputCpuTensor, input1Tensor, input2Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input1Tensor, input2Tensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

// --- Test activation parameters ---

TEST(TestGpuPointwiseRefSwishForward, WithBetaVal)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> inputTensor({1, 1, 4, 4});
    Tensor<float> outputCpuTensor({1, 1, 4, 4});
    Tensor<float> outputGpuTensor({1, 1, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    inputTensor.fillWithRandomValues(-fillRange, fillRange, seed);

    auto operation = PointwiseMode::SWISH_FWD;
    const float beta = 2.0f;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        operation, outputCpuTensor, inputTensor, 0.f, 0.f, 0.f, beta);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, inputTensor, 0.f, 0.f, 0.f, beta);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
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

    auto operation = PointwiseMode::RELU_FWD;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        operation, outputCpuTensor, inputTensor, lowerClip, upperClip, lowerSlope);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, inputTensor, lowerClip, upperClip, lowerSlope);

    assertAllClose(outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, 5.0f));
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

    auto operation = PointwiseMode::RELU_BWD;
    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor, lowerClip, upperClip, lowerSlope);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor, lowerClip, upperClip, lowerSlope);
    assertAllClose(outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, 5.0f));
}

// --- Test suite instantiations ---

using TestGpuPointwiseUnaryRef4DFp16 = PointwiseTestSuite<half>;
using TestGpuPointwiseUnaryRef5DFp16 = PointwiseTestSuite<half>;
using TestGpuPointwiseBinaryRef4DFp16 = PointwiseTestSuite<half>;
using TestGpuPointwiseBinaryRef5DFp16 = PointwiseTestSuite<half>;
using TestGpuPointwiseUnaryRef4DBfp16 = PointwiseTestSuite<bfloat16>;
using TestGpuPointwiseUnaryRef5DBfp16 = PointwiseTestSuite<bfloat16>;
using TestGpuPointwiseBinaryRef4DBfp16 = PointwiseTestSuite<bfloat16>;
using TestGpuPointwiseBinaryRef5DBfp16 = PointwiseTestSuite<bfloat16>;
using TestGpuPointwiseUnaryRef4DFp32 = PointwiseTestSuite<float>;
using TestGpuPointwiseUnaryRef5DFp32 = PointwiseTestSuite<float>;
using TestGpuPointwiseBinaryRef4DFp32 = PointwiseTestSuite<float>;
using TestGpuPointwiseBinaryRef5DFp32 = PointwiseTestSuite<float>;

TEST_P(TestGpuPointwiseUnaryRef4DFp16, MatchesCpuRef)
{
    this->runPointwiseUnaryTest();
}

TEST_P(TestGpuPointwiseUnaryRef5DFp16, MatchesCpuRef)
{
    this->runPointwiseUnaryTest();
}

TEST_P(TestGpuPointwiseBinaryRef4DFp16, MatchesCpuRef)
{
    this->runPointwiseBinaryTest();
}

TEST_P(TestGpuPointwiseBinaryRef5DFp16, MatchesCpuRef)
{
    this->runPointwiseBinaryTest();
}

TEST_P(TestGpuPointwiseUnaryRef4DBfp16, MatchesCpuRef)
{
    this->runPointwiseUnaryTest();
}

TEST_P(TestGpuPointwiseUnaryRef5DBfp16, MatchesCpuRef)
{
    this->runPointwiseUnaryTest();
}

TEST_P(TestGpuPointwiseBinaryRef4DBfp16, MatchesCpuRef)
{
    this->runPointwiseBinaryTest();
}

TEST_P(TestGpuPointwiseBinaryRef5DBfp16, MatchesCpuRef)
{
    this->runPointwiseBinaryTest();
}

TEST_P(TestGpuPointwiseUnaryRef4DFp32, MatchesCpuRef)
{
    this->runPointwiseUnaryTest();
}

TEST_P(TestGpuPointwiseUnaryRef5DFp32, MatchesCpuRef)
{
    this->runPointwiseUnaryTest();
}

TEST_P(TestGpuPointwiseBinaryRef4DFp32, MatchesCpuRef)
{
    this->runPointwiseBinaryTest();
}

TEST_P(TestGpuPointwiseBinaryRef5DFp32, MatchesCpuRef)
{
    this->runPointwiseBinaryTest();
}

// ============================================================================
// 4D tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseUnaryRef4DFp32,
                         ::testing::ValuesIn(getSmall4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseBinaryRef4DFp32,
                         ::testing::ValuesIn(getSmall4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseUnaryRef4DFp16,
                         ::testing::ValuesIn(getSmall4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseBinaryRef4DFp16,
                         ::testing::ValuesIn(getSmall4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseUnaryRef4DBfp16,
                         ::testing::ValuesIn(getSmall4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseBinaryRef4DBfp16,
                         ::testing::ValuesIn(getSmall4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnaryRef4DFp32,
                         ::testing::ValuesIn(getMedium4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseBinaryRef4DFp32,
                         ::testing::ValuesIn(getMedium4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnaryRef4DFp16,
                         ::testing::ValuesIn(getMedium4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseBinaryRef4DFp16,
                         ::testing::ValuesIn(getMedium4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnaryRef4DBfp16,
                         ::testing::ValuesIn(getMedium4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseBinaryRef4DBfp16,
                         ::testing::ValuesIn(getMedium4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseUnaryRef4DFp32,
                         ::testing::ValuesIn(getLarge4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseBinaryRef4DFp32,
                         ::testing::ValuesIn(getLarge4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseUnaryRef4DFp16,
                         ::testing::ValuesIn(getLarge4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseBinaryRef4DFp16,
                         ::testing::ValuesIn(getLarge4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseUnaryRef4DBfp16,
                         ::testing::ValuesIn(getLarge4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseBinaryRef4DBfp16,
                         ::testing::ValuesIn(getLarge4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseUnaryRef4DFp32, ::testing::ValuesIn([]() {
                             auto v = getSmall4dUnaryPointwiseCases();
                             auto m = getMedium4dUnaryPointwiseCases();
                             auto l = getLarge4dUnaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseBinaryRef4DFp32, ::testing::ValuesIn([]() {
                             auto v = getSmall4dBinaryPointwiseCases();
                             auto m = getMedium4dBinaryPointwiseCases();
                             auto l = getLarge4dBinaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseUnaryRef4DFp16, ::testing::ValuesIn([]() {
                             auto v = getSmall4dUnaryPointwiseCases();
                             auto m = getMedium4dUnaryPointwiseCases();
                             auto l = getLarge4dUnaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseBinaryRef4DFp16, ::testing::ValuesIn([]() {
                             auto v = getSmall4dBinaryPointwiseCases();
                             auto m = getMedium4dBinaryPointwiseCases();
                             auto l = getLarge4dBinaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseUnaryRef4DBfp16, ::testing::ValuesIn([]() {
                             auto v = getSmall4dUnaryPointwiseCases();
                             auto m = getMedium4dUnaryPointwiseCases();
                             auto l = getLarge4dUnaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseBinaryRef4DBfp16, ::testing::ValuesIn([]() {
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
                         TestGpuPointwiseUnaryRef5DFp32,
                         ::testing::ValuesIn(getSmall5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseBinaryRef5DFp32,
                         ::testing::ValuesIn(getSmall5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseUnaryRef5DFp16,
                         ::testing::ValuesIn(getSmall5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseBinaryRef5DFp16,
                         ::testing::ValuesIn(getSmall5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseUnaryRef5DBfp16,
                         ::testing::ValuesIn(getSmall5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseBinaryRef5DBfp16,
                         ::testing::ValuesIn(getSmall5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnaryRef5DFp32,
                         ::testing::ValuesIn(getMedium5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseBinaryRef5DFp32,
                         ::testing::ValuesIn(getMedium5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnaryRef5DFp16,
                         ::testing::ValuesIn(getMedium5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseBinaryRef5DFp16,
                         ::testing::ValuesIn(getMedium5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnaryRef5DBfp16,
                         ::testing::ValuesIn(getMedium5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseBinaryRef5DBfp16,
                         ::testing::ValuesIn(getMedium5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseUnaryRef5DFp32,
                         ::testing::ValuesIn(getLarge5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseBinaryRef5DFp32,
                         ::testing::ValuesIn(getLarge5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseUnaryRef5DFp16,
                         ::testing::ValuesIn(getLarge5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseBinaryRef5DFp16,
                         ::testing::ValuesIn(getLarge5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseUnaryRef5DBfp16,
                         ::testing::ValuesIn(getLarge5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseBinaryRef5DBfp16,
                         ::testing::ValuesIn(getLarge5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseUnaryRef5DFp32, ::testing::ValuesIn([]() {
                             auto v = getSmall5dUnaryPointwiseCases();
                             auto m = getMedium5dUnaryPointwiseCases();
                             auto l = getLarge5dUnaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseBinaryRef5DFp32, ::testing::ValuesIn([]() {
                             auto v = getSmall5dBinaryPointwiseCases();
                             auto m = getMedium5dBinaryPointwiseCases();
                             auto l = getLarge5dBinaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseUnaryRef5DFp16, ::testing::ValuesIn([]() {
                             auto v = getSmall5dUnaryPointwiseCases();
                             auto m = getMedium5dUnaryPointwiseCases();
                             auto l = getLarge5dUnaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseBinaryRef5DFp16, ::testing::ValuesIn([]() {
                             auto v = getSmall5dBinaryPointwiseCases();
                             auto m = getMedium5dBinaryPointwiseCases();
                             auto l = getLarge5dBinaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseUnaryRef5DBfp16, ::testing::ValuesIn([]() {
                             auto v = getSmall5dUnaryPointwiseCases();
                             auto m = getMedium5dUnaryPointwiseCases();
                             auto l = getLarge5dUnaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuPointwiseBinaryRef5DBfp16, ::testing::ValuesIn([]() {
                             auto v = getSmall5dBinaryPointwiseCases();
                             auto m = getMedium5dBinaryPointwiseCases();
                             auto l = getLarge5dBinaryPointwiseCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
