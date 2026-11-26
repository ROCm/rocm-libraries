// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GpuPointwiseRefTestFixture.hpp"
#include "PointwiseShapeCatalog.hpp"
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

using namespace gpu_pointwise_ref_test;

using HalfType = hipdnn_data_sdk::types::half;
using BFloat16Type = hipdnn_data_sdk::types::bfloat16;

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
    fillWithRandomValues(inputTensor, -fillRange, fillRange, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    inputTensor.memory().hostData();

    const auto operation = PointwiseMode::ABS;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, inputTensor);

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
    fillWithRandomValues(input0Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::ADD;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

TEST(TestGpuPointwiseMixedLayouts, BinaryMax)
{
    SKIP_IF_NO_DEVICES();

    // Input tensors have different layouts
    Tensor<float> input0Tensor({1, 3, 2, 2}, TensorLayout::NHWC);
    Tensor<float> input1Tensor({1, 3, 2, 2}, TensorLayout::NCHW);
    Tensor<float> outputCpuTensor({1, 3, 2, 2}, TensorLayout::NHWC);
    Tensor<float> outputGpuTensor({1, 3, 2, 2}, TensorLayout::NHWC);

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(input0Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed + 1);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::MAX_OP;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllEqual(outputCpuTensor, outputGpuTensor);
}

TEST(TestGpuPointwiseMixedLayouts, Compare)
{
    SKIP_IF_NO_DEVICES();

    // Input tensors have different layouts
    Tensor<float> input0Tensor({1, 3, 2, 2}, TensorLayout::NHWC);
    Tensor<float> input1Tensor({1, 3, 2, 2}, TensorLayout::NCHW);
    Tensor<bool> outputCpuTensor({1, 3, 2, 2}, TensorLayout::NHWC);
    Tensor<bool> outputGpuTensor({1, 3, 2, 2}, TensorLayout::NHWC);

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(input0Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed + 1);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::CMP_GT;
    CpuReferencePointwiseImpl<bool, double, float, float>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<bool, float, float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllEqual(outputCpuTensor, outputGpuTensor);
}

TEST(TestGpuPointwiseMixedLayouts, Ternary)
{
    SKIP_IF_NO_DEVICES();

    // The inputs and the output use different layouts
    Tensor<float> input0Tensor({1, 3, 2, 2}, TensorLayout::NHWC);
    Tensor<float> input1Tensor({1, 3, 2, 2}, TensorLayout::NCHW);
    Tensor<bool> maskTensor({1, 3, 2, 2}, TensorLayout::NHWC);
    Tensor<float> outputCpuTensor({1, 3, 2, 2}, TensorLayout::NCHW);
    Tensor<float> outputGpuTensor({1, 3, 2, 2}, TensorLayout::NCHW);

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(input0Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed + 1);
    fillMaskWithAlternatingValues(maskTensor);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::BINARY_SELECT;
    CpuReferencePointwiseImpl<float, double, float, float, bool>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor, maskTensor);

    GpuReferencePointwise::pointwiseCompute<float, float, float, bool>(
        operation, outputGpuTensor, input0Tensor, input1Tensor, maskTensor);

    assertAllEqual(outputCpuTensor, outputGpuTensor);
}

// --- Test mixed types ---

TEST(TestGpuPointwiseMixedTypes, UnaryUpcast)
{
    SKIP_IF_NO_DEVICES();

    // Input and output tensor have different types
    Tensor<HalfType> inputTensor({1, 3, 2, 2});
    Tensor<float> outputCpuTensor({1, 3, 2, 2});
    Tensor<float> outputGpuTensor({1, 3, 2, 2});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(inputTensor, HalfType{-fillRange}, HalfType{fillRange}, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    inputTensor.memory().hostData();

    const auto operation = PointwiseMode::ABS;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, inputTensor);

    GpuReferencePointwise::pointwiseCompute<float>(operation, outputGpuTensor, inputTensor);

    assertAllClose(outputCpuTensor,
                   outputGpuTensor,
                   getDynamicTolerance<float, HalfType>(operation, fillRange));
}

TEST(TestGpuPointwiseMixedTypes, UnaryDowncast)
{
    SKIP_IF_NO_DEVICES();

    // Input and output tensor have different types
    Tensor<HalfType> inputTensor({1, 3, 2, 2});
    Tensor<BFloat16Type> outputCpuTensor({1, 3, 2, 2});
    Tensor<BFloat16Type> outputGpuTensor({1, 3, 2, 2});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(inputTensor, HalfType{-fillRange}, HalfType{fillRange}, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    inputTensor.memory().hostData();

    const auto operation = PointwiseMode::ABS;
    CpuReferencePointwiseImpl<BFloat16Type, double>::pointwiseCompute(
        operation, outputCpuTensor, inputTensor);

    GpuReferencePointwise::pointwiseCompute<BFloat16Type>(operation, outputGpuTensor, inputTensor);

    assertAllClose(outputCpuTensor,
                   outputGpuTensor,
                   getDynamicTolerance<BFloat16Type, HalfType>(operation, fillRange));
}

TEST(TestGpuPointwiseMixedTypes, BinaryMixedInputs)
{
    SKIP_IF_NO_DEVICES();

    // Input tensors have different types
    Tensor<HalfType> input0Tensor({1, 3, 2, 2});
    Tensor<BFloat16Type> input1Tensor({1, 3, 2, 2});
    Tensor<float> outputCpuTensor({1, 3, 2, 2});
    Tensor<float> outputGpuTensor({1, 3, 2, 2});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(input0Tensor, HalfType{-fillRange}, HalfType{fillRange}, seed);
    fillWithRandomValues(input1Tensor, BFloat16Type{-fillRange}, BFloat16Type{fillRange}, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::ADD;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllClose(outputCpuTensor,
                   outputGpuTensor,
                   getDynamicTolerance<float, HalfType>(operation, fillRange));
}

TEST(TestGpuPointwiseMixedTypes, BinaryMinMixedInputs)
{
    SKIP_IF_NO_DEVICES();

    // Input tensors have different types
    Tensor<HalfType> input0Tensor({1, 3, 2, 2});
    Tensor<BFloat16Type> input1Tensor({1, 3, 2, 2});
    Tensor<float> outputCpuTensor({1, 3, 2, 2});
    Tensor<float> outputGpuTensor({1, 3, 2, 2});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(input0Tensor, HalfType{-fillRange}, HalfType{fillRange}, seed);
    fillWithRandomValues(input1Tensor, BFloat16Type{-fillRange}, BFloat16Type{fillRange}, seed + 1);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::MIN_OP;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllEqual(outputCpuTensor, outputGpuTensor);
}

TEST(TestGpuPointwiseMixedTypes, CompareMixedInputs)
{
    SKIP_IF_NO_DEVICES();

    // Input tensors have different types, output is boolean
    Tensor<HalfType> input0Tensor({1, 3, 2, 2});
    Tensor<BFloat16Type> input1Tensor({1, 3, 2, 2});
    Tensor<bool> outputCpuTensor({1, 3, 2, 2});
    Tensor<bool> outputGpuTensor({1, 3, 2, 2});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(input0Tensor, HalfType{-fillRange}, HalfType{fillRange}, seed);
    fillWithRandomValues(input1Tensor, BFloat16Type{-fillRange}, BFloat16Type{fillRange}, seed + 1);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::CMP_GT;
    CpuReferencePointwiseImpl<bool, double, HalfType, BFloat16Type>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<bool, HalfType, BFloat16Type>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllEqual(outputCpuTensor, outputGpuTensor);
}

TEST(TestGpuPointwiseMixedTypes, CompareInt32)
{
    SKIP_IF_NO_DEVICES();

    Tensor<int32_t> input0Tensor({1, 1, 2, 2});
    Tensor<int32_t> input1Tensor({1, 1, 2, 2});
    Tensor<bool> outputGpuTensor({1, 1, 2, 2});

    // 16777217 is not representable as a float, so these only compare correctly when two
    // integral inputs are compared exactly rather than through the float compute type
    input0Tensor.setHostValue(16777217, 0, 0, 0, 0);
    input1Tensor.setHostValue(16777216, 0, 0, 0, 0);
    input0Tensor.setHostValue(16777216, 0, 0, 0, 1);
    input1Tensor.setHostValue(16777217, 0, 0, 0, 1);
    input0Tensor.setHostValue(-3, 0, 0, 1, 0);
    input1Tensor.setHostValue(-4, 0, 0, 1, 0);
    input0Tensor.setHostValue(5, 0, 0, 1, 1);
    input1Tensor.setHostValue(5, 0, 0, 1, 1);

    GpuReferencePointwise::pointwiseCompute<bool, int32_t, int32_t, float>(
        PointwiseMode::CMP_GT, outputGpuTensor, input0Tensor, input1Tensor);

    // Non-const access to migrate the result back to the host
    outputGpuTensor.memory().hostData();

    EXPECT_TRUE(outputGpuTensor.getHostValue(0, 0, 0, 0));
    EXPECT_FALSE(outputGpuTensor.getHostValue(0, 0, 0, 1));
    EXPECT_TRUE(outputGpuTensor.getHostValue(0, 0, 1, 0));
    EXPECT_FALSE(outputGpuTensor.getHostValue(0, 0, 1, 1));
}

TEST(TestGpuPointwiseMixedTypes, TernaryMixedInputs)
{
    SKIP_IF_NO_DEVICES();

    // Input tensors have different types, output is a third type
    Tensor<HalfType> input0Tensor({1, 3, 2, 2});
    Tensor<BFloat16Type> input1Tensor({1, 3, 2, 2});
    Tensor<bool> maskTensor({1, 3, 2, 2});
    Tensor<float> outputCpuTensor({1, 3, 2, 2});
    Tensor<float> outputGpuTensor({1, 3, 2, 2});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(input0Tensor, HalfType{-fillRange}, HalfType{fillRange}, seed);
    fillWithRandomValues(input1Tensor, BFloat16Type{-fillRange}, BFloat16Type{fillRange}, seed + 1);
    fillMaskWithAlternatingValues(maskTensor);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::BINARY_SELECT;
    CpuReferencePointwiseImpl<float, double, HalfType, BFloat16Type, bool>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor, maskTensor);

    GpuReferencePointwise::pointwiseCompute<float, HalfType, BFloat16Type, bool>(
        operation, outputGpuTensor, input0Tensor, input1Tensor, maskTensor);

    assertAllEqual(outputCpuTensor, outputGpuTensor);
}

TEST(TestGpuPointwiseMixedTypes, TernaryInt32)
{
    SKIP_IF_NO_DEVICES();

    Tensor<int32_t> input0Tensor({1, 1, 2, 2});
    Tensor<int32_t> input1Tensor({1, 1, 2, 2});
    Tensor<bool> maskTensor({1, 1, 2, 2});
    Tensor<int32_t> outputGpuTensor({1, 1, 2, 2});

    input0Tensor.setHostValue(10, 0, 0, 0, 0);
    input0Tensor.setHostValue(11, 0, 0, 0, 1);
    input0Tensor.setHostValue(12, 0, 0, 1, 0);
    input0Tensor.setHostValue(13, 0, 0, 1, 1);
    input1Tensor.setHostValue(-10, 0, 0, 0, 0);
    input1Tensor.setHostValue(-11, 0, 0, 0, 1);
    input1Tensor.setHostValue(-12, 0, 0, 1, 0);
    input1Tensor.setHostValue(-13, 0, 0, 1, 1);
    maskTensor.setHostValue(true, 0, 0, 0, 0);
    maskTensor.setHostValue(false, 0, 0, 0, 1);
    maskTensor.setHostValue(false, 0, 0, 1, 0);
    maskTensor.setHostValue(true, 0, 0, 1, 1);

    GpuReferencePointwise::pointwiseCompute<int32_t, int32_t, int32_t, bool>(
        PointwiseMode::BINARY_SELECT, outputGpuTensor, input0Tensor, input1Tensor, maskTensor);

    // Non-const access to migrate the result back to the host
    outputGpuTensor.memory().hostData();

    EXPECT_EQ(outputGpuTensor.getHostValue(0, 0, 0, 0), 10);
    EXPECT_EQ(outputGpuTensor.getHostValue(0, 0, 0, 1), -11);
    EXPECT_EQ(outputGpuTensor.getHostValue(0, 0, 1, 0), -12);
    EXPECT_EQ(outputGpuTensor.getHostValue(0, 0, 1, 1), 13);
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
    fillWithRandomValues(inputTensor, -fillRange, fillRange, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    inputTensor.memory().hostData();

    const auto operation = PointwiseMode::ABS;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, inputTensor);

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
    fillWithRandomValues(input0Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::ADD;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

TEST(TestGpuPointwise1DShapes, Compare)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> input0Tensor({3});
    Tensor<float> input1Tensor({3});
    Tensor<bool> outputCpuTensor({3});
    Tensor<bool> outputGpuTensor({3});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(input0Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed + 1);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::CMP_GT;
    CpuReferencePointwiseImpl<bool, double, float, float>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<bool, float, float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllEqual(outputCpuTensor, outputGpuTensor);
}

TEST(TestGpuPointwise1DShapes, Ternary)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> input0Tensor({3});
    Tensor<float> input1Tensor({3});
    Tensor<bool> maskTensor({3});
    Tensor<float> outputCpuTensor({3});
    Tensor<float> outputGpuTensor({3});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(input0Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed + 1);
    fillMaskWithAlternatingValues(maskTensor);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::BINARY_SELECT;
    CpuReferencePointwiseImpl<float, double, float, float, bool>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor, maskTensor);

    GpuReferencePointwise::pointwiseCompute<float, float, float, bool>(
        operation, outputGpuTensor, input0Tensor, input1Tensor, maskTensor);

    assertAllEqual(outputCpuTensor, outputGpuTensor);
}

TEST(TestGpuPointwise2DShapes, Unary)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> inputTensor({3, 2});
    Tensor<float> outputCpuTensor({3, 2});
    Tensor<float> outputGpuTensor({3, 2});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(inputTensor, -fillRange, fillRange, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    inputTensor.memory().hostData();

    const auto operation = PointwiseMode::ABS;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, inputTensor);

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
    fillWithRandomValues(input0Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::ADD;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

TEST(TestGpuPointwise2DShapes, Compare)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> input0Tensor({3, 2});
    Tensor<float> input1Tensor({3, 2});
    Tensor<bool> outputCpuTensor({3, 2});
    Tensor<bool> outputGpuTensor({3, 2});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(input0Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed + 1);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::CMP_GT;
    CpuReferencePointwiseImpl<bool, double, float, float>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<bool, float, float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllEqual(outputCpuTensor, outputGpuTensor);
}

TEST(TestGpuPointwise2DShapes, Ternary)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> input0Tensor({3, 2});
    Tensor<float> input1Tensor({3, 2});
    Tensor<bool> maskTensor({3, 2});
    Tensor<float> outputCpuTensor({3, 2});
    Tensor<float> outputGpuTensor({3, 2});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(input0Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed + 1);
    fillMaskWithAlternatingValues(maskTensor);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::BINARY_SELECT;
    CpuReferencePointwiseImpl<float, double, float, float, bool>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor, maskTensor);

    GpuReferencePointwise::pointwiseCompute<float, float, float, bool>(
        operation, outputGpuTensor, input0Tensor, input1Tensor, maskTensor);

    assertAllEqual(outputCpuTensor, outputGpuTensor);
}

TEST(TestGpuPointwise3DShapes, Unary)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> inputTensor({3, 2, 4});
    Tensor<float> outputCpuTensor({3, 2, 4});
    Tensor<float> outputGpuTensor({3, 2, 4});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(inputTensor, -fillRange, fillRange, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    inputTensor.memory().hostData();

    const auto operation = PointwiseMode::ABS;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, inputTensor);

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
    fillWithRandomValues(input0Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::ADD;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

TEST(TestGpuPointwise3DShapes, Compare)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> input0Tensor({3, 2, 4});
    Tensor<float> input1Tensor({3, 2, 4});
    Tensor<bool> outputCpuTensor({3, 2, 4});
    Tensor<bool> outputGpuTensor({3, 2, 4});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(input0Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed + 1);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::CMP_GT;
    CpuReferencePointwiseImpl<bool, double, float, float>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<bool, float, float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllEqual(outputCpuTensor, outputGpuTensor);
}

TEST(TestGpuPointwise3DShapes, Ternary)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> input0Tensor({3, 2, 4});
    Tensor<float> input1Tensor({3, 2, 4});
    Tensor<bool> maskTensor({3, 2, 4});
    Tensor<float> outputCpuTensor({3, 2, 4});
    Tensor<float> outputGpuTensor({3, 2, 4});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(input0Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed + 1);
    fillMaskWithAlternatingValues(maskTensor);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::BINARY_SELECT;
    CpuReferencePointwiseImpl<float, double, float, float, bool>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor, maskTensor);

    GpuReferencePointwise::pointwiseCompute<float, float, float, bool>(
        operation, outputGpuTensor, input0Tensor, input1Tensor, maskTensor);

    assertAllEqual(outputCpuTensor, outputGpuTensor);
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
    fillWithRandomValues(inputTensor, -fillRange, fillRange, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    inputTensor.memory().hostData();

    const auto operation = PointwiseMode::ABS;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, inputTensor);

    GpuReferencePointwise::pointwiseCompute<float>(operation, outputGpuTensor, inputTensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

// --- Test validation ---

TEST(TestGpuPointwiseValidation, UnaryNotBroadcastable)
{
    SKIP_IF_NO_DEVICES();

    const auto operation = PointwiseMode::ABS;

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

TEST(TestGpuPointwiseValidation, InvalidTernaryOp)
{
    SKIP_IF_NO_DEVICES();

    // ADD is a binary operation rather than a ternary operation
    Tensor<float> input0Tensor({2, 3, 2, 4});
    Tensor<float> input1Tensor({2, 3, 2, 4});
    Tensor<bool> maskTensor({2, 3, 2, 4});
    Tensor<float> outputTensor({2, 3, 2, 4});
    EXPECT_THROW((GpuReferencePointwise::pointwiseCompute<float, float, float, bool>(
                     PointwiseMode::ADD, outputTensor, input0Tensor, input1Tensor, maskTensor)),
                 std::invalid_argument);
}

TEST(TestGpuPointwiseValidation, TernaryNotBroadcastable)
{
    SKIP_IF_NO_DEVICES();

    const auto operation = PointwiseMode::BINARY_SELECT;
    Tensor<float> outputTensor({1, 3, 2, 4});

    // Not broadcastable if the mask dims mismatch and are not 1
    {
        Tensor<float> input0Tensor({1, 3, 2, 4});
        Tensor<float> input1Tensor({1, 3, 2, 4});
        Tensor<bool> maskTensor({1, 3, 2, 2});
        EXPECT_THROW((GpuReferencePointwise::pointwiseCompute<float, float, float, bool>(
                         operation, outputTensor, input0Tensor, input1Tensor, maskTensor)),
                     std::invalid_argument);
    }

    // Not broadcastable if input1 dims mismatch and are not 1
    {
        Tensor<float> input0Tensor({1, 3, 2, 4});
        Tensor<float> input1Tensor({1, 3, 2, 2});
        Tensor<bool> maskTensor({1, 3, 2, 4});
        EXPECT_THROW((GpuReferencePointwise::pointwiseCompute<float, float, float, bool>(
                         operation, outputTensor, input0Tensor, input1Tensor, maskTensor)),
                     std::invalid_argument);
    }
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
    fillWithRandomValues(inputTensor, -fillRange, fillRange, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    inputTensor.memory().hostData();

    const auto operation = PointwiseMode::ABS;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, inputTensor);

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
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input2Tensor, -fillRange, fillRange, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input1Tensor.memory().hostData();
    input2Tensor.memory().hostData();

    const auto operation = PointwiseMode::MUL;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
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
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input2Tensor, -fillRange, fillRange, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input1Tensor.memory().hostData();
    input2Tensor.memory().hostData();

    const auto operation = PointwiseMode::MUL;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
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
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input2Tensor, -fillRange, fillRange, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input1Tensor.memory().hostData();
    input2Tensor.memory().hostData();

    const auto operation = PointwiseMode::MUL;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
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
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input2Tensor, -fillRange, fillRange, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input1Tensor.memory().hostData();
    input2Tensor.memory().hostData();

    const auto operation = PointwiseMode::MUL;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
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
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input2Tensor, -fillRange, fillRange, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input1Tensor.memory().hostData();
    input2Tensor.memory().hostData();

    const auto operation = PointwiseMode::MUL;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, input1Tensor, input2Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input1Tensor, input2Tensor);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

TEST(TestGpuPointwiseBroadcast, BinaryBroadcast3D)
{
    SKIP_IF_NO_DEVICES();

    // broadcast [4, 1, 3] and [1, 5, 3] -> [4, 5, 3]
    Tensor<float> input0Tensor({4, 1, 3});
    Tensor<float> input1Tensor({1, 5, 3});
    Tensor<float> outputCpuTensor({4, 5, 3});
    Tensor<float> outputGpuTensor({4, 5, 3});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(input0Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed + 1);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::MAX_OP;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllEqual(outputCpuTensor, outputGpuTensor);
}

TEST(TestGpuPointwiseBroadcast, BinaryMinBroadcast3D)
{
    SKIP_IF_NO_DEVICES();

    // broadcast [4, 1, 3] and [1, 5, 3] -> [4, 5, 3]
    Tensor<float> input0Tensor({4, 1, 3});
    Tensor<float> input1Tensor({1, 5, 3});
    Tensor<float> outputCpuTensor({4, 5, 3});
    Tensor<float> outputGpuTensor({4, 5, 3});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(input0Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed + 1);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::MIN_OP;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllEqual(outputCpuTensor, outputGpuTensor);
}

TEST(TestGpuPointwiseBroadcast, CompareBroadcast4D)
{
    SKIP_IF_NO_DEVICES();

    // broadcast [2, 1, 4, 1] and [1, 3, 1, 5] -> [2, 3, 4, 5]
    Tensor<float> input0Tensor({2, 1, 4, 1});
    Tensor<float> input1Tensor({1, 3, 1, 5});
    Tensor<bool> outputCpuTensor({2, 3, 4, 5});
    Tensor<bool> outputGpuTensor({2, 3, 4, 5});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(input0Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed + 1);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::CMP_GT;
    CpuReferencePointwiseImpl<bool, double, float, float>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor);

    GpuReferencePointwise::pointwiseCompute<bool, float, float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor);

    assertAllEqual(outputCpuTensor, outputGpuTensor);
}

TEST(TestGpuPointwiseBroadcast, TernaryBroadcast5D)
{
    SKIP_IF_NO_DEVICES();

    // broadcast [2, 1, 3, 1, 4], [1, 5, 1, 2, 1] and mask [3, 1, 4] -> [2, 5, 3, 2, 4]
    Tensor<float> input0Tensor({2, 1, 3, 1, 4});
    Tensor<float> input1Tensor({1, 5, 1, 2, 1});
    Tensor<bool> maskTensor({3, 1, 4});
    Tensor<float> outputCpuTensor({2, 5, 3, 2, 4});
    Tensor<float> outputGpuTensor({2, 5, 3, 2, 4});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 1.0f;
    fillWithRandomValues(input0Tensor, -fillRange, fillRange, seed);
    fillWithRandomValues(input1Tensor, -fillRange, fillRange, seed + 1);
    fillMaskWithAlternatingValues(maskTensor);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    const auto operation = PointwiseMode::BINARY_SELECT;
    CpuReferencePointwiseImpl<float, double, float, float, bool>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor, maskTensor);

    GpuReferencePointwise::pointwiseCompute<float, float, float, bool>(
        operation, outputGpuTensor, input0Tensor, input1Tensor, maskTensor);

    assertAllEqual(outputCpuTensor, outputGpuTensor);
}

// --- Test fp16 and bf16 compute types ---

TEST(TestGpuPointwiseOwnTypeComputeFp16, Binary)
{
    SKIP_IF_NO_DEVICES();

    for(const auto operation : {PointwiseMode::ADD,
                                PointwiseMode::MAX_OP,
                                PointwiseMode::SIGMOID_BWD,
                                PointwiseMode::TANH_BWD})
    {
        runGpuVsCpuPointwiseBinary<HalfType, HalfType>(operation, {1, 3, 4, 4});
    }
}

TEST(TestGpuPointwiseOwnTypeComputeBfp16, Binary)
{
    SKIP_IF_NO_DEVICES();

    for(const auto operation : {PointwiseMode::ADD,
                                PointwiseMode::MAX_OP,
                                PointwiseMode::SIGMOID_BWD,
                                PointwiseMode::TANH_BWD})
    {
        runGpuVsCpuPointwiseBinary<BFloat16Type, BFloat16Type>(operation, {1, 3, 4, 4});
    }
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
    fillWithRandomValues(inputTensor, -fillRange, fillRange, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    inputTensor.memory().hostData();

    const auto operation = PointwiseMode::SWISH_FWD;
    const float beta = 2.0f;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
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

    const auto operation = PointwiseMode::RELU_FWD;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
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

    const auto operation = PointwiseMode::RELU_BWD;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, input0Tensor, input1Tensor, lowerClip, upperClip, lowerSlope);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, input0Tensor, input1Tensor, lowerClip, upperClip, lowerSlope);
    assertAllClose(outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, 5.0f));
}

TEST(TestGpuPointwiseRefEluForward, WithAlphaVal)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> inputTensor({1, 1, 4, 4});
    Tensor<float> outputCpuTensor({1, 1, 4, 4});
    Tensor<float> outputGpuTensor({1, 1, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 3.0f;
    fillWithRandomValues(inputTensor, -fillRange, fillRange, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    inputTensor.memory().hostData();

    const auto operation = PointwiseMode::ELU_FWD;
    const float alpha = 2.5f;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, inputTensor, 0.f, 0.f, 0.f, 1.f, alpha);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, inputTensor, 0.f, 0.f, 0.f, 1.f, alpha);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

TEST(TestGpuPointwiseRefSoftplusForward, WithBetaVal)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> inputTensor({1, 1, 4, 4});
    Tensor<float> outputCpuTensor({1, 1, 4, 4});
    Tensor<float> outputGpuTensor({1, 1, 4, 4});

    // Spread the values so that some are above the threshold where softplus falls back to x
    const unsigned int seed = getGlobalTestSeed();
    const float fillRange = 12.0f;
    fillWithRandomValues(inputTensor, -fillRange, fillRange, seed);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    inputTensor.memory().hostData();

    const auto operation = PointwiseMode::SOFTPLUS_FWD;
    const float beta = 2.0f;
    CpuReferencePointwiseImpl<float, double>::pointwiseCompute(
        operation, outputCpuTensor, inputTensor, 0.f, 0.f, 0.f, 1.f, 1.f, beta);

    GpuReferencePointwise::pointwiseCompute<float>(
        operation, outputGpuTensor, inputTensor, 0.f, 0.f, 0.f, 1.f, 1.f, beta);

    assertAllClose(
        outputCpuTensor, outputGpuTensor, getDynamicTolerance<float>(operation, fillRange));
}

// --- Test NaN and infinity handling ---

// The CPU reference uses std::max and std::min: a NaN first operand propagates and a NaN second
// operand is dropped.
TEST(TestGpuPointwiseSpecialValues, Max)
{
    SKIP_IF_NO_DEVICES();

    runGpuVsCpuPointwiseBinarySpecialValues<float>(PointwiseMode::MAX_OP);
}

TEST(TestGpuPointwiseSpecialValues, Min)
{
    SKIP_IF_NO_DEVICES();

    runGpuVsCpuPointwiseBinarySpecialValues<float>(PointwiseMode::MIN_OP);
}

// Every ordered comparison with NaN is false
TEST(TestGpuPointwiseSpecialValues, CompareGreater)
{
    SKIP_IF_NO_DEVICES();

    runGpuVsCpuPointwiseBinarySpecialValues<bool>(PointwiseMode::CMP_GT);
}

// --- Test suite instantiations ---

using TestGpuPointwiseUnaryRef4DFp16 = PointwiseTestSuite<HalfType>;
using TestGpuPointwiseUnaryRef5DFp16 = PointwiseTestSuite<HalfType>;
using TestGpuPointwiseBinaryRef4DFp16 = PointwiseTestSuite<HalfType>;
using TestGpuPointwiseCompareRef4DFp16 = PointwiseTestSuite<HalfType>;
using TestGpuPointwiseBinaryRef5DFp16 = PointwiseTestSuite<HalfType>;
using TestGpuPointwiseCompareRef5DFp16 = PointwiseTestSuite<HalfType>;
using TestGpuPointwiseUnaryRef4DBfp16 = PointwiseTestSuite<BFloat16Type>;
using TestGpuPointwiseUnaryRef5DBfp16 = PointwiseTestSuite<BFloat16Type>;
using TestGpuPointwiseBinaryRef4DBfp16 = PointwiseTestSuite<BFloat16Type>;
using TestGpuPointwiseCompareRef4DBfp16 = PointwiseTestSuite<BFloat16Type>;
using TestGpuPointwiseBinaryRef5DBfp16 = PointwiseTestSuite<BFloat16Type>;
using TestGpuPointwiseCompareRef5DBfp16 = PointwiseTestSuite<BFloat16Type>;
using TestGpuPointwiseUnaryRef4DFp32 = PointwiseTestSuite<float>;
using TestGpuPointwiseUnaryRef5DFp32 = PointwiseTestSuite<float>;
using TestGpuPointwiseBinaryRef4DFp32 = PointwiseTestSuite<float>;
using TestGpuPointwiseCompareRef4DFp32 = PointwiseTestSuite<float>;
using TestGpuPointwiseBinaryRef5DFp32 = PointwiseTestSuite<float>;
using TestGpuPointwiseCompareRef5DFp32 = PointwiseTestSuite<float>;
using TestGpuPointwiseUnarySpecialValuesFp16 = PointwiseTestSuite<HalfType>;
using TestGpuPointwiseUnarySpecialValuesBfp16 = PointwiseTestSuite<BFloat16Type>;
using TestGpuPointwiseUnarySpecialValuesFp32 = PointwiseTestSuite<float>;
using TestGpuPointwiseTernaryRef4DFp16 = PointwiseTestSuite<HalfType>;
using TestGpuPointwiseTernaryRef5DFp16 = PointwiseTestSuite<HalfType>;
using TestGpuPointwiseTernaryRef4DBfp16 = PointwiseTestSuite<BFloat16Type>;
using TestGpuPointwiseTernaryRef5DBfp16 = PointwiseTestSuite<BFloat16Type>;
using TestGpuPointwiseTernaryRef4DFp32 = PointwiseTestSuite<float>;
using TestGpuPointwiseTernaryRef5DFp32 = PointwiseTestSuite<float>;

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

TEST_P(TestGpuPointwiseCompareRef4DFp16, MatchesCpuRef)
{
    this->runPointwiseCompareTest();
}

TEST_P(TestGpuPointwiseBinaryRef5DFp16, MatchesCpuRef)
{
    this->runPointwiseBinaryTest();
}

TEST_P(TestGpuPointwiseCompareRef5DFp16, MatchesCpuRef)
{
    this->runPointwiseCompareTest();
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

TEST_P(TestGpuPointwiseCompareRef4DBfp16, MatchesCpuRef)
{
    this->runPointwiseCompareTest();
}

TEST_P(TestGpuPointwiseBinaryRef5DBfp16, MatchesCpuRef)
{
    this->runPointwiseBinaryTest();
}

TEST_P(TestGpuPointwiseCompareRef5DBfp16, MatchesCpuRef)
{
    this->runPointwiseCompareTest();
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

TEST_P(TestGpuPointwiseCompareRef4DFp32, MatchesCpuRef)
{
    this->runPointwiseCompareTest();
}

TEST_P(TestGpuPointwiseBinaryRef5DFp32, MatchesCpuRef)
{
    this->runPointwiseBinaryTest();
}

TEST_P(TestGpuPointwiseCompareRef5DFp32, MatchesCpuRef)
{
    this->runPointwiseCompareTest();
}

TEST_P(TestGpuPointwiseTernaryRef4DFp16, MatchesCpuRef)
{
    this->runPointwiseTernaryTest();
}

TEST_P(TestGpuPointwiseTernaryRef5DFp16, MatchesCpuRef)
{
    this->runPointwiseTernaryTest();
}

TEST_P(TestGpuPointwiseTernaryRef4DBfp16, MatchesCpuRef)
{
    this->runPointwiseTernaryTest();
}

TEST_P(TestGpuPointwiseTernaryRef5DBfp16, MatchesCpuRef)
{
    this->runPointwiseTernaryTest();
}

TEST_P(TestGpuPointwiseTernaryRef4DFp32, MatchesCpuRef)
{
    this->runPointwiseTernaryTest();
}

TEST_P(TestGpuPointwiseTernaryRef5DFp32, MatchesCpuRef)
{
    this->runPointwiseTernaryTest();
}

TEST_P(TestGpuPointwiseUnarySpecialValuesFp16, MatchesCpuRef)
{
    this->runPointwiseUnarySpecialValuesTest();
}

TEST_P(TestGpuPointwiseUnarySpecialValuesBfp16, MatchesCpuRef)
{
    this->runPointwiseUnarySpecialValuesTest();
}

TEST_P(TestGpuPointwiseUnarySpecialValuesFp32, MatchesCpuRef)
{
    this->runPointwiseUnarySpecialValuesTest();
}

// ============================================================================
// 4D tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseUnaryRef4DFp32,
                         ::testing::ValuesIn(getSmall4dCoreUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseBinaryRef4DFp32,
                         ::testing::ValuesIn(getSmall4dCoreBinaryPointwiseCases()));
// The extended modes compile kernels the core modes do not, so their small fp32 cases run in the
// standard tier.
INSTANTIATE_TEST_SUITE_P(StandardSmall,
                         TestGpuPointwiseUnaryRef4DFp32,
                         ::testing::ValuesIn(getSmall4dExtendedUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(StandardSmall,
                         TestGpuPointwiseBinaryRef4DFp32,
                         ::testing::ValuesIn(getSmall4dExtendedBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(StandardSmall,
                         TestGpuPointwiseCompareRef4DFp32,
                         ::testing::ValuesIn(getSmall4dComparePointwiseCases()));
INSTANTIATE_TEST_SUITE_P(StandardSmall,
                         TestGpuPointwiseTernaryRef4DFp32,
                         ::testing::ValuesIn(getSmall4dTernaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnaryRef4DFp16,
                         ::testing::ValuesIn(getSmall4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseBinaryRef4DFp16,
                         ::testing::ValuesIn(getSmall4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseCompareRef4DFp16,
                         ::testing::ValuesIn(getSmall4dComparePointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseTernaryRef4DFp16,
                         ::testing::ValuesIn(getSmall4dTernaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnaryRef4DBfp16,
                         ::testing::ValuesIn(getSmall4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseBinaryRef4DBfp16,
                         ::testing::ValuesIn(getSmall4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseCompareRef4DBfp16,
                         ::testing::ValuesIn(getSmall4dComparePointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseTernaryRef4DBfp16,
                         ::testing::ValuesIn(getSmall4dTernaryPointwiseCases()));
// Medium shapes are fp32-only: dtype enters only via per-element load/store conversion (P1);
// Fp16/Bfp16 are covered by the standard tier's small-shape run.
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnaryRef4DFp32,
                         ::testing::ValuesIn(getMedium4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseBinaryRef4DFp32,
                         ::testing::ValuesIn(getMedium4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseCompareRef4DFp32,
                         ::testing::ValuesIn(getMedium4dComparePointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseTernaryRef4DFp32,
                         ::testing::ValuesIn(getMedium4dTernaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseUnaryRef4DFp32,
                         ::testing::ValuesIn(getLarge4dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseBinaryRef4DFp32,
                         ::testing::ValuesIn(getLarge4dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseCompareRef4DFp32,
                         ::testing::ValuesIn(getLarge4dComparePointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseTernaryRef4DFp32,
                         ::testing::ValuesIn(getLarge4dTernaryPointwiseCases()));

// ============================================================================
// 5D tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseUnaryRef5DFp32,
                         ::testing::ValuesIn(getSmall5dCoreUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuPointwiseBinaryRef5DFp32,
                         ::testing::ValuesIn(getSmall5dCoreBinaryPointwiseCases()));
// The extended modes compile kernels the core modes do not, so their small fp32 cases run in the
// standard tier.
INSTANTIATE_TEST_SUITE_P(StandardSmall,
                         TestGpuPointwiseUnaryRef5DFp32,
                         ::testing::ValuesIn(getSmall5dExtendedUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(StandardSmall,
                         TestGpuPointwiseBinaryRef5DFp32,
                         ::testing::ValuesIn(getSmall5dExtendedBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(StandardSmall,
                         TestGpuPointwiseCompareRef5DFp32,
                         ::testing::ValuesIn(getSmall5dComparePointwiseCases()));
INSTANTIATE_TEST_SUITE_P(StandardSmall,
                         TestGpuPointwiseTernaryRef5DFp32,
                         ::testing::ValuesIn(getSmall5dTernaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnaryRef5DFp16,
                         ::testing::ValuesIn(getSmall5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseBinaryRef5DFp16,
                         ::testing::ValuesIn(getSmall5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseCompareRef5DFp16,
                         ::testing::ValuesIn(getSmall5dComparePointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseTernaryRef5DFp16,
                         ::testing::ValuesIn(getSmall5dTernaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnaryRef5DBfp16,
                         ::testing::ValuesIn(getSmall5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseBinaryRef5DBfp16,
                         ::testing::ValuesIn(getSmall5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseCompareRef5DBfp16,
                         ::testing::ValuesIn(getSmall5dComparePointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseTernaryRef5DBfp16,
                         ::testing::ValuesIn(getSmall5dTernaryPointwiseCases()));
// Medium shapes are fp32-only: dtype enters only via per-element load/store conversion (P1);
// Fp16/Bfp16 are covered by the standard tier's small-shape run.
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnaryRef5DFp32,
                         ::testing::ValuesIn(getMedium5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseBinaryRef5DFp32,
                         ::testing::ValuesIn(getMedium5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseCompareRef5DFp32,
                         ::testing::ValuesIn(getMedium5dComparePointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseTernaryRef5DFp32,
                         ::testing::ValuesIn(getMedium5dTernaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseUnaryRef5DFp32,
                         ::testing::ValuesIn(getLarge5dUnaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseBinaryRef5DFp32,
                         ::testing::ValuesIn(getLarge5dBinaryPointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseCompareRef5DFp32,
                         ::testing::ValuesIn(getLarge5dComparePointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuPointwiseTernaryRef5DFp32,
                         ::testing::ValuesIn(getLarge5dTernaryPointwiseCases()));

// ============================================================================
// Special values (signed zeros, NaN, infinities, a negative number)
// ============================================================================

INSTANTIATE_TEST_SUITE_P(StandardSmall,
                         TestGpuPointwiseUnarySpecialValuesFp32,
                         ::testing::ValuesIn(getUnarySpecialValuePointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnarySpecialValuesFp16,
                         ::testing::ValuesIn(getUnarySpecialValuePointwiseCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuPointwiseUnarySpecialValuesBfp16,
                         ::testing::ValuesIn(getUnarySpecialValuePointwiseCases()));
