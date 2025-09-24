// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

#include <hipdnn_sdk/test_utilities/CpuFpReferenceConvolution.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionFwdInferenceBasic)
{
    // Basic convolution: 1x1x4x4 input, 1x1x3x3 weight -> 1x1x2x2 output
    Tensor<float> inputTensor({1, 1, 4, 4});
    Tensor<float> weightTensor({1, 1, 3, 3});
    Tensor<float> outputTensor({1, 1, 2, 2});

    // Fill input with sequential values
    for(int i = 0; i < 16; ++i)
    {
        inputTensor.memory().hostData()[i] = static_cast<float>(i + 1);
    }

    // Fill weights with 1s for simple summation
    for(int i = 0; i < 9; ++i)
    {
        weightTensor.memory().hostData()[i] = 1.0f;
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Expected output values for this configuration
    // Top-left 3x3 window: 1+2+3+5+6+7+9+10+11 = 54
    // Top-right 3x3 window: 2+3+4+6+7+8+10+11+12 = 63
    // Bottom-left 3x3 window: 5+6+7+9+10+11+13+14+15 = 90
    // Bottom-right 3x3 window: 6+7+8+10+11+12+14+15+16 = 99
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 0), 54.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 1), 63.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 0), 90.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 1), 99.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionFwdInferenceWithStride)
{
    // Test with stride = 2
    Tensor<float> inputTensor({1, 1, 5, 5});
    Tensor<float> weightTensor({1, 1, 3, 3});
    Tensor<float> outputTensor({1, 1, 2, 2});

    // Fill input with sequential values
    for(int i = 0; i < 25; ++i)
    {
        inputTensor.memory().hostData()[i] = static_cast<float>(i + 1);
    }

    // Fill weights with 1s
    for(int i = 0; i < 9; ++i)
    {
        weightTensor.memory().hostData()[i] = 1.0f;
    }

    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // With stride 2, we sample every other position
    // Output should be non-zero values
    EXPECT_GT(outputTensor.getHostValue(0, 0, 0, 0), 0.0f);
    EXPECT_GT(outputTensor.getHostValue(0, 0, 0, 1), 0.0f);
    EXPECT_GT(outputTensor.getHostValue(0, 0, 1, 0), 0.0f);
    EXPECT_GT(outputTensor.getHostValue(0, 0, 1, 1), 0.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionFwdInferenceWithPadding)
{
    // Test with padding = 1
    Tensor<float> inputTensor({1, 1, 3, 3});
    Tensor<float> weightTensor({1, 1, 3, 3});
    Tensor<float> outputTensor({1, 1, 3, 3});

    // Fill input with sequential values
    for(int i = 0; i < 9; ++i)
    {
        inputTensor.memory().hostData()[i] = static_cast<float>(i + 1);
    }

    // Fill weights with 1s
    for(int i = 0; i < 9; ++i)
    {
        weightTensor.memory().hostData()[i] = 1.0f;
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {1, 1};

    CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // With padding, output size should match input size
    // Center element should have the maximum value (sum of all inputs)
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 1), 45.0f); // Sum of 1-9
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionFwdInferenceMultiChannel)
{
    // Test with multiple input channels
    Tensor<float> inputTensor({1, 2, 3, 3}); // 2 input channels
    Tensor<float> weightTensor({1, 2, 2, 2}); // 1 output channel, 2 input channels
    Tensor<float> outputTensor({1, 1, 2, 2});

    // Fill input tensors
    for(int i = 0; i < 18; ++i)
    {
        inputTensor.memory().hostData()[i] = static_cast<float>(i + 1);
    }

    // Fill weights with 1s
    for(int i = 0; i < 8; ++i)
    {
        weightTensor.memory().hostData()[i] = 1.0f;
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Output should be non-zero
    EXPECT_GT(outputTensor.getHostValue(0, 0, 0, 0), 0.0f);
    EXPECT_GT(outputTensor.getHostValue(0, 0, 0, 1), 0.0f);
    EXPECT_GT(outputTensor.getHostValue(0, 0, 1, 0), 0.0f);
    EXPECT_GT(outputTensor.getHostValue(0, 0, 1, 1), 0.0f);
}

TEST(TestCpuFpReferenceConvolutionBfp16, ConvolutionFwdInferenceBasic)
{
    Tensor<hip_bfloat16> inputTensor({1, 1, 4, 4});
    Tensor<hip_bfloat16> weightTensor({1, 1, 3, 3});
    Tensor<hip_bfloat16> outputTensor({1, 1, 2, 2});

    // Fill with simple values
    for(int i = 0; i < 16; ++i)
    {
        inputTensor.memory().hostData()[i] = static_cast<hip_bfloat16>(static_cast<float>(i + 1));
    }

    for(int i = 0; i < 9; ++i)
    {
        weightTensor.memory().hostData()[i] = static_cast<hip_bfloat16>(1.0f);
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<hip_bfloat16, float>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Test that computation produces reasonable results
    EXPECT_GT(static_cast<float>(outputTensor.getHostValue(0, 0, 0, 0)), 0.0f);
    EXPECT_GT(static_cast<float>(outputTensor.getHostValue(0, 0, 1, 1)), 0.0f);
}

TEST(TestCpuFpReferenceConvolutionFp16, ConvolutionFwdInferenceBasic)
{
    Tensor<half> inputTensor({1, 1, 4, 4});
    Tensor<half> weightTensor({1, 1, 3, 3});
    Tensor<half> outputTensor({1, 1, 2, 2});

    // Fill with simple values
    for(int i = 0; i < 16; ++i)
    {
        inputTensor.memory().hostData()[i] = static_cast<half>(static_cast<float>(i + 1));
    }

    for(int i = 0; i < 9; ++i)
    {
        weightTensor.memory().hostData()[i] = static_cast<half>(1.0f);
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<half, float>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Test that computation produces reasonable results
    EXPECT_GT(static_cast<float>(outputTensor.getHostValue(0, 0, 0, 0)), 0.0f);
    EXPECT_GT(static_cast<float>(outputTensor.getHostValue(0, 0, 1, 1)), 0.0f);
}

TEST(TestCpuFpReferenceConvolutionFp64, ConvolutionFwdInferenceBasic)
{
    Tensor<double> inputTensor({1, 1, 4, 4});
    Tensor<double> weightTensor({1, 1, 3, 3});
    Tensor<double> outputTensor({1, 1, 2, 2});

    // Fill input with sequential values
    for(int i = 0; i < 16; ++i)
    {
        inputTensor.memory().hostData()[i] = static_cast<double>(i + 1);
    }

    // Fill weights with 1s
    for(int i = 0; i < 9; ++i)
    {
        weightTensor.memory().hostData()[i] = 1.0;
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<double, double>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Same expected values as fp32 test
    EXPECT_DOUBLE_EQ(outputTensor.getHostValue(0, 0, 0, 0), 54.0);
    EXPECT_DOUBLE_EQ(outputTensor.getHostValue(0, 0, 0, 1), 63.0);
    EXPECT_DOUBLE_EQ(outputTensor.getHostValue(0, 0, 1, 0), 90.0);
    EXPECT_DOUBLE_EQ(outputTensor.getHostValue(0, 0, 1, 1), 99.0);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionFwdInferenceWithDilation)
{
    // Test with dilation = 2
    Tensor<float> inputTensor({1, 1, 5, 5});
    Tensor<float> weightTensor({1, 1, 3, 3});
    Tensor<float> outputTensor({1, 1, 1, 1});

    // Fill input with sequential values
    for(int i = 0; i < 25; ++i)
    {
        inputTensor.memory().hostData()[i] = static_cast<float>(i + 1);
    }

    // Fill weights with 1s
    for(int i = 0; i < 9; ++i)
    {
        weightTensor.memory().hostData()[i] = 1.0f;
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {2, 2};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // With dilation=2, kernel samples positions: (0,0), (0,2), (0,4), (2,0), (2,2), (2,4), (4,0), (4,2), (4,4)
    // Values: 1, 3, 5, 11, 13, 15, 21, 23, 25
    // Sum: 1+3+5+11+13+15+21+23+25 = 117
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 0), 117.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionFwdInferenceSanityValidation)
{
    // Simple 1x1 convolution test for validation
    Tensor<float> inputTensor({1, 1, 2, 2});
    Tensor<float> weightTensor({1, 1, 1, 1});
    Tensor<float> outputTensor({1, 1, 2, 2});

    // Input: [1, 2; 3, 4]
    inputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0f, 0, 0, 1, 1);

    // Weight: [2]
    weightTensor.setHostValue(2.0f, 0, 0, 0, 0);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Expected output: input * weight = [2, 4; 6, 8]
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 0), 2.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 1), 4.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 0), 6.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 1), 8.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataSanityValidation)
{
    // Basic backward data convolution test
    Tensor<float> inputTensor({1, 1, 4, 4});
    Tensor<float> weightTensor({1, 1, 1, 1});
    Tensor<float> outputTensor({1, 1, 4, 4});

    // Input: [1, 2; 3, 4]
    outputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    outputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    outputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    outputTensor.setHostValue(4.0f, 0, 0, 1, 1);

    // Weight: [2]
    weightTensor.setHostValue(2.0f, 0, 0, 0, 0);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Expected output: input * weight = [2, 4; 6, 8]
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 0), 2.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 1), 4.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 0), 6.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 1), 8.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataBasic)
{
    // Basic convolution: 1x1x4x4 input, 1x1x3x3 weight -> 1x1x2x2 output
    Tensor<float> inputTensor({1, 1, 4, 4});
    Tensor<float> weightTensor({1, 1, 3, 3});
    Tensor<float> outputTensor({1, 1, 2, 2});

    // gradOutput values: simple pattern
    outputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    outputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    outputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    outputTensor.setHostValue(4.0f, 0, 0, 1, 1);

    // Weight values: simple 3x3 kernel
    std::array<float, 9> weightData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    for(size_t i = 0; i < 9; ++i)
    {
        weightTensor.memory().hostData()[i] = weightData[i];
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);
}

TEST(TestCpuFpReferenceConvolutionFp16, ConvolutionBwdDataBasic)
{
    // Basic convolution: 1x1x4x4 input, 1x1x3x3 weight -> 1x1x2x2 output
    Tensor<half> inputTensor({1, 1, 4, 4});
    Tensor<half> weightTensor({1, 1, 3, 3});
    Tensor<half> outputTensor({1, 1, 2, 2});

    // gradOutput values: simple pattern
    outputTensor.setHostValue(1.0_h, 0, 0, 0, 0);
    outputTensor.setHostValue(2.0_h, 0, 0, 0, 1);
    outputTensor.setHostValue(3.0_h, 0, 0, 1, 0);
    outputTensor.setHostValue(4.0_h, 0, 0, 1, 1);

    // Weight values: simple 3x3 kernel
    std::array<half, 9> weightData
        = {1.0_h, 2.0_h, 3.0_h, 4.0_h, 5.0_h, 6.0_h, 7.0_h, 8.0_h, 9.0_h};
    for(size_t i = 0; i < 9; ++i)
    {
        weightTensor.memory().hostData()[i] = weightData[i];
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<half, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);
}

TEST(TestCpuFpReferenceConvolutionBfp16, ConvolutionBwdDataBasic)
{
    // Basic convolution: 1x1x4x4 input, 1x1x3x3 weight -> 1x1x2x2 output
    Tensor<hip_bfloat16> inputTensor({1, 1, 4, 4});
    Tensor<hip_bfloat16> weightTensor({1, 1, 3, 3});
    Tensor<hip_bfloat16> outputTensor({1, 1, 2, 2});

    // gradOutput values: simple pattern
    outputTensor.setHostValue(1.0_bf, 0, 0, 0, 0);
    outputTensor.setHostValue(2.0_bf, 0, 0, 0, 1);
    outputTensor.setHostValue(3.0_bf, 0, 0, 1, 0);
    outputTensor.setHostValue(4.0_bf, 0, 0, 1, 1);

    // Weight values: simple 3x3 kernel
    std::array<hip_bfloat16, 9> weightData
        = {1.0_bf, 2.0_bf, 3.0_bf, 4.0_bf, 5.0_bf, 6.0_bf, 7.0_bf, 8.0_bf, 9.0_bf};
    for(size_t i = 0; i < 9; ++i)
    {
        weightTensor.memory().hostData()[i] = weightData[i];
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<hip_bfloat16, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);
}

TEST(TestCpuFpReferenceConvolutionFp64, ConvolutionBwdDataBasic)
{
    // Basic convolution: 1x1x4x4 input, 1x1x3x3 weight -> 1x1x2x2 output
    Tensor<double> inputTensor({1, 1, 4, 4});
    Tensor<double> weightTensor({1, 1, 3, 3});
    Tensor<double> outputTensor({1, 1, 2, 2});

    // gradOutput values: simple pattern
    outputTensor.setHostValue(1.0, 0, 0, 0, 0);
    outputTensor.setHostValue(2.0, 0, 0, 0, 1);
    outputTensor.setHostValue(3.0, 0, 0, 1, 0);
    outputTensor.setHostValue(4.0, 0, 0, 1, 1);

    // Weight values: simple 3x3 kernel
    std::array<double, 9> weightData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    for(size_t i = 0; i < 9; ++i)
    {
        weightTensor.memory().hostData()[i] = weightData[i];
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<double, double>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataSimple)
{
    // Basic convolution: 1x1x4x4 input, 1x1x3x3 weight -> 1x1x2x2 output
    Tensor<float> inputTensor({1, 1, 2, 2});
    Tensor<float> weightTensor({1, 1, 1, 1});
    Tensor<float> outputTensor({1, 1, 2, 2});

    // gradOutput values: simple pattern
    outputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    outputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    outputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    outputTensor.setHostValue(4.0f, 0, 0, 1, 1);

    // Weight values: simple 3x3 kernel
    std::array<float, 1> weightData = {0.5f};
    for(size_t i = 0; i < weightData.size(); ++i)
    {
        weightTensor.memory().hostData()[i] = weightData[i];
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 0), 0.5f); // 1 * 0.5
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 1), 1.0f); // 2 * 0.5
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 0), 1.5f); // 3 * 0.5
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 1), 2.0f); // 4 * 0.5
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataSimple2x2)
{
    // Basic convolution: 1x1x2x2 input, 1x1x2x2 weight -> 1x1x1x1 output
    Tensor<float> inputTensor({1, 1, 2, 2});
    Tensor<float> weightTensor({1, 1, 2, 2});
    Tensor<float> outputTensor({1, 1, 1, 1});

    // gradOutput values: simple pattern
    outputTensor.setHostValue(1.0f, 0, 0, 0, 0);

    // Weight values: simple 2x2 kernel
    std::array<float, 4> weightData = {1.0f, 2.0f, 3.0f, 4.0f};
    for(size_t i = 0; i < weightData.size(); ++i)
    {
        weightTensor.memory().hostData()[i] = weightData[i];
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 1), 2.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 0), 3.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 1), 4.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataSimple3x3)
{
    // Basic convolution: 1x1x3x3 input, 1x1x2x2 weight -> 1x1x2x2 output
    Tensor<float> inputTensor({1, 1, 3, 3});
    Tensor<float> weightTensor({1, 1, 2, 2});
    Tensor<float> outputTensor({1, 1, 2, 2});

    // gradOutput values: simple pattern
    outputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    outputTensor.setHostValue(1.0f, 0, 0, 0, 1);
    outputTensor.setHostValue(1.0f, 0, 0, 1, 0);
    outputTensor.setHostValue(1.0f, 0, 0, 1, 1);

    // Weight values: simple 2x2 kernel
    std::array<float, 4> weightData = {1.0f, 2.0f, 3.0f, 4.0f};
    for(size_t i = 0; i < weightData.size(); ++i)
    {
        weightTensor.memory().hostData()[i] = weightData[i];
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 1), 3.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 2), 2.0f);

    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 0), 4.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 1), 10.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 2), 6.0f);

    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 2, 0), 3.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 2, 1), 7.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 2, 2), 4.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataSimple3x3DifferentOutputs)
{
    // Basic convolution: 1x1x3x3 input, 1x1x2x2 weight -> 1x1x2x2 output
    Tensor<float> inputTensor({1, 1, 3, 3});
    Tensor<float> weightTensor({1, 1, 2, 2});
    Tensor<float> outputTensor({1, 1, 2, 2});

    // gradOutput values: simple pattern
    outputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    outputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    outputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    outputTensor.setHostValue(4.0f, 0, 0, 1, 1);

    // Weight values: simple 2x2 kernel
    std::array<float, 4> weightData = {1.0f, 2.0f, 3.0f, 4.0f};
    for(size_t i = 0; i < weightData.size(); ++i)
    {
        weightTensor.memory().hostData()[i] = weightData[i];
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 1), 4.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 2), 4.0f);

    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 0), 6.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 1), 20.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 2), 16.0f);

    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 2, 0), 9.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 2, 1), 24.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 2, 2), 16.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataChannels)
{
    // Test backward data convolution with multiple input and output channels
    // Input: 1x4x3x3 (1 batch, 4 input channels, 3x3 spatial)
    // Weight: 2x4x2x2 (2 output channels, 4 input channels, 2x2 kernel)
    // Output: 1x2x2x2 (1 batch, 2 output channels, 2x2 spatial)
    Tensor<float> inputTensor({1, 4, 3, 3});
    Tensor<float> weightTensor({2, 4, 2, 2});
    Tensor<float> outputTensor({1, 2, 2, 2});

    // Initialize gradient output with distinct values for each channel and position
    // Channel 0: [1, 2; 3, 4]
    // Channel 1: [5, 6; 7, 8]
    outputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    outputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    outputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    outputTensor.setHostValue(4.0f, 0, 0, 1, 1);

    outputTensor.setHostValue(5.0f, 0, 1, 0, 0);
    outputTensor.setHostValue(6.0f, 0, 1, 0, 1);
    outputTensor.setHostValue(7.0f, 0, 1, 1, 0);
    outputTensor.setHostValue(8.0f, 0, 1, 1, 1);

    // Initialize weights with distinct values for each output channel and input channel
    // Output channel 0, input channel 0: [1, 2; 3, 4]
    // Output channel 0, input channel 1: [5, 6; 7, 8]
    // Output channel 0, input channel 2: [9, 10; 11, 12]
    // Output channel 0, input channel 3: [13, 14; 15, 16]
    // Output channel 1, input channel 0: [17, 18; 19, 20]
    // Output channel 1, input channel 1: [21, 22; 23, 24]
    // Output channel 1, input channel 2: [25, 26; 27, 28]
    // Output channel 1, input channel 3: [29, 30; 31, 32]

    float weightValue = 1.0f;
    for(int oc = 0; oc < 2; ++oc)
    {
        for(int ic = 0; ic < 4; ++ic)
        {
            for(int kh = 0; kh < 2; ++kh)
            {
                for(int kw = 0; kw < 2; ++kw)
                {
                    weightTensor.setHostValue(weightValue, oc, ic, kh, kw);
                    weightValue += 1.0f;
                }
            }
        }
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Verify that all input channels receive gradients
    // Each input channel should have non-zero gradients due to contributions from both output channels
    for(int ic = 0; ic < 4; ++ic)
    {
        for(int h = 0; h < 3; ++h)
        {
            for(int w = 0; w < 3; ++w)
            {
                float gradValue = inputTensor.getHostValue(0, ic, h, w);

                // For positions that receive contributions from the kernel, gradients should be non-zero
                // Corner positions (0,0), (0,2), (2,0), (2,2) receive fewer contributions
                // Center and edge positions receive more contributions
                if((h == 0 && w == 0) || (h == 0 && w == 2) || (h == 2 && w == 0)
                   || (h == 2 && w == 2))
                {
                    // Corner positions should still have some gradient
                    EXPECT_NE(gradValue, 0.0f) << "Input channel " << ic << " at position (" << h
                                               << "," << w << ") should have non-zero gradient";
                }
                else
                {
                    // Non-corner positions should have larger gradients
                    EXPECT_GT(std::abs(gradValue), 0.0f)
                        << "Input channel " << ic << " at position (" << h << "," << w
                        << ") should have non-zero gradient";
                }
            }
        }
    }

    // Test specific expected values for input channel 0, position (0,0)
    // This position receives contribution from:
    // - Output channel 0, position (0,0) with weight (0,0,0,0) = 1.0
    // - Output channel 1, position (0,0) with weight (1,0,0,0) = 17.0
    // Expected: 1.0 * 1.0 + 5.0 * 17.0 = 1.0 + 85.0 = 86.0
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 0), 86.0f);

    // Test input channel 1, position (0,0)
    // This position receives contribution from:
    // - Output channel 0, position (0,0) with weight (0,1,0,0) = 5.0
    // - Output channel 1, position (0,0) with weight (1,1,0,0) = 21.0
    // Expected: 1.0 * 5.0 + 5.0 * 21.0 = 5.0 + 105.0 = 110.0
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 1, 0, 0), 110.0f);

    // Test input channel 2, position (0,0)
    // This position receives contribution from:
    // - Output channel 0, position (0,0) with weight (0,2,0,0) = 9.0
    // - Output channel 1, position (0,0) with weight (1,2,0,0) = 25.0
    // Expected: 1.0 * 9.0 + 5.0 * 25.0 = 9.0 + 125.0 = 134.0
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 2, 0, 0), 134.0f);

    // Test input channel 3, position (0,0)
    // This position receives contribution from:
    // - Output channel 0, position (0,0) with weight (0,3,0,0) = 13.0
    // - Output channel 1, position (0,0) with weight (1,3,0,0) = 29.0
    // Expected: 1.0 * 13.0 + 5.0 * 29.0 = 13.0 + 145.0 = 158.0
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 3, 0, 0), 158.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataStrides)
{
    // Basic convolution: 1x1x3x3 input, 1x1x2x2 weight -> 1x1x2x2 output
    Tensor<float> inputTensor({1, 1, 3, 3});
    Tensor<float> weightTensor({1, 1, 2, 2});
    Tensor<float> outputTensor({1, 1, 2, 2});

    // gradOutput values: simple pattern
    outputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    outputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    outputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    outputTensor.setHostValue(4.0f, 0, 0, 1, 1);

    // Weight values: simple 2x2 kernel
    std::array<float, 4> weightData = {1.0f, 2.0f, 3.0f, 4.0f};
    for(size_t i = 0; i < weightData.size(); ++i)
    {
        weightTensor.memory().hostData()[i] = weightData[i];
    }

    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 1), 2.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 2), 2.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 0), 3.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 1), 4.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 2), 6.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 2, 0), 3.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 2, 1), 6.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 2, 2), 4.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataDilation)
{
    // Test backward data convolution with dilation = 2
    // Input: 1x1x5x5, Weight: 1x1x3x3, Output: 1x1x1x1
    // With dilation=2, the effective kernel size becomes 5x5 (3x3 kernel with gaps)
    Tensor<float> inputTensor({1, 1, 5, 5});
    Tensor<float> weightTensor({1, 1, 3, 3});
    Tensor<float> outputTensor({1, 1, 1, 1});

    // Set gradient output value
    outputTensor.setHostValue(1.0f, 0, 0, 0, 0);

    std::array<float, 9> weightData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    for(size_t i = 0; i < weightData.size(); ++i)
    {
        weightTensor.memory().hostData()[i] = weightData[i];
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {2, 2};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 2), 2.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 4), 3.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 2, 0), 4.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 2, 2), 5.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 2, 4), 6.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 4, 0), 7.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 4, 2), 8.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 4, 4), 9.0f);

    // All other positions should be 0 (no contribution from the dilated kernel)
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 1), 0.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 3), 0.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 0), 0.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 1), 0.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 2), 0.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 3), 0.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 4), 0.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 3, 1), 0.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 3, 3), 0.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataPadding)
{
    // Test backward data convolution with padding = 1
    // Input: 1x1x4x4, Weight: 1x1x3x3, Output: 1x1x4x4 (same size due to padding)
    Tensor<float> inputTensor({1, 1, 4, 4});
    Tensor<float> weightTensor({1, 1, 3, 3});
    Tensor<float> outputTensor({1, 1, 4, 4});

    // Set gradient output values in a simple pattern
    for(int h = 0; h < 4; ++h)
    {
        for(int w = 0; w < 4; ++w)
        {
            outputTensor.setHostValue(static_cast<float>((h * 4) + w + 1), 0, 0, h, w);
        }
    }

    // Set weight values: simple 3x3 kernel
    std::array<float, 9> weightData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    for(size_t i = 0; i < weightData.size(); ++i)
    {
        weightTensor.memory().hostData()[i] = weightData[i];
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {1, 1};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // With padding=1, the input gradient should be computed correctly
    // Corner elements should have fewer contributions due to padding
    // Center elements should have more contributions

    // Test corner elements (should have fewer contributions)
    EXPECT_GT(inputTensor.getHostValue(0, 0, 0, 0), 0.0f);
    EXPECT_GT(inputTensor.getHostValue(0, 0, 0, 3), 0.0f);
    EXPECT_GT(inputTensor.getHostValue(0, 0, 3, 0), 0.0f);
    EXPECT_GT(inputTensor.getHostValue(0, 0, 3, 3), 0.0f);

    // Test center elements (should have more contributions)
    EXPECT_GT(inputTensor.getHostValue(0, 0, 1, 1), inputTensor.getHostValue(0, 0, 0, 0));
    EXPECT_GT(inputTensor.getHostValue(0, 0, 2, 2), inputTensor.getHostValue(0, 0, 0, 0));
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataPaddingAsymmetric)
{
    // Test backward data convolution with asymmetric padding
    // Input: 1x1x3x4, Weight: 1x1x2x2, Output: 1x1x3x4 (asymmetric padding [1,1])
    Tensor<float> inputTensor({1, 1, 3, 4});
    Tensor<float> weightTensor({1, 1, 2, 2});
    Tensor<float> outputTensor({1, 1, 3, 4});

    // Set gradient output values
    for(int h = 0; h < 3; ++h)
    {
        for(int w = 0; w < 4; ++w)
        {
            outputTensor.setHostValue(1.0f, 0, 0, h, w);
        }
    }

    // Set weight values: simple 2x2 kernel
    std::array<float, 4> weightData = {1.0f, 2.0f, 3.0f, 4.0f};
    for(size_t i = 0; i < weightData.size(); ++i)
    {
        weightTensor.memory().hostData()[i] = weightData[i];
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {1, 1};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // All values should be positive due to uniform gradient output
    for(int h = 0; h < 3; ++h)
    {
        for(int w = 0; w < 4; ++w)
        {
            EXPECT_GT(inputTensor.getHostValue(0, 0, h, w), 0.0f);
        }
    }
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataPaddingZero)
{
    // Test backward data convolution with zero padding (no padding)
    // Input: 1x1x3x3, Weight: 1x1x2x2, Output: 1x1x2x2
    Tensor<float> inputTensor({1, 1, 3, 3});
    Tensor<float> weightTensor({1, 1, 2, 2});
    Tensor<float> outputTensor({1, 1, 2, 2});

    // Set gradient output values
    outputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    outputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    outputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    outputTensor.setHostValue(4.0f, 0, 0, 1, 1);

    // Set weight values: simple 2x2 kernel
    std::array<float, 4> weightData = {1.0f, 1.0f, 1.0f, 1.0f};
    for(size_t i = 0; i < weightData.size(); ++i)
    {
        weightTensor.memory().hostData()[i] = weightData[i];
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // With no padding and uniform weights, verify expected values
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 1), 3.0f); // 1 + 2
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 2), 2.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 0), 4.0f); // 1 + 3
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 1), 10.0f); // 1 + 2 + 3 + 4
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 2), 6.0f); // 2 + 4
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 2, 0), 3.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 2, 1), 7.0f); // 3 + 4
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 2, 2), 4.0f);
}
