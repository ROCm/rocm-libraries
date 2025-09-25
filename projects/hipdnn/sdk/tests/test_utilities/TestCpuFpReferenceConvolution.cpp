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
