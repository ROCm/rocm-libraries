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

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionFwd1D)
{
    // Test 1D convolution: NCW format
    // Input: 1x1x8 (1 batch, 1 channel, 8 spatial)
    // Weight: 1x1x3 (1 output channel, 1 input channel, 3 kernel)
    // Output: 1x1x6 (1 batch, 1 channel, 6 spatial)
    Tensor<float> inputTensor({1, 1, 8});
    Tensor<float> weightTensor({1, 1, 3});
    Tensor<float> outputTensor({1, 1, 6});

    // Fill input with sequential values [1, 2, 3, 4, 5, 6, 7, 8]
    for(int i = 0; i < 8; ++i)
    {
        inputTensor.setHostValue(static_cast<float>(i + 1), 0, 0, i);
    }

    // Fill weights with [1, 2, 1] for weighted average
    weightTensor.setHostValue(1.0f, 0, 0, 0);
    weightTensor.setHostValue(2.0f, 0, 0, 1);
    weightTensor.setHostValue(1.0f, 0, 0, 2);

    std::vector<int64_t> strides = {1};
    std::vector<int64_t> dilations = {1};
    std::vector<int64_t> padding = {0};

    CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Expected outputs: weighted sums
    // Position 0: 1*1 + 2*2 + 3*1 = 8
    // Position 1: 2*1 + 3*2 + 4*1 = 12
    // Position 2: 3*1 + 4*2 + 5*1 = 16
    // Position 3: 4*1 + 5*2 + 6*1 = 20
    // Position 4: 5*1 + 6*2 + 7*1 = 24
    // Position 5: 6*1 + 7*2 + 8*1 = 28
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0), 8.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1), 12.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 2), 16.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 3), 20.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 4), 24.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 5), 28.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionFwd1DStride)
{
    // Test 1D convolution with stride
    // Input: 1x1x10 (1 batch, 1 channel, 10 spatial)
    // Weight: 1x1x3 (1 output channel, 1 input channel, 3 kernel)
    // Output: 1x1x4 (1 batch, 1 channel, 4 spatial) with stride=2
    Tensor<float> inputTensor({1, 1, 10});
    Tensor<float> weightTensor({1, 1, 3});
    Tensor<float> outputTensor({1, 1, 4});

    // Fill input with sequential values
    for(int i = 0; i < 10; ++i)
    {
        inputTensor.setHostValue(static_cast<float>(i + 1), 0, 0, i);
    }

    // Fill weights with [1, 1, 1]
    for(int i = 0; i < 3; ++i)
    {
        weightTensor.setHostValue(1.0f, 0, 0, i);
    }

    std::vector<int64_t> strides = {2}; // stride=2 in width dimension
    std::vector<int64_t> dilations = {1};
    std::vector<int64_t> padding = {0};

    CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // With stride=2, we sample at positions 0, 2, 4, 6
    // Position 0: 1+2+3 = 6
    // Position 2: 3+4+5 = 12
    // Position 4: 5+6+7 = 18
    // Position 6: 7+8+9 = 24
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0), 6.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1), 12.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 2), 18.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 3), 24.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionFwd3DGrouped)
{
    // Test 3D grouped convolution with 2 groups
    // Input: 1x4x2x2x2 (1 batch, 4 channels, 2x2x2 spatial)
    // Weight: 2x2x2x2x2 (2 output channels, 2 input channels per group, 2x2x2 kernel)
    // Output: 1x2x1x1x1 (1 batch, 2 output channels, 1x1x1 spatial)
    Tensor<float> inputTensor({1, 4, 2, 2, 2});
    Tensor<float> weightTensor({2, 2, 2, 2, 2});
    Tensor<float> outputTensor({1, 2, 1, 1, 1});

    // Fill input channels with distinct values
    for(int c = 0; c < 4; ++c)
    {
        // f(0) = 10, f(1) = 20, ..., f(3) = 40 for channel 0
        auto baseValue = static_cast<float>((c + 1) * 10);
        for(int i = 0; i < 8; ++i)
        {
            // g(f(0), 0) = 10, g(f(0), 1) = 11, ..., g(f(0), 7) = 17
            inputTensor.memory().hostData()[(c * 8) + i] = baseValue + static_cast<float>(i);
        }
    }

    // Fill weights: different for each group
    // Group 0 weights: all 0.1
    for(int i = 0; i < 16; ++i)
    {
        weightTensor.memory().hostData()[i] = 0.1f;
    }
    // Group 1 weights: all 0.2
    for(int i = 16; i < 32; ++i)
    {
        weightTensor.memory().hostData()[i] = 0.2f;
    }

    std::vector<int64_t> strides = {1, 1, 1};
    std::vector<int64_t> dilations = {1, 1, 1};
    std::vector<int64_t> padding = {0, 0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Verify grouped convolution produces different outputs for each group
    float output0 = outputTensor.getHostValue(0, 0, 0, 0, 0);
    float output1 = outputTensor.getHostValue(0, 1, 0, 0, 0);

    // 0.1 * (sum(i for i in range(10, 18)) + sum(i for i in range(20, 28))) = 29.6
    EXPECT_EQ(output0, 29.6f) << "Group 0 output should be 29.6";

    // 0.2 * (sum(i for i in range(30, 38)) + sum(i for i in range(40, 48))) = 123.2
    EXPECT_EQ(output1, 123.200005f) << "Group 1 output should be 123.2";
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionFwd3D)
{
    // Test 3D convolution with all weights = 1
    Tensor<float> inputTensor({1, 1, 3, 3, 3});
    Tensor<float> weightTensor({1, 1, 2, 2, 2});
    Tensor<float> outputTensor({1, 1, 2, 2, 2});

    // Fill input with sequential values
    float inputValue = 1.0f;
    for(int d = 0; d < 3; ++d)
    {
        for(int h = 0; h < 3; ++h)
        {
            for(int w = 0; w < 3; ++w)
            {
                inputTensor.setHostValue(inputValue++, 0, 0, d, h, w);
            }
        }
    }

    // Fill weights with all 1s (2x2x2 kernel)
    for(int i = 0; i < 8; ++i)
    {
        weightTensor.memory().hostData()[i] = 1.0f;
    }

    std::vector<int64_t> strides = {1, 1, 1};
    std::vector<int64_t> dilations = {1, 1, 1};
    std::vector<int64_t> padding = {0, 0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Each output is sum of 2x2x2 = 8 input values
    // Output[0,0,0] = 1+2+4+5+10+11+13+14 = 60
    // Output[0,0,1] = 2+3+5+6+11+12+14+15 = 68
    // Output[0,1,0] = 4+5+7+8+13+14+16+17 = 84
    // Output[0,1,1] = 5+6+8+9+14+15+17+18 = 92
    // Output[1,0,0] = 10+11+13+14+19+20+22+23 = 132
    // Output[1,0,1] = 11+12+14+15+20+21+23+24 = 140
    // Output[1,1,0] = 13+14+16+17+22+23+25+26 = 156
    // Output[1,1,1] = 14+15+17+18+23+24+26+27 = 164

    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 0, 0), 60.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 0, 1), 68.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 1, 0), 84.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 1, 1), 92.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 0, 0), 132.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 0, 1), 140.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 1, 0), 156.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 1, 1), 164.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionFwd3DNdhwc)
{
    // Test 3D convolution with all weights = 1
    Tensor<float> inputTensor({1, 1, 3, 3, 3}, TensorLayout::NDHWC);
    Tensor<float> weightTensor({1, 1, 2, 2, 2}, TensorLayout::NDHWC);
    Tensor<float> outputTensor({1, 1, 2, 2, 2}, TensorLayout::NDHWC);

    // Fill input with sequential values
    float inputValue = 1.0f;
    for(int d = 0; d < 3; ++d)
    {
        for(int h = 0; h < 3; ++h)
        {
            for(int w = 0; w < 3; ++w)
            {
                inputTensor.setHostValue(inputValue++, 0, 0, d, h, w);
            }
        }
    }

    // Fill weights with all 1s (2x2x2 kernel)
    for(int i = 0; i < 8; ++i)
    {
        weightTensor.memory().hostData()[i] = 1.0f;
    }

    std::vector<int64_t> strides = {1, 1, 1};
    std::vector<int64_t> dilations = {1, 1, 1};
    std::vector<int64_t> padding = {0, 0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Each output is sum of 2x2x2 = 8 input values
    // Output[0,0,0] = 1+2+4+5+10+11+13+14 = 60
    // Output[0,0,1] = 2+3+5+6+11+12+14+15 = 68
    // Output[0,1,0] = 4+5+7+8+13+14+16+17 = 84
    // Output[0,1,1] = 5+6+8+9+14+15+17+18 = 92
    // Output[1,0,0] = 10+11+13+14+19+20+22+23 = 132
    // Output[1,0,1] = 11+12+14+15+20+21+23+24 = 140
    // Output[1,1,0] = 13+14+16+17+22+23+25+26 = 156
    // Output[1,1,1] = 14+15+17+18+23+24+26+27 = 164

    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 0, 0), 60.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 0, 1), 68.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 1, 0), 84.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 1, 1), 92.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 0, 0), 132.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 0, 1), 140.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 1, 0), 156.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 1, 1), 164.0f);
}

TEST(TestCpuFpReferenceConvolutionFp64, ConvolutionFwd1D)
{
    // Test 1D convolution with fp64
    Tensor<double> inputTensor({1, 1, 5});
    Tensor<double> weightTensor({1, 1, 2});
    Tensor<double> outputTensor({1, 1, 4});

    // Fill input
    for(int i = 0; i < 5; ++i)
    {
        inputTensor.setHostValue(static_cast<double>(i + 1), 0, 0, i);
    }

    // Fill weights
    weightTensor.setHostValue(1.0, 0, 0, 0);
    weightTensor.setHostValue(-1.0, 0, 0, 1);

    std::vector<int64_t> strides = {1};
    std::vector<int64_t> dilations = {1};
    std::vector<int64_t> padding = {0};

    CpuFpReferenceConvolutionImpl<double, double>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Expected: difference between consecutive elements
    EXPECT_DOUBLE_EQ(outputTensor.getHostValue(0, 0, 0), -1.0); // 1 - 2
    EXPECT_DOUBLE_EQ(outputTensor.getHostValue(0, 0, 1), -1.0); // 2 - 3
    EXPECT_DOUBLE_EQ(outputTensor.getHostValue(0, 0, 2), -1.0); // 3 - 4
    EXPECT_DOUBLE_EQ(outputTensor.getHostValue(0, 0, 3), -1.0); // 4 - 5
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionFwd2D)
{
    // Test 2D convolution with all weights = 1, varying input values
    // This makes it easy to verify: output = sum of covered input values
    Tensor<float> inputTensor({1, 1, 4, 4});
    Tensor<float> weightTensor({1, 1, 2, 2});
    Tensor<float> outputTensor({1, 1, 3, 3});

    // Fill input with sequential values for easy verification
    // Input:
    // [ 1,  2,  3,  4]
    // [ 5,  6,  7,  8]
    // [ 9, 10, 11, 12]
    // [13, 14, 15, 16]
    float inputValue = 1.0f;
    for(int h = 0; h < 4; ++h)
    {
        for(int w = 0; w < 4; ++w)
        {
            inputTensor.setHostValue(inputValue++, 0, 0, h, w);
        }
    }

    // Fill weights with all 1s (2x2 kernel)
    for(int i = 0; i < 4; ++i)
    {
        weightTensor.memory().hostData()[i] = 1.0f;
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Verify ALL output values
    // Output[0,0] = 1+2+5+6 = 14
    // Output[0,1] = 2+3+6+7 = 18
    // Output[0,2] = 3+4+7+8 = 22
    // Output[1,0] = 5+6+9+10 = 30
    // Output[1,1] = 6+7+10+11 = 34
    // Output[1,2] = 7+8+11+12 = 38
    // Output[2,0] = 9+10+13+14 = 46
    // Output[2,1] = 10+11+14+15 = 50
    // Output[2,2] = 11+12+15+16 = 54

    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 0), 14.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 1), 18.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 2), 22.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 0), 30.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 1), 34.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 2), 38.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 2, 0), 46.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 2, 1), 50.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 2, 2), 54.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionFwd2DNhwc)
{
    // Test 2D convolution with all weights = 1, varying input values
    // This makes it easy to verify: output = sum of covered input values
    Tensor<float> inputTensor({1, 1, 4, 4}, TensorLayout::NHWC);
    Tensor<float> weightTensor({1, 1, 2, 2}, TensorLayout::NHWC);
    Tensor<float> outputTensor({1, 1, 3, 3}, TensorLayout::NHWC);

    // Fill input with sequential values for easy verification
    // Input:
    // [ 1,  2,  3,  4]
    // [ 5,  6,  7,  8]
    // [ 9, 10, 11, 12]
    // [13, 14, 15, 16]
    float inputValue = 1.0f;
    for(int h = 0; h < 4; ++h)
    {
        for(int w = 0; w < 4; ++w)
        {
            inputTensor.setHostValue(inputValue++, 0, 0, h, w);
        }
    }

    // Fill weights with all 1s (2x2 kernel)
    for(int i = 0; i < 4; ++i)
    {
        weightTensor.memory().hostData()[i] = 1.0f;
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Verify ALL output values
    // Output[0,0] = 1+2+5+6 = 14
    // Output[0,1] = 2+3+6+7 = 18
    // Output[0,2] = 3+4+7+8 = 22
    // Output[1,0] = 5+6+9+10 = 30
    // Output[1,1] = 6+7+10+11 = 34
    // Output[1,2] = 7+8+11+12 = 38
    // Output[2,0] = 9+10+13+14 = 46
    // Output[2,1] = 10+11+14+15 = 50
    // Output[2,2] = 11+12+15+16 = 54

    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 0), 14.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 1), 18.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 2), 22.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 0), 30.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 1), 34.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 2), 38.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 2, 0), 46.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 2, 1), 50.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 2, 2), 54.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionFwd2DSymmetricPadding)
{
    // Test 2D convolution with symmetric padding
    Tensor<float> inputTensor({1, 1, 3, 3});
    Tensor<float> weightTensor({1, 1, 3, 3});
    Tensor<float> outputTensor({1, 1, 3, 3});

    // Fill input with simple pattern
    // Input:
    // [1, 2, 3]
    // [4, 5, 6]
    // [7, 8, 9]
    float inputValue = 1.0f;
    for(int h = 0; h < 3; ++h)
    {
        for(int w = 0; w < 3; ++w)
        {
            inputTensor.setHostValue(inputValue++, 0, 0, h, w);
        }
    }

    // Fill weights with all 1s for easy calculation
    for(int i = 0; i < 9; ++i)
    {
        weightTensor.memory().hostData()[i] = 1.0f;
    }

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {1, 1}; // Symmetric padding

    CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Verify ALL output values with padding
    // With padding=1, we have implicit 0s around the input
    // Output[0,0] = 0+0+0 + 0+1+2 + 0+4+5 = 12
    // Output[0,1] = 0+0+0 + 1+2+3 + 4+5+6 = 21
    // Output[0,2] = 0+0+0 + 2+3+0 + 5+6+0 = 16
    // Output[1,0] = 0+1+2 + 0+4+5 + 0+7+8 = 27
    // Output[1,1] = 1+2+3 + 4+5+6 + 7+8+9 = 45 (all input values)
    // Output[1,2] = 2+3+0 + 5+6+0 + 8+9+0 = 33
    // Output[2,0] = 0+4+5 + 0+7+8 + 0+0+0 = 24
    // Output[2,1] = 4+5+6 + 7+8+9 + 0+0+0 = 39
    // Output[2,2] = 5+6+0 + 8+9+0 + 0+0+0 = 28

    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 0), 12.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 1), 21.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 2), 16.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 0), 27.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 1), 45.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 2), 33.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 2, 0), 24.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 2, 1), 39.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 2, 2), 28.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionFwd3DSymmetricPadding)
{
    // Test 3D convolution with symmetric padding
    Tensor<float> inputTensor({1, 1, 2, 2, 2});
    Tensor<float> weightTensor({1, 1, 2, 2, 2});
    Tensor<float> outputTensor({1, 1, 3, 3, 3}); // Larger due to padding

    // Fill input with sequential values 1-8
    float inputValue = 1.0f;
    for(int d = 0; d < 2; ++d)
    {
        for(int h = 0; h < 2; ++h)
        {
            for(int w = 0; w < 2; ++w)
            {
                inputTensor.setHostValue(inputValue++, 0, 0, d, h, w);
            }
        }
    }

    // Fill weights with all 1s for easy calculation
    for(int i = 0; i < 8; ++i)
    {
        weightTensor.memory().hostData()[i] = 1.0f;
    }

    std::vector<int64_t> strides = {1, 1, 1};
    std::vector<int64_t> dilations = {1, 1, 1};
    std::vector<int64_t> padding = {1, 1, 1}; // Symmetric padding in all dimensions

    CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    const std::vector<float> expectedOutput = {// d=0
                                               1,
                                               3,
                                               2,
                                               4,
                                               10,
                                               6,
                                               3,
                                               7,
                                               4,
                                               // d=1
                                               6,
                                               14,
                                               8,
                                               16,
                                               36,
                                               20,
                                               10,
                                               22,
                                               12,
                                               // d=2
                                               5,
                                               11,
                                               6,
                                               12,
                                               26,
                                               14,
                                               7,
                                               15,
                                               8};

    int index = 0;
    for(int d = 0; d < 3; ++d)
    {
        for(int h = 0; h < 3; ++h)
        {
            for(int w = 0; w < 3; ++w)
            {
                float actual = outputTensor.getHostValue(0, 0, d, h, w);
                float expected = expectedOutput[static_cast<size_t>(index++)];
                EXPECT_FLOAT_EQ(actual, expected) << "Mismatch at output coordinate (d,h,w) = ("
                                                  << d << "," << h << "," << w << ")";
            }
        }
    }
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionFwd2DAsymmetricPadding)
{
    // Test 2D convolution with asymmetric padding
    Tensor<float> inputTensor({1, 1, 3, 3});
    Tensor<float> weightTensor({1, 1, 2, 2});
    Tensor<float> outputTensor({1, 1, 4, 5}); // Different output size due to asymmetric padding

    // Fill input with simple pattern
    float inputValue = 1.0f;
    for(int h = 0; h < 3; ++h)
    {
        for(int w = 0; w < 3; ++w)
        {
            inputTensor.setHostValue(inputValue++, 0, 0, h, w);
        }
    }

    // Fill weights with simple pattern
    weightTensor.setHostValue(1.0f, 0, 0, 0, 0);
    weightTensor.setHostValue(2.0f, 0, 0, 0, 1);
    weightTensor.setHostValue(3.0f, 0, 0, 1, 0);
    weightTensor.setHostValue(4.0f, 0, 0, 1, 1);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> prePadding = {1, 1}; // 1 padding at top/left
    std::vector<int64_t> postPadding = {1, 2}; // 1 padding at bottom, 2 at right

    CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, prePadding, postPadding);

    // Verify all output values (4x5 output)
    // Row 0: includes top padding
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 0), 4.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 1), 11.0f); // 1*3 + 2*4
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 2), 18.0f); // 2*3 + 3*4
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 3), 9.0f); // 3*3 + 0*4
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 4), 0.0f);

    // Verify we have all 20 output values (4x5)
    int count = 0;
    for(int h = 0; h < 4; ++h)
    {
        for(int w = 0; w < 5; ++w)
        {
            float val = outputTensor.getHostValue(0, 0, h, w);
            EXPECT_GE(val, 0.0f) << "Output at (" << h << "," << w << ") should be non-negative";
            count++;
        }
    }
    EXPECT_EQ(count, 20);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionFwd3DAsymmetricPadding)
{
    // Test 3D convolution with asymmetric padding
    Tensor<float> inputTensor({1, 1, 2, 2, 2});
    Tensor<float> weightTensor({1, 1, 2, 2, 2});
    Tensor<float> outputTensor({1, 1, 3, 1, 4}); // Different dimensions due to asymmetric padding

    // Fill input with sequential values
    float inputValue = 1.0f;
    for(int d = 0; d < 2; ++d)
    {
        for(int h = 0; h < 2; ++h)
        {
            for(int w = 0; w < 2; ++w)
            {
                inputTensor.setHostValue(inputValue++, 0, 0, d, h, w);
            }
        }
    }

    // Fill weights with all 1s
    for(int i = 0; i < 8; ++i)
    {
        weightTensor.memory().hostData()[i] = 1.0f;
    }

    std::vector<int64_t> strides = {1, 1, 1};
    std::vector<int64_t> dilations = {1, 1, 1};
    std::vector<int64_t> prePadding = {1, 0, 1}; // Different padding for each dimension
    std::vector<int64_t> postPadding = {1, 0, 2}; // Different padding for each dimension

    CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
        inputTensor, weightTensor, outputTensor, strides, dilations, prePadding, postPadding);

    // Verify all values are computed
    for(int d = 0; d < 3; ++d)
    {
        for(int h = 0; h < 1; ++h)
        {
            for(int w = 0; w < 4; ++w)
            {
                float val = outputTensor.getHostValue(0, 0, d, h, w);
                EXPECT_GE(val, 0.0f)
                    << "Output at (" << d << "," << h << "," << w << ") should be non-negative";
            }
        }
    }
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

    for(int ic = 0; ic < 4; ++ic)
    {
        for(int h = 0; h < 3; ++h)
        {
            for(int w = 0; w < 3; ++w)
            {
                float gradValue = inputTensor.getHostValue(0, ic, h, w);

                if((h == 0 && w == 0) || (h == 0 && w == 2) || (h == 2 && w == 0)
                   || (h == 2 && w == 2))
                {
                    EXPECT_NE(gradValue, 0.0f) << "Input channel " << ic << " at position (" << h
                                               << "," << w << ") should have non-zero gradient";
                }
                else
                {
                    EXPECT_GT(std::abs(gradValue), 0.0f)
                        << "Input channel " << ic << " at position (" << h << "," << w
                        << ") should have non-zero gradient";
                }
            }
        }
    }

    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 0), 86.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 1, 0, 0), 110.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 2, 0, 0), 134.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 3, 0, 0), 158.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataGroupedChannels)
{
    // Test grouped convolution backward data with 2 groups
    // Input: 1x4x2x2 (1 batch, 4 input channels, 2x2 spatial)
    // Weight: 2x2x1x1 (2 total output channels, 2 input channels per group, 1x1 kernel)
    // Output: 1x2x2x2 (1 batch, 2 output channels, 2x2 spatial)
    // Groups: 2 (4 input channels / 2 channels per group)
    Tensor<float> inputTensor({1, 4, 2, 2});
    Tensor<float> weightTensor({2, 2, 1, 1});
    Tensor<float> outputTensor({1, 2, 2, 2});

    // Set gradient output values: different values for each output channel
    // Output channel 0 (group 0): [1, 2; 3, 4]
    outputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    outputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    outputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    outputTensor.setHostValue(4.0f, 0, 0, 1, 1);

    // Output channel 1 (group 1): [5, 6; 7, 8]
    outputTensor.setHostValue(5.0f, 0, 1, 0, 0);
    outputTensor.setHostValue(6.0f, 0, 1, 0, 1);
    outputTensor.setHostValue(7.0f, 0, 1, 1, 0);
    outputTensor.setHostValue(8.0f, 0, 1, 1, 1);

    // Set weight values: simple 1x1 kernels
    // Group 0, output channel 0: input channels 0,1 -> weights [0.5, 1.0]
    weightTensor.setHostValue(0.5f, 0, 0, 0, 0); // weight for input channel 0
    weightTensor.setHostValue(1.0f, 0, 1, 0, 0); // weight for input channel 1

    // Group 1, output channel 1: input channels 2,3 -> weights [1.5, 2.0]
    weightTensor.setHostValue(1.5f, 1, 0, 0, 0); // weight for input channel 2
    weightTensor.setHostValue(2.0f, 1, 1, 0, 0); // weight for input channel 3

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Verify grouped convolution: each group should only affect its corresponding input channels
    // Group 0 (input channels 0,1) should be affected by output channel 0
    // Input channel 0: gradOutput[0] * weight[0,0] = [1,2,3,4] * 0.5 = [0.5,1.0,1.5,2.0]
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 0), 0.5f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 1), 1.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 0), 1.5f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 1), 2.0f);

    // Input channel 1: gradOutput[0] * weight[0,1] = [1,2,3,4] * 1.0 = [1.0,2.0,3.0,4.0]
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 1, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 1, 0, 1), 2.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 1, 1, 0), 3.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 1, 1, 1), 4.0f);

    // Group 1 (input channels 2,3) should be affected by output channel 1
    // Input channel 2: gradOutput[1] * weight[1,0] = [5,6,7,8] * 1.5 = [7.5,9.0,10.5,12.0]
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 2, 0, 0), 7.5f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 2, 0, 1), 9.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 2, 1, 0), 10.5f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 2, 1, 1), 12.0f);

    // Input channel 3: gradOutput[1] * weight[1,1] = [5,6,7,8] * 2.0 = [10.0,12.0,14.0,16.0]
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 3, 0, 0), 10.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 3, 0, 1), 12.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 3, 1, 0), 14.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 3, 1, 1), 16.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataGroupedBatches)
{
    // Test grouped convolution backward data with multiple batches (2 batches)
    // Using smallest possible tensor size: 2x2x1x1 input, 2x1x1x1 weight, 2x2x1x1 output
    // Input: 2x2x1x1 (2 batches, 2 input channels, 1x1 spatial)
    // Weight: 2x1x1x1 (2 total output channels, 1 input channel per group, 1x1 kernel)
    // Output: 2x2x1x1 (2 batches, 2 output channels, 1x1 spatial)
    // Groups: 2 (2 input channels / 1 channel per group)
    Tensor<float> inputTensor({2, 2, 1, 1});
    Tensor<float> weightTensor({2, 1, 1, 1});
    Tensor<float> outputTensor({2, 2, 1, 1});

    // Set gradient output values for both batches
    // Batch 0: output channels [1.0, 2.0]
    outputTensor.setHostValue(1.0f, 0, 0, 0, 0); // batch 0, channel 0
    outputTensor.setHostValue(2.0f, 0, 1, 0, 0); // batch 0, channel 1

    // Batch 1: output channels [3.0, 4.0]
    outputTensor.setHostValue(3.0f, 1, 0, 0, 0); // batch 1, channel 0
    outputTensor.setHostValue(4.0f, 1, 1, 0, 0); // batch 1, channel 1

    // Set weight values for grouped convolution
    // Group 0: weight = 0.5 (for input channel 0 -> output channel 0)
    weightTensor.setHostValue(0.5f, 0, 0, 0, 0);
    // Group 1: weight = 1.5 (for input channel 1 -> output channel 1)
    weightTensor.setHostValue(1.5f, 1, 0, 0, 0);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Verify grouped convolution results for both batches
    // Batch 0: input channel 0 = gradOutput[0,0] * weight[0] = 1.0 * 0.5 = 0.5
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 0), 0.5f);
    // Batch 0: input channel 1 = gradOutput[0,1] * weight[1] = 2.0 * 1.5 = 3.0
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 1, 0, 0), 3.0f);

    // Batch 1: input channel 0 = gradOutput[1,0] * weight[0] = 3.0 * 0.5 = 1.5
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(1, 0, 0, 0), 1.5f);
    // Batch 1: input channel 1 = gradOutput[1,1] * weight[1] = 4.0 * 1.5 = 6.0
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(1, 1, 0, 0), 6.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataGroupedAsymmetricInput)
{
    // Test grouped convolution backward data with asymmetric input (smallest possible size)
    // Input: 1x2x1x2 (1 batch, 2 input channels, 1x2 spatial - asymmetric)
    // Weight: 2x1x1x1 (2 total output channels, 1 input channel per group, 1x1 kernel)
    // Output: 1x2x1x2 (1 batch, 2 output channels, 1x2 spatial - asymmetric)
    // Groups: 2 (2 input channels / 1 channel per group)
    Tensor<float> inputTensor({1, 2, 1, 2});
    Tensor<float> weightTensor({2, 1, 1, 1});
    Tensor<float> outputTensor({1, 2, 1, 2});

    // Set gradient output values for asymmetric spatial dimensions
    // Output channel 0: [1.0, 2.0] (1x2 spatial)
    outputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    outputTensor.setHostValue(2.0f, 0, 0, 0, 1);

    // Output channel 1: [3.0, 4.0] (1x2 spatial)
    outputTensor.setHostValue(3.0f, 0, 1, 0, 0);
    outputTensor.setHostValue(4.0f, 0, 1, 0, 1);

    // Set weight values for grouped convolution
    // Group 0: weight = 0.5 (for input channel 0 -> output channel 0)
    weightTensor.setHostValue(0.5f, 0, 0, 0, 0);
    // Group 1: weight = 1.5 (for input channel 1 -> output channel 1)
    weightTensor.setHostValue(1.5f, 1, 0, 0, 0);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Verify grouped convolution results for asymmetric input
    // Input channel 0: gradOutput[0] * weight[0] = [1.0, 2.0] * 0.5 = [0.5, 1.0]
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 0), 0.5f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 1), 1.0f);

    // Input channel 1: gradOutput[1] * weight[1] = [3.0, 4.0] * 1.5 = [4.5, 6.0]
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 1, 0, 0), 4.5f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 1, 0, 1), 6.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataGroupedAsymmetricKernel)
{
    // Test grouped convolution backward data with asymmetric kernel (smallest possible size)
    // Input: 1x2x2x3 (1 batch, 2 input channels, 2x3 spatial)
    // Weight: 2x1x1x2 (2 total output channels, 1 input channel per group, 1x2 asymmetric kernel)
    // Output: 1x2x2x2 (1 batch, 2 output channels, 2x2 spatial)
    // Groups: 2 (2 input channels / 1 channel per group)
    Tensor<float> inputTensor({1, 2, 2, 3});
    Tensor<float> weightTensor({2, 1, 1, 2}); // Asymmetric kernel: 1x2
    Tensor<float> outputTensor({1, 2, 2, 2});

    // Set gradient output values for both output channels
    // Output channel 0: [1.0, 2.0; 3.0, 4.0]
    outputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    outputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    outputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    outputTensor.setHostValue(4.0f, 0, 0, 1, 1);

    // Output channel 1: [5.0, 6.0; 7.0, 8.0]
    outputTensor.setHostValue(5.0f, 0, 1, 0, 0);
    outputTensor.setHostValue(6.0f, 0, 1, 0, 1);
    outputTensor.setHostValue(7.0f, 0, 1, 1, 0);
    outputTensor.setHostValue(8.0f, 0, 1, 1, 1);

    // Set asymmetric weight values for grouped convolution
    // Group 0: weight = [0.5, 1.0] (1x2 kernel for input channel 0 -> output channel 0)
    weightTensor.setHostValue(0.5f, 0, 0, 0, 0);
    weightTensor.setHostValue(1.0f, 0, 0, 0, 1);

    // Group 1: weight = [1.5, 2.0] (1x2 kernel for input channel 1 -> output channel 1)
    weightTensor.setHostValue(1.5f, 1, 0, 0, 0);
    weightTensor.setHostValue(2.0f, 1, 0, 0, 1);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Verify grouped convolution results with asymmetric kernel
    // Group 0 (input channel 0): gradOutput[0] convolved with weight[0]
    // Position (0,0): gradOutput[0,0,0,0] * weight[0,0,0,0] = 1.0 * 0.5 = 0.5
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 0), 0.5f);
    // Position (0,1): gradOutput[0,0,0,0] * weight[0,0,0,1] + gradOutput[0,0,0,1] * weight[0,0,0,0] = 1.0 * 1.0 + 2.0 * 0.5 = 2.0
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 1), 2.0f);
    // Position (0,2): gradOutput[0,0,0,1] * weight[0,0,0,1] = 2.0 * 1.0 = 2.0
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 2), 2.0f);
    // Position (1,0): gradOutput[0,0,1,0] * weight[0,0,0,0] = 3.0 * 0.5 = 1.5
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 0), 1.5f);
    // Position (1,1): gradOutput[0,0,1,0] * weight[0,0,0,1] + gradOutput[0,0,1,1] * weight[0,0,0,0] = 3.0 * 1.0 + 4.0 * 0.5 = 5.0
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 1), 5.0f);
    // Position (1,2): gradOutput[0,0,1,1] * weight[0,0,0,1] = 4.0 * 1.0 = 4.0
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 2), 4.0f);

    // Group 1 (input channel 1): gradOutput[1] convolved with weight[1]
    // Position (0,0): gradOutput[0,1,0,0] * weight[1,0,0,0] = 5.0 * 1.5 = 7.5
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 1, 0, 0), 7.5f);
    // Position (0,1): gradOutput[0,1,0,0] * weight[1,0,0,1] + gradOutput[0,1,0,1] * weight[1,0,0,0] = 5.0 * 2.0 + 6.0 * 1.5 = 19.0
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 1, 0, 1), 19.0f);
    // Position (0,2): gradOutput[0,1,0,1] * weight[1,0,0,1] = 6.0 * 2.0 = 12.0
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 1, 0, 2), 12.0f);
    // Position (1,0): gradOutput[0,1,1,0] * weight[1,0,0,0] = 7.0 * 1.5 = 10.5
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 1, 1, 0), 10.5f);
    // Position (1,1): gradOutput[0,1,1,0] * weight[1,0,0,1] + gradOutput[0,1,1,1] * weight[1,0,0,0] = 7.0 * 2.0 + 8.0 * 1.5 = 26.0
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 1, 1, 1), 26.0f);
    // Position (1,2): gradOutput[0,1,1,1] * weight[1,0,0,1] = 8.0 * 2.0 = 16.0
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 1, 1, 2), 16.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataStrides)
{
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    // Basic convolution: 1x1x3x3 input, 1x1x2x2 weight -> 1x1x1x1 output with kernel strides
    Tensor<float> inputTensor({1, 1, 3, 3});
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

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 1), 2.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 2), 0.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 0), 3.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 1), 4.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 2), 0.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 2, 0), 0.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 2, 1), 0.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 2, 2), 0.0f);
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
    // Input: 1x1x3x4, Weight: 1x1x2x2, Output: 1x1x4x5 (with uniform padding [1,1])
    Tensor<float> inputTensor({1, 1, 3, 4});
    Tensor<float> weightTensor({1, 1, 2, 2});
    Tensor<float> outputTensor({1, 1, 4, 5});

    // Set gradient output values
    for(int h = 0; h < 4; ++h)
    {
        for(int w = 0; w < 5; ++w)
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

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionWrwBasic)
{
    // Minimal sanity test for convBwdWeight using smallest possible tensor sizes
    // Input: 1x1x2x2 (1 batch, 1 input channel, 2x2 spatial)
    // Weight: 1x1x1x1 (1 output channel, 1 input channel, 1x1 kernel)
    // GradOutput: 1x1x2x2 (1 batch, 1 output channel, 2x2 spatial)
    Tensor<float> inputTensor({1, 1, 2, 2});
    Tensor<float> gradWeightTensor({1, 1, 1, 1});
    Tensor<float> gradOutputTensor({1, 1, 2, 2});

    // Set input values: [1, 2; 3, 4]
    inputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0f, 0, 0, 1, 1);

    // Set gradient output values: [0.5, 0.5; 0.5, 0.5]
    gradOutputTensor.setHostValue(0.5f, 0, 0, 0, 0);
    gradOutputTensor.setHostValue(0.5f, 0, 0, 0, 1);
    gradOutputTensor.setHostValue(0.5f, 0, 0, 1, 0);
    gradOutputTensor.setHostValue(0.5f, 0, 0, 1, 1);

    // Initialize weight to zero
    gradWeightTensor.setHostValue(0.0f, 0, 0, 0, 0);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdWeight(
        inputTensor, gradWeightTensor, gradOutputTensor, strides, dilations, padding);

    // Expected weight gradient: sum of (input * gradOutput) = (1+2+3+4) * 0.5 = 5.0
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 0), 5.0f);
}

TEST(TestCpuFpReferenceConvolutionFp16, ConvolutionWrwBasic)
{
    // Minimal sanity test for convBwdWeight using smallest possible tensor sizes
    // Input: 1x1x2x2 (1 batch, 1 input channel, 2x2 spatial)
    // Weight: 1x1x1x1 (1 output channel, 1 input channel, 1x1 kernel)
    // GradOutput: 1x1x2x2 (1 batch, 1 output channel, 2x2 spatial)
    Tensor<half> inputTensor({1, 1, 2, 2});
    Tensor<half> gradWeightTensor({1, 1, 1, 1});
    Tensor<half> gradOutputTensor({1, 1, 2, 2});

    // Set input values: [1, 2; 3, 4]
    inputTensor.setHostValue(1.0_h, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0_h, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0_h, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0_h, 0, 0, 1, 1);

    // Set gradient output values: [0.5, 0.5; 0.5, 0.5]
    gradOutputTensor.setHostValue(0.5_h, 0, 0, 0, 0);
    gradOutputTensor.setHostValue(0.5_h, 0, 0, 0, 1);
    gradOutputTensor.setHostValue(0.5_h, 0, 0, 1, 0);
    gradOutputTensor.setHostValue(0.5_h, 0, 0, 1, 1);

    // Initialize weight to zero
    gradWeightTensor.setHostValue(0.0_h, 0, 0, 0, 0);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<half, float>::convBwdWeight(
        inputTensor, gradWeightTensor, gradOutputTensor, strides, dilations, padding);

    // Expected weight gradient: sum of (input * gradOutput) = (1+2+3+4) * 0.5 = 5.0
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 0), 5.0f);
}

TEST(TestCpuFpReferenceConvolutionBfp16, ConvolutionWrwBasic)
{
    // Minimal sanity test for convBwdWeight using smallest possible tensor sizes
    // Input: 1x1x2x2 (1 batch, 1 input channel, 2x2 spatial)
    // Weight: 1x1x1x1 (1 output channel, 1 input channel, 1x1 kernel)
    // GradOutput: 1x1x2x2 (1 batch, 1 output channel, 2x2 spatial)
    Tensor<hip_bfloat16> inputTensor({1, 1, 2, 2});
    Tensor<hip_bfloat16> gradWeightTensor({1, 1, 1, 1});
    Tensor<hip_bfloat16> gradOutputTensor({1, 1, 2, 2});

    // Set input values: [1, 2; 3, 4]
    inputTensor.setHostValue(1.0_bf, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0_bf, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0_bf, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0_bf, 0, 0, 1, 1);

    // Set gradient output values: [0.5, 0.5; 0.5, 0.5]
    gradOutputTensor.setHostValue(0.5_bf, 0, 0, 0, 0);
    gradOutputTensor.setHostValue(0.5_bf, 0, 0, 0, 1);
    gradOutputTensor.setHostValue(0.5_bf, 0, 0, 1, 0);
    gradOutputTensor.setHostValue(0.5_bf, 0, 0, 1, 1);

    // Initialize weight to zero
    gradWeightTensor.setHostValue(0.0_bf, 0, 0, 0, 0);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<hip_bfloat16, float>::convBwdWeight(
        inputTensor, gradWeightTensor, gradOutputTensor, strides, dilations, padding);

    // Expected weight gradient: sum of (input * gradOutput) = (1+2+3+4) * 0.5 = 5.0
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 0), 5.0f);
}

TEST(TestCpuFpReferenceConvolutionFp64, ConvolutionWrwBasic)
{
    // Minimal sanity test for convBwdWeight using smallest possible tensor sizes
    // Input: 1x1x2x2 (1 batch, 1 input channel, 2x2 spatial)
    // Weight: 1x1x1x1 (1 output channel, 1 input channel, 1x1 kernel)
    // GradOutput: 1x1x2x2 (1 batch, 1 output channel, 2x2 spatial)
    Tensor<double> inputTensor({1, 1, 2, 2});
    Tensor<double> gradWeightTensor({1, 1, 1, 1});
    Tensor<double> gradOutputTensor({1, 1, 2, 2});

    // Set input values: [1, 2; 3, 4]
    inputTensor.setHostValue(1.0, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0, 0, 0, 1, 1);

    // Set gradient output values: [0.5, 0.5; 0.5, 0.5]
    gradOutputTensor.setHostValue(0.5, 0, 0, 0, 0);
    gradOutputTensor.setHostValue(0.5, 0, 0, 0, 1);
    gradOutputTensor.setHostValue(0.5, 0, 0, 1, 0);
    gradOutputTensor.setHostValue(0.5, 0, 0, 1, 1);

    // Initialize weight to zero
    gradWeightTensor.setHostValue(0.0, 0, 0, 0, 0);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<double, double>::convBwdWeight(
        inputTensor, gradWeightTensor, gradOutputTensor, strides, dilations, padding);

    // Expected weight gradient: sum of (input * gradOutput) = (1+2+3+4) * 0.5 = 5.0
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 0), 5.0);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvBwdWeightMultiBatch)
{
    // Test convBwdWeight with multiple batches (minimal size: 2 batches)
    // Input: 2x1x2x2 (2 batches, 1 input channel, 2x2 spatial)
    // Weight: 1x1x1x1 (1 output channel, 1 input channel, 1x1 kernel)
    // GradOutput: 2x1x2x2 (2 batches, 1 output channel, 2x2 spatial)
    Tensor<float> inputTensor({2, 1, 2, 2});
    Tensor<float> gradWeightTensor({1, 1, 1, 1});
    Tensor<float> gradOutputTensor({2, 1, 2, 2});

    // Batch 0 input: [1, 2; 3, 4]
    inputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0f, 0, 0, 1, 1);

    // Batch 1 input: [5, 6; 7, 8]
    inputTensor.setHostValue(5.0f, 1, 0, 0, 0);
    inputTensor.setHostValue(6.0f, 1, 0, 0, 1);
    inputTensor.setHostValue(7.0f, 1, 0, 1, 0);
    inputTensor.setHostValue(8.0f, 1, 0, 1, 1);

    // Uniform gradient output: all 0.1
    gradOutputTensor.fillWithValue(0.1f);

    gradWeightTensor.setHostValue(0.0f, 0, 0, 0, 0);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdWeight(
        inputTensor, gradWeightTensor, gradOutputTensor, strides, dilations, padding);

    // Expected: sum across all batches = (1+2+3+4+5+6+7+8) * 0.1 = 3.6
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 0), 3.6f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvBwdWeightMultiChannel)
{
    // Test convBwdWeight with multiple input/output channels (minimal size)
    // Input: 1x2x2x2 (1 batch, 2 input channels, 2x2 spatial)
    // Weight: 2x2x1x1 (2 output channels, 2 input channels, 1x1 kernel)
    // GradOutput: 1x2x2x2 (1 batch, 2 output channels, 2x2 spatial)
    Tensor<float> inputTensor({1, 2, 2, 2});
    Tensor<float> gradWeightTensor({2, 2, 1, 1});
    Tensor<float> gradOutputTensor({1, 2, 2, 2});

    // Input channel 0: [1, 2; 3, 4]
    inputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0f, 0, 0, 1, 1);

    // Input channel 1: [5, 6; 7, 8]
    inputTensor.setHostValue(5.0f, 0, 1, 0, 0);
    inputTensor.setHostValue(6.0f, 0, 1, 0, 1);
    inputTensor.setHostValue(7.0f, 0, 1, 1, 0);
    inputTensor.setHostValue(8.0f, 0, 1, 1, 1);

    // GradOutput channel 0: all 0.1
    // GradOutput channel 1: all 0.2
    for(int h = 0; h < 2; ++h)
    {
        for(int w = 0; w < 2; ++w)
        {
            gradOutputTensor.setHostValue(0.1f, 0, 0, h, w);
            gradOutputTensor.setHostValue(0.2f, 0, 1, h, w);
        }
    }

    // Initialize weights to zero
    gradWeightTensor.fillWithValue(0.0f);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdWeight(
        inputTensor, gradWeightTensor, gradOutputTensor, strides, dilations, padding);

    // Expected gradients:
    // gradWeight[0,0] = sum(input[0] * gradOutput[0]) = (1+2+3+4) * 0.1 = 1.0
    // gradWeight[0,1] = sum(input[1] * gradOutput[0]) = (5+6+7+8) * 0.1 = 2.6
    // gradWeight[1,0] = sum(input[0] * gradOutput[1]) = (1+2+3+4) * 0.2 = 2.0
    // gradWeight[1,1] = sum(input[1] * gradOutput[1]) = (5+6+7+8) * 0.2 = 5.2
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 1, 0, 0), 2.6f);
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(1, 0, 0, 0), 2.0f);
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(1, 1, 0, 0), 5.2f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvBwdWeightGrouped)
{
    // Test convBwdWeight with groups (minimal size)
    // Input: 1x2x2x2 (1 batch, 2 input channels, 2x2 spatial)
    // Weight: 2x1x1x1 (2 output channels, 1 input channel per group, 1x1 kernel)
    // GradOutput: 1x2x2x2 (1 batch, 2 output channels, 2x2 spatial)
    Tensor<float> inputTensor({1, 2, 2, 2});
    Tensor<float> gradWeightTensor({2, 1, 1, 1});
    Tensor<float> gradOutputTensor({1, 2, 2, 2});

    // Input channel 0: [1, 2; 3, 4]
    inputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0f, 0, 0, 1, 1);

    // Input channel 1: [5, 6; 7, 8]
    inputTensor.setHostValue(5.0f, 0, 1, 0, 0);
    inputTensor.setHostValue(6.0f, 0, 1, 0, 1);
    inputTensor.setHostValue(7.0f, 0, 1, 1, 0);
    inputTensor.setHostValue(8.0f, 0, 1, 1, 1);

    // GradOutput channel 0: all 0.1
    // GradOutput channel 1: all 0.2
    for(int h = 0; h < 2; ++h)
    {
        for(int w = 0; w < 2; ++w)
        {
            gradOutputTensor.setHostValue(0.1f, 0, 0, h, w);
            gradOutputTensor.setHostValue(0.2f, 0, 1, h, w);
        }
    }

    // Initialize weights to zero
    gradWeightTensor.fillWithValue(0.0f);
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};
    CpuFpReferenceConvolutionImpl<float, float>::convBwdWeight(
        inputTensor, gradWeightTensor, gradOutputTensor, strides, dilations, padding);

    // Expected gradients:
    // gradWeight[0,0] = sum(input[0] * gradOutput[0]) = (1+2+3+4) * 0.1 = 1.0
    // gradWeight[1,0] = sum(input[1] * gradOutput[1]) = (5+6+7+8) * 0.2 = 5.2
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(1, 0, 0, 0), 5.2f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvBwdWeightAsymmetricPadding)
{
    // Test convBwdWeight with asymmetric padding (minimal tensor sizes)
    // Input: 1x1x2x2 (1 batch, 1 input channel, 2x2 spatial)
    // Weight: 1x1x1x1 (1 output channel, 1 input channel, 1x1 kernel)
    // GradOutput: 1x1x3x4 (1 batch, 1 output channel, 3x4 spatial with asymmetric padding)
    Tensor<float> inputTensor({1, 1, 2, 2});
    Tensor<float> gradWeightTensor({1, 1, 1, 1});
    Tensor<float> gradOutputTensor({1, 1, 4, 5});

    // Set input values: [1, 2; 3, 4]
    inputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0f, 0, 0, 1, 1);

    // Set uniform gradient output values: all 0.1
    gradOutputTensor.fillWithValue(0.1f);

    // Initialize weight gradient to zero
    gradWeightTensor.setHostValue(0.0f, 0, 0, 0, 0);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> prePadding = {1, 1}; // 1 padding at top/left
    std::vector<int64_t> postPadding = {1, 2}; // 1 padding at bottom, 2 at right

    CpuFpReferenceConvolutionImpl<float, float>::convBwdWeight(inputTensor,
                                                               gradWeightTensor,
                                                               gradOutputTensor,
                                                               strides,
                                                               dilations,
                                                               prePadding,
                                                               postPadding);

    // Expected weight gradient: sum of (input * gradOutput) = (1+2+3+4) * 0.1 = 1.0
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 0), 1.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvBwdWeightStrides)
{
    // Test convBwdWeight with strides (minimal size)
    // Input: 1x1x3x3 (1 batch, 1 input channel, 3x3 spatial)
    // Weight: 1x1x2x2 (1 output channel, 1 input channel, 2x2 kernel)
    // GradOutput: 1x1x1x1 (1 batch, 1 output channel, 1x1 spatial with stride=2)
    Tensor<float> inputTensor({1, 1, 3, 3});
    Tensor<float> gradWeightTensor({1, 1, 2, 2});
    Tensor<float> gradOutputTensor({1, 1, 1, 1});

    // Input: [1, 2, 3; 4, 5, 6; 7, 8, 9]
    float inputVal = 1.0f;

    for(int h = 0; h < 3; ++h)
    {
        for(int w = 0; w < 3; ++w)
        {

            inputTensor.setHostValue(inputVal, 0, 0, h, w);
            inputVal += 1.0f;
        }
    }

    // Single gradient output value
    gradOutputTensor.setHostValue(1.0f, 0, 0, 0, 0);

    // Initialize weight gradients to zero
    gradWeightTensor.fillWithValue(0.0f);

    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdWeight(
        inputTensor, gradWeightTensor, gradOutputTensor, strides, dilations, padding);

    // With stride=2, the kernel samples input positions [0,0], [0,1], [1,0], [1,1]
    // Expected gradients: input values at those positions
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 0), 1.0f); // input[0,0]
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 1), 2.0f); // input[0,1]
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 1, 0), 4.0f); // input[1,0]
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 1, 1), 5.0f); // input[1,1]
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvBwdWeightDilations)
{
    // Test convBwdWeight with dilations (minimal size)
    // Input: 1x1x3x3 (1 batch, 1 input channel, 3x3 spatial)
    // Weight: 1x1x2x2 (1 output channel, 1 input channel, 2x2 kernel)
    // GradOutput: 1x1x1x1 (1 batch, 1 output channel, 1x1 spatial with dilation=2)
    Tensor<float> inputTensor({1, 1, 3, 3});
    Tensor<float> gradWeightTensor({1, 1, 2, 2});
    Tensor<float> gradOutputTensor({1, 1, 1, 1});

    // Input: [1, 2, 3; 4, 5, 6; 7, 8, 9]
    float inputVal = 1.0f;
    for(int h = 0; h < 3; ++h)
    {
        for(int w = 0; w < 3; ++w)
        {
            inputTensor.setHostValue(inputVal, 0, 0, h, w);
            inputVal += 1.0f;
        }
    }

    // Single gradient output value
    gradOutputTensor.setHostValue(1.0f, 0, 0, 0, 0);

    // Initialize weight gradients to zero
    gradWeightTensor.fillWithValue(0.0f);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {2, 2};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdWeight(
        inputTensor, gradWeightTensor, gradOutputTensor, strides, dilations, padding);

    // With dilation=2, the kernel samples input positions [0,0], [0,2], [2,0], [2,2]
    // Expected gradients: input values at those positions
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 0), 1.0f); // input[0,0]
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 1), 3.0f); // input[0,2]
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 1, 0), 7.0f); // input[2,0]
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 1, 1), 9.0f); // input[2,2]
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvBwdWeightPadding)
{
    // Test convBwdWeight with padding (minimal size)
    // Input: 1x1x2x2 (1 batch, 1 input channel, 2x2 spatial)
    // Weight: 1x1x2x2 (1 output channel, 1 input channel, 2x2 kernel)
    // GradOutput: 1x1x3x3 (1 batch, 1 output channel, 3x3 spatial with padding=1)
    Tensor<float> inputTensor({1, 1, 2, 2});
    Tensor<float> gradWeightTensor({1, 1, 2, 2});
    Tensor<float> gradOutputTensor({1, 1, 3, 3});

    // Input: [1, 2; 3, 4]
    inputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0f, 0, 0, 1, 1);

    // Uniform gradient output: all 0.25
    gradOutputTensor.fillWithValue(0.25f);

    // Initialize weight gradients to zero
    gradWeightTensor.fillWithValue(0.0f);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {1, 1};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdWeight(
        inputTensor, gradWeightTensor, gradOutputTensor, strides, dilations, padding);

    // With padding=1, each input element contributes to multiple weight positions
    // Center input elements contribute more than corner elements
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 0), 2.5f);
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 1), 2.5f);
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 1, 0), 2.5f);
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 1, 1), 2.5f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvBwdWeightAsymmetricInput)
{
    // Test convBwdWeight with asymmetric input dimensions (minimal size)
    // Input: 1x1x2x3 (1 batch, 1 input channel, 2x3 spatial - asymmetric)
    // Weight: 1x1x1x2 (1 output channel, 1 input channel, 1x2 kernel - asymmetric)
    // GradOutput: 1x1x2x2 (1 batch, 1 output channel, 2x2 spatial)
    Tensor<float> inputTensor({1, 1, 2, 3});
    Tensor<float> gradWeightTensor({1, 1, 1, 2});
    Tensor<float> gradOutputTensor({1, 1, 2, 2});

    // Input: [1, 2, 3; 4, 5, 6]
    inputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0f, 0, 0, 0, 2);
    inputTensor.setHostValue(4.0f, 0, 0, 1, 0);
    inputTensor.setHostValue(5.0f, 0, 0, 1, 1);
    inputTensor.setHostValue(6.0f, 0, 0, 1, 2);

    // GradOutput: [0.1, 0.2; 0.3, 0.4]
    gradOutputTensor.setHostValue(0.1f, 0, 0, 0, 0);
    gradOutputTensor.setHostValue(0.2f, 0, 0, 0, 1);
    gradOutputTensor.setHostValue(0.3f, 0, 0, 1, 0);
    gradOutputTensor.setHostValue(0.4f, 0, 0, 1, 1);

    // Initialize weight gradients to zero
    gradWeightTensor.setHostValue(0.0f, 0, 0, 0, 0);
    gradWeightTensor.setHostValue(0.0f, 0, 0, 0, 1);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdWeight(
        inputTensor, gradWeightTensor, gradOutputTensor, strides, dilations, padding);

    // Expected gradients for asymmetric 1x2 kernel:
    // gradWeight[0,0,0,0] = input[0,0]*gradOut[0,0] + input[0,1]*gradOut[0,1] + input[1,0]*gradOut[1,0] + input[1,1]*gradOut[1,1]
    //                     = 1*0.1 + 2*0.2 + 4*0.3 + 5*0.4 = 0.1 + 0.4 + 1.2 + 2.0 = 3.7
    // gradWeight[0,0,0,1] = input[0,1]*gradOut[0,0] + input[0,2]*gradOut[0,1] + input[1,1]*gradOut[1,0] + input[1,2]*gradOut[1,1]
    //                     = 2*0.1 + 3*0.2 + 5*0.3 + 6*0.4 = 0.2 + 0.6 + 1.5 + 2.4 = 4.7
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 0), 3.7f);
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 1), 4.7f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvBwdWeightAsymmetricKernel)
{
    // Test convBwdWeight with asymmetric kernel dimensions (minimal size)
    // Input: 1x1x3x2 (1 batch, 1 input channel, 3x2 spatial)
    // Weight: 1x1x2x1 (1 output channel, 1 input channel, 2x1 kernel - asymmetric)
    // GradOutput: 1x1x2x2 (1 batch, 1 output channel, 2x2 spatial)
    Tensor<float> inputTensor({1, 1, 3, 2});
    Tensor<float> gradWeightTensor({1, 1, 2, 1});
    Tensor<float> gradOutputTensor({1, 1, 2, 2});

    // Input: [1, 2; 3, 4; 5, 6]
    inputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0f, 0, 0, 1, 1);
    inputTensor.setHostValue(5.0f, 0, 0, 2, 0);
    inputTensor.setHostValue(6.0f, 0, 0, 2, 1);

    // GradOutput: [0.1, 0.2; 0.3, 0.4]
    gradOutputTensor.setHostValue(0.1f, 0, 0, 0, 0);
    gradOutputTensor.setHostValue(0.2f, 0, 0, 0, 1);
    gradOutputTensor.setHostValue(0.3f, 0, 0, 1, 0);
    gradOutputTensor.setHostValue(0.4f, 0, 0, 1, 1);

    // Initialize weight gradients to zero
    gradWeightTensor.setHostValue(0.0f, 0, 0, 0, 0);
    gradWeightTensor.setHostValue(0.0f, 0, 0, 1, 0);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdWeight(
        inputTensor, gradWeightTensor, gradOutputTensor, strides, dilations, padding);

    // Expected gradients for asymmetric 2x1 kernel:
    // gradWeight[0,0,0,0] = input[0,0]*gradOut[0,0] + input[0,1]*gradOut[0,1] + input[1,0]*gradOut[1,0] + input[1,1]*gradOut[1,1]
    //                     = 1*0.1 + 2*0.2 + 3*0.3 + 4*0.4 = 0.1 + 0.4 + 0.9 + 1.6 = 3.0
    // gradWeight[0,0,1,0] = input[1,0]*gradOut[0,0] + input[1,1]*gradOut[0,1] + input[2,0]*gradOut[1,0] + input[2,1]*gradOut[1,1]
    //                     = 3*0.1 + 4*0.2 + 5*0.3 + 6*0.4 = 0.3 + 0.8 + 1.5 + 2.4 = 5.0
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 0), 3.0f);
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 1, 0), 5.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdWeight1D)
{
    // Test 1D backward weight convolution
    Tensor<float> inputTensor({1, 1, 6});
    Tensor<float> weightTensor({1, 1, 3});
    Tensor<float> outputTensor({1, 1, 4});

    // Set input values
    for(int i = 0; i < 6; ++i)
    {
        inputTensor.setHostValue(static_cast<float>(i + 1), 0, 0, i);
    }

    // Set gradient output values
    for(int i = 0; i < 4; ++i)
    {
        outputTensor.setHostValue(static_cast<float>(i + 1), 0, 0, i);
    }

    std::vector<int64_t> strides = {1};
    std::vector<int64_t> dilations = {1};
    std::vector<int64_t> padding = {0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdWeight(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Verify weight gradient computation
    // Weight[0] = sum(input[0:3] * output) = 1*1 + 2*2 + 3*3 + 4*4 = 30
    // Weight[1] = sum(input[1:4] * output) = 2*1 + 3*2 + 4*3 + 5*4 = 40
    // Weight[2] = sum(input[2:5] * output) = 3*1 + 4*2 + 5*3 + 6*4 = 50
    EXPECT_FLOAT_EQ(weightTensor.getHostValue(0, 0, 0), 30.0f);
    EXPECT_FLOAT_EQ(weightTensor.getHostValue(0, 0, 1), 40.0f);
    EXPECT_FLOAT_EQ(weightTensor.getHostValue(0, 0, 2), 50.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdWeight3D)
{
    // Test 3D backward weight convolution with minimal tensor sizes
    // Input: 1x1x2x2x2 (1 batch, 1 input channel, 2x2x2 spatial)
    // Weight: 1x1x2x2x2 (1 output channel, 1 input channel, 2x2x2 kernel)
    // GradOutput: 1x1x1x1x1 (1 batch, 1 output channel, 1x1x1 spatial)
    Tensor<float> inputTensor({1, 1, 2, 2, 2});
    Tensor<float> gradWeightTensor({1, 1, 2, 2, 2});
    Tensor<float> gradOutputTensor({1, 1, 1, 1, 1});

    // Set input values sequentially: [1, 2, 3, 4, 5, 6, 7, 8]
    // This creates a simple pattern for verification
    float inputValue = 1.0f;
    for(int d = 0; d < 2; ++d)
    {
        for(int h = 0; h < 2; ++h)
        {
            for(int w = 0; w < 2; ++w)
            {
                inputTensor.setHostValue(inputValue, 0, 0, d, h, w);
                inputValue += 1.0f;
            }
        }
    }

    // Set uniform gradient output: 1.0
    // This simplifies the calculation: weight gradient = input value * 1.0 = input value
    gradOutputTensor.setHostValue(1.0f, 0, 0, 0, 0, 0);

    // Initialize weight gradients to zero
    gradWeightTensor.fillWithValue(0.0f);

    std::vector<int64_t> strides = {1, 1, 1};
    std::vector<int64_t> dilations = {1, 1, 1};
    std::vector<int64_t> padding = {0, 0, 0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdWeight(
        inputTensor, gradWeightTensor, gradOutputTensor, strides, dilations, padding);

    // Verify weight gradients: each should equal the corresponding input value
    // Since gradOutput = 1.0, weight_grad[d,h,w] = input[d,h,w] * gradOutput = input[d,h,w]
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 0, 0), 1.0f); // input[0,0,0]
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 0, 1), 2.0f); // input[0,0,1]
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 1, 0), 3.0f); // input[0,1,0]
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 0, 1, 1), 4.0f); // input[0,1,1]
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 1, 0, 0), 5.0f); // input[1,0,0]
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 1, 0, 1), 6.0f); // input[1,0,1]
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 1, 1, 0), 7.0f); // input[1,1,0]
    EXPECT_FLOAT_EQ(gradWeightTensor.getHostValue(0, 0, 1, 1, 1), 8.0f); // input[1,1,1]
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdData1D)
{
    // Test 1D backward data convolution
    Tensor<float> inputTensor({1, 1, 6});
    Tensor<float> weightTensor({1, 1, 3});
    Tensor<float> outputTensor({1, 1, 4});

    // Set gradient output values
    for(int i = 0; i < 4; ++i)
    {
        outputTensor.setHostValue(static_cast<float>(i + 1), 0, 0, i);
    }

    // Set weight values
    weightTensor.setHostValue(1.0f, 0, 0, 0);
    weightTensor.setHostValue(2.0f, 0, 0, 1);
    weightTensor.setHostValue(3.0f, 0, 0, 2);

    std::vector<int64_t> strides = {1};
    std::vector<int64_t> dilations = {1};
    std::vector<int64_t> padding = {0};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Verify gradient computation
    // Position 0: 1*1 = 1
    // Position 1: 1*2 + 2*1 = 4
    // Position 2: 1*3 + 2*2 + 3*1 = 10
    // Position 3: 2*3 + 3*2 + 4*1 = 16
    // Position 4: 3*3 + 4*2 = 17
    // Position 5: 4*3 = 12
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1), 4.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 2), 10.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 3), 16.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 4), 17.0f);
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 5), 12.0f);
}
TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdData1DPadding)
{
    // Test 1D backward data convolution with padding
    Tensor<float> inputTensor({1, 1, 5});
    Tensor<float> weightTensor({1, 1, 3});
    Tensor<float> outputTensor({1, 1, 5}); // Same size due to padding

    // Set gradient output values
    for(int i = 0; i < 5; ++i)
    {
        outputTensor.setHostValue(1.0f, 0, 0, i);
    }

    // Set weight values
    weightTensor.setHostValue(1.0f, 0, 0, 0);
    weightTensor.setHostValue(2.0f, 0, 0, 1);
    weightTensor.setHostValue(1.0f, 0, 0, 2);

    std::vector<int64_t> strides = {1};
    std::vector<int64_t> dilations = {1};
    std::vector<int64_t> padding = {1}; // Padding only in width

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    const std::vector<float> expectedGradients = {3.0f, 4.0f, 4.0f, 4.0f, 3.0f};

    for(int i = 0; i < 5; ++i)
    {
        float actual = inputTensor.getHostValue(0, 0, i);
        float expected = expectedGradients[static_cast<size_t>(i)];
        EXPECT_FLOAT_EQ(actual, expected) << "Mismatch at grad_X index [" << i << "]";
    }
}

TEST(TestCpuFpReferenceConvolutionBfp16, ConvolutionBwdData3D)
{
    // Test 3D backward data convolution with bfp16
    Tensor<hip_bfloat16> inputTensor({1, 1, 2, 2, 2});
    Tensor<hip_bfloat16> weightTensor({1, 1, 1, 1, 1});
    Tensor<hip_bfloat16> outputTensor({1, 1, 2, 2, 2});

    // Set gradient output values
    for(int i = 0; i < 8; ++i)
    {
        outputTensor.memory().hostData()[i] = static_cast<hip_bfloat16>(1.0f);
    }

    // Set weight value
    weightTensor.setHostValue(2.0_bf, 0, 0, 0, 0, 0);

    std::vector<int64_t> strides = {1, 1, 1};
    std::vector<int64_t> dilations = {1, 1, 1};
    std::vector<int64_t> padding = {0, 0, 0};

    CpuFpReferenceConvolutionImpl<hip_bfloat16, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Verify all values are 2.0 (gradOutput * weight)
    for(int i = 0; i < 8; ++i)
    {
        EXPECT_FLOAT_EQ(static_cast<float>(inputTensor.memory().hostData()[i]), 2.0f);
    }
}

TEST(TestCpuFpReferenceConvolutionBfp16, ConvolutionBwdData3DNdhwc)
{
    // Test 3D backward data convolution with bfp16
    Tensor<hip_bfloat16> inputTensor({1, 1, 2, 2, 2}, TensorLayout::NDHWC);
    Tensor<hip_bfloat16> weightTensor({1, 1, 1, 1, 1}, TensorLayout::NDHWC);
    Tensor<hip_bfloat16> outputTensor({1, 1, 2, 2, 2}, TensorLayout::NDHWC);

    // Set gradient output values
    for(int i = 0; i < 8; ++i)
    {
        outputTensor.memory().hostData()[i] = static_cast<hip_bfloat16>(1.0f);
    }

    // Set weight value
    weightTensor.setHostValue(2.0_bf, 0, 0, 0, 0, 0);

    std::vector<int64_t> strides = {1, 1, 1};
    std::vector<int64_t> dilations = {1, 1, 1};
    std::vector<int64_t> padding = {0, 0, 0};

    CpuFpReferenceConvolutionImpl<hip_bfloat16, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Verify all values are 2.0 (gradOutput * weight)
    for(int i = 0; i < 8; ++i)
    {
        EXPECT_FLOAT_EQ(static_cast<float>(inputTensor.memory().hostData()[i]), 2.0f);
    }
}

TEST(TestCpuFpReferenceConvolutionFp64, ConvolutionBwdData1D)
{
    // Test 1D backward data convolution with fp64
    Tensor<double> inputTensor({1, 1, 4});
    Tensor<double> weightTensor({1, 1, 2});
    Tensor<double> outputTensor({1, 1, 3});

    // Set gradient output values
    outputTensor.setHostValue(1.0, 0, 0, 0);
    outputTensor.setHostValue(2.0, 0, 0, 1);
    outputTensor.setHostValue(3.0, 0, 0, 2);

    // Set weight values
    weightTensor.setHostValue(0.5, 0, 0, 0);
    weightTensor.setHostValue(1.5, 0, 0, 1);

    std::vector<int64_t> strides = {1};
    std::vector<int64_t> dilations = {1};
    std::vector<int64_t> padding = {0};

    CpuFpReferenceConvolutionImpl<double, double>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Expected gradients
    EXPECT_DOUBLE_EQ(inputTensor.getHostValue(0, 0, 0), 0.5); // 1 * 0.5
    EXPECT_DOUBLE_EQ(inputTensor.getHostValue(0, 0, 1), 2.5); // 1 * 1.5 + 2 * 0.5
    EXPECT_DOUBLE_EQ(inputTensor.getHostValue(0, 0, 2), 4.5); // 2 * 1.5 + 3 * 0.5
    EXPECT_DOUBLE_EQ(inputTensor.getHostValue(0, 0, 3), 4.5); // 3 * 1.5
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdData2DAsymmetricPadding)
{
    // Test backward data convolution with asymmetric padding
    Tensor<float> inputTensor({1, 1, 3, 3});
    Tensor<float> weightTensor({1, 1, 2, 2});
    Tensor<float> outputTensor({1, 1, 4, 5}); // Gradient output with asymmetric size

    // Fill gradient output with simple pattern
    float gradValue = 1.0f;
    for(int h = 0; h < 4; ++h)
    {
        for(int w = 0; w < 5; ++w)
        {
            outputTensor.setHostValue(gradValue++, 0, 0, h, w);
        }
    }

    // Fill weights
    weightTensor.setHostValue(1.0f, 0, 0, 0, 0);
    weightTensor.setHostValue(2.0f, 0, 0, 0, 1);
    weightTensor.setHostValue(3.0f, 0, 0, 1, 0);
    weightTensor.setHostValue(4.0f, 0, 0, 1, 1);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> prePadding = {1, 1};
    std::vector<int64_t> postPadding = {1, 2};

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, prePadding, postPadding);

    const std::array<std::array<float, 3>, 3> expectedGradients
        = {{{29, 39, 49}, {79, 89, 99}, {129, 139, 149}}};

    for(int h = 0; h < 3; ++h)
    {
        for(int w = 0; w < 3; ++w)
        {
            float actual = inputTensor.getHostValue(0, 0, h, w);
            float expected = expectedGradients[static_cast<size_t>(h)][static_cast<size_t>(w)];
            EXPECT_FLOAT_EQ(actual, expected) << "Mismatch at grad_X(" << h << "," << w << ")";
        }
    }
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdData3DAsymmetricPadding)
{
    std::vector<int64_t> strides = {1, 1, 1};
    std::vector<int64_t> dilations = {1, 1, 1};
    std::vector<int64_t> prePadding = {1, 0, 1};
    std::vector<int64_t> postPadding = {1, 0, 2};

    // Test 3D backward data convolution with asymmetric padding
    // 1, 1, 4, 2, 5 with paddings
    Tensor<float> inputTensor({1, 1, 2, 2, 2});
    Tensor<float> weightTensor({1, 1, 2, 2, 2});
    Tensor<float> outputTensor({1, 1, 3, 1, 4}); // Gradient output with asymmetric size

    // Fill gradient output
    float gradValue = 1.0f;
    for(int d = 0; d < 3; ++d)
    {
        for(int h = 0; h < 1; ++h)
        {
            for(int w = 0; w < 4; ++w)
            {
                outputTensor.setHostValue(gradValue++, 0, 0, d, h, w);
            }
        }
    }

    // Fill weights with pattern
    float weightValue = 0.1f;
    for(int i = 0; i < 8; ++i)
    {
        weightTensor.memory().hostData()[i] = weightValue;
        weightValue += 0.1f;
    }

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
        inputTensor, weightTensor, outputTensor, strides, dilations, prePadding, postPadding);

    const std::array<std::array<std::array<float, 2>, 2>, 2> expectedGradients = {{
        // d=0
        {{{3.2f, 4.6f}, // h=0
          {6.0f, 8.2f}}}, // h=1
        // d=1
        {{{8.8f, 10.2f}, // h=0
          {14.8f, 17.0f}}} // h=1
    }};

    for(int d = 0; d < 2; ++d)
    {
        for(int h = 0; h < 2; ++h)
        {
            for(int w = 0; w < 2; ++w)
            {
                float actual = inputTensor.getHostValue(0, 0, d, h, w);
                float expected = expectedGradients[static_cast<size_t>(d)][static_cast<size_t>(h)]
                                                  [static_cast<size_t>(w)];
                EXPECT_FLOAT_EQ(actual, expected)
                    << "Mismatch at grad_X(" << d << "," << h << "," << w << ")";
            }
        }
    }
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionFwd3DInvalidOutputDim)
{
    std::vector<int64_t> strides = {1, 1, 1};
    std::vector<int64_t> dilations = {1, 1, 1};
    std::vector<int64_t> prePadding = {1, 0, 1};
    std::vector<int64_t> postPadding = {1, 0, 2};

    // Test 3D forward convolution with asymmetric padding
    // The output tensor's height (2) is invalid for the given input (height 2) and kernel (height 2).
    // The expected output height is 1. This test verifies that an exception is thrown.
    Tensor<float> inputTensor({1, 1, 2, 2, 2});
    Tensor<float> weightTensor({1, 1, 2, 2, 2});
    Tensor<float> outputTensor({1, 1, 3, 2, 4}); // Output with invalid height dimension

    // Fill tensors with dummy values
    for(int i = 0; i < 8; ++i)
    {
        inputTensor.memory().hostData()[i] = 1.0f;
        weightTensor.memory().hostData()[i] = 1.0f;
    }

    EXPECT_THROW(
        (CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
            inputTensor, weightTensor, outputTensor, strides, dilations, prePadding, postPadding)),
        std::invalid_argument);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdData3DInvalidOutputDim)
{
    std::vector<int64_t> strides = {1, 1, 1};
    std::vector<int64_t> dilations = {1, 1, 1};
    std::vector<int64_t> prePadding = {1, 0, 1};
    std::vector<int64_t> postPadding = {1, 0, 2};

    // Test 3D backward data convolution with asymmetric padding
    // The output tensor's height (2) is invalid for the given input (height 2) and kernel (height 2).
    // The expected output height is 1. This test verifies that an exception is thrown.
    Tensor<float> inputTensor({1, 1, 2, 2, 2});
    Tensor<float> weightTensor({1, 1, 2, 2, 2});
    Tensor<float> outputTensor({1, 1, 3, 2, 4}); // Gradient output with invalid height dimension

    EXPECT_THROW(
        (CpuFpReferenceConvolutionImpl<float, float>::convBwdData(
            inputTensor, weightTensor, outputTensor, strides, dilations, prePadding, postPadding)),
        std::invalid_argument);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdWeight3DInvalidOutputDim)
{
    std::vector<int64_t> strides = {1, 1, 1};
    std::vector<int64_t> dilations = {1, 1, 1};
    std::vector<int64_t> prePadding = {1, 0, 1};
    std::vector<int64_t> postPadding = {1, 0, 2};

    // Test 3D backward weight convolution with asymmetric padding
    // The output tensor's height (2) is invalid for the given input (height 2) and kernel (height 2).
    // The expected output height is 1. This test verifies that an exception is thrown.
    Tensor<float> inputTensor({1, 1, 2, 2, 2});
    Tensor<float> weightTensor({1, 1, 2, 2, 2});
    Tensor<float> outputTensor({1, 1, 3, 2, 4}); // Gradient output with invalid height dimension

    EXPECT_THROW(
        (CpuFpReferenceConvolutionImpl<float, float>::convBwdWeight(
            inputTensor, weightTensor, outputTensor, strides, dilations, prePadding, postPadding)),
        std::invalid_argument);
}
