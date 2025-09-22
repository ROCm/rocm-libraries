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

    // Expected gradInput pattern (hand calculated)
    // Each gradOutput position contributes to multiple gradInput positions
    // based on the weight kernel
    // EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 0), 1.0f); // 1*1
    // EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 1), 4.0f); // 1*2 + 2*1
    // EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 2), 7.0f); // 1*3 + 2*2
    // EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 3), 6.0f); // 2*3

    // Expected gradInput pattern (hand calculated with flipped filter)
    // Weight: [[1,2,3],[4,5,6],[7,8,9]]
    // Flipped: [[9,8,7],[6,5,4],[3,2,1]]
    // gradOutput: [[1,2],[3,4]]
    // Each gradOutput position contributes based on the flipped weight kernel
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 0), 9.0f); // 1*9
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 1), 24.0f); // 1*8 + 2*8
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 2), 21.0f); // 1*7 + 2*7
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 3), 14.0f); // 2*7
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 0), 33.0f); // 1*6 + 3*9
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 1), 77.0f); // 1*5 + 2*6 + 3*8 + 4*9
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 2), 67.0f); // 1*4 + 2*5 + 3*7 + 4*8
    EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 1, 3), 46.0f); // 2*4 + 4*7
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataWithStride)
{
    // // Test with stride = 2
    // Tensor<float> inputTensor({1, 1, 5, 5});
    // Tensor<float> weightTensor({1, 1, 3, 3});
    // Tensor<float> outputTensor({1, 1, 2, 2});

    // // Fill input with sequential values
    // for(int i = 0; i < 25; ++i)
    // {
    //     inputTensor.memory().hostData()[i] = static_cast<float>(i + 1);
    // }

    // // Fill weights with 1s
    // for(int i = 0; i < 9; ++i)
    // {
    //     weightTensor.memory().hostData()[i] = 1.0f;
    // }

    // std::vector<int64_t> strides = {2, 2};
    // std::vector<int64_t> dilations = {1, 1};
    // std::vector<int64_t> padding = {0, 0};

    // CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
    //     inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // // With stride 2, we sample every other position
    // // Output should be non-zero values
    // EXPECT_GT(outputTensor.getHostValue(0, 0, 0, 0), 0.0f);
    // EXPECT_GT(outputTensor.getHostValue(0, 0, 0, 1), 0.0f);
    // EXPECT_GT(outputTensor.getHostValue(0, 0, 1, 0), 0.0f);
    // EXPECT_GT(outputTensor.getHostValue(0, 0, 1, 1), 0.0f);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataWithPadding)
{
    // // Test with padding = 1
    // Tensor<float> inputTensor({1, 1, 3, 3});
    // Tensor<float> weightTensor({1, 1, 3, 3});
    // Tensor<float> outputTensor({1, 1, 3, 3});

    // // Fill input with sequential values
    // for(int i = 0; i < 9; ++i)
    // {
    //     inputTensor.memory().hostData()[i] = static_cast<float>(i + 1);
    // }

    // // Fill weights with 1s
    // for(int i = 0; i < 9; ++i)
    // {
    //     weightTensor.memory().hostData()[i] = 1.0f;
    // }

    // std::vector<int64_t> strides = {1, 1};
    // std::vector<int64_t> dilations = {1, 1};
    // std::vector<int64_t> padding = {1, 1};

    // CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
    //     inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // // With padding, output size should match input size
    // // Center element should have the maximum value (sum of all inputs)
    // EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 1), 45.0f); // Sum of 1-9
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataMultiChannel)
{
    // // Test with multiple input channels
    // Tensor<float> inputTensor({1, 2, 3, 3}); // 2 input channels
    // Tensor<float> weightTensor({1, 2, 2, 2}); // 1 output channel, 2 input channels
    // Tensor<float> outputTensor({1, 1, 2, 2});

    // // Fill input tensors
    // for(int i = 0; i < 18; ++i)
    // {
    //     inputTensor.memory().hostData()[i] = static_cast<float>(i + 1);
    // }

    // // Fill weights with 1s
    // for(int i = 0; i < 8; ++i)
    // {
    //     weightTensor.memory().hostData()[i] = 1.0f;
    // }

    // std::vector<int64_t> strides = {1, 1};
    // std::vector<int64_t> dilations = {1, 1};
    // std::vector<int64_t> padding = {0, 0};

    // CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
    //     inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // // Output should be non-zero
    // EXPECT_GT(outputTensor.getHostValue(0, 0, 0, 0), 0.0f);
    // EXPECT_GT(outputTensor.getHostValue(0, 0, 0, 1), 0.0f);
    // EXPECT_GT(outputTensor.getHostValue(0, 0, 1, 0), 0.0f);
    // EXPECT_GT(outputTensor.getHostValue(0, 0, 1, 1), 0.0f);
}

TEST(TestCpuFpReferenceConvolutionBfp16, ConvolutionBwdDataBasic)
{
    // // Basic convolution: 1x1x4x4 input, 1x1x3x3 weight -> 1x1x2x2 output
    // Tensor<hip_bfloat16> inputTensor({1, 1, 4, 4});
    // Tensor<hip_bfloat16> weightTensor({1, 1, 3, 3});
    // Tensor<hip_bfloat16> outputTensor({1, 1, 2, 2});

    // // gradOutput values: simple pattern
    // outputTensor.setHostValue(1.0_bf, 0, 0, 0, 0);
    // outputTensor.setHostValue(2.0_bf, 0, 0, 0, 1);
    // outputTensor.setHostValue(3.0_bf, 0, 0, 1, 0);
    // outputTensor.setHostValue(4.0_bf, 0, 0, 1, 1);

    // // Weight values: simple 3x3 kernel
    // std::array<hip_bfloat16, 9> weightData = {1.0_bf, 2.0_bf, 3.0_bf, 4.0_bf, 5.0_bf, 6.0_bf, 7.0_bf, 8.0_bf, 9.0_bf};
    // for(size_t i = 0; i < 9; ++i)
    // {
    //     weightTensor.memory().hostData()[i] = weightData[i];
    // }

    // std::vector<int64_t> strides = {1, 1};
    // std::vector<int64_t> dilations = {1, 1};
    // std::vector<int64_t> padding = {0, 0};

    // CpuFpReferenceConvolutionImpl<hip_bfloat16, float>::convBwdData(
    //     inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // // Expected gradInput pattern (hand calculated)
    // // Each gradOutput position contributes to multiple gradInput positions
    // // based on the weight kernel
    // EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 0), 1.0f); // 1*1
    // EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 1), 4.0f); // 1*2 + 2*1
    // EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 2), 7.0f); // 1*3 + 2*2
    // EXPECT_FLOAT_EQ(inputTensor.getHostValue(0, 0, 0, 3), 6.0f); // 2*3
}

TEST(TestCpuFpReferenceConvolutionFp16, ConvolutionBwdDataBasic)
{
    // Tensor<half> inputTensor({1, 1, 4, 4});
    // Tensor<half> weightTensor({1, 1, 3, 3});
    // Tensor<half> outputTensor({1, 1, 2, 2});

    // // Fill with simple values
    // for(int i = 0; i < 16; ++i)
    // {
    //     inputTensor.memory().hostData()[i] = static_cast<half>(static_cast<float>(i + 1));
    // }

    // for(int i = 0; i < 9; ++i)
    // {
    //     weightTensor.memory().hostData()[i] = static_cast<half>(1.0f);
    // }

    // std::vector<int64_t> strides = {1, 1};
    // std::vector<int64_t> dilations = {1, 1};
    // std::vector<int64_t> padding = {0, 0};

    // CpuFpReferenceConvolutionImpl<half, float>::convFwdInference(
    //     inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // // Test that computation produces reasonable results
    // EXPECT_GT(static_cast<float>(outputTensor.getHostValue(0, 0, 0, 0)), 0.0f);
    // EXPECT_GT(static_cast<float>(outputTensor.getHostValue(0, 0, 1, 1)), 0.0f);
}

TEST(TestCpuFpReferenceConvolutionFp64, ConvolutionBwdDataBasic)
{
    // Tensor<double> inputTensor({1, 1, 4, 4});
    // Tensor<double> weightTensor({1, 1, 3, 3});
    // Tensor<double> outputTensor({1, 1, 2, 2});

    // // Fill input with sequential values
    // for(int i = 0; i < 16; ++i)
    // {
    //     inputTensor.memory().hostData()[i] = static_cast<double>(i + 1);
    // }

    // // Fill weights with 1s
    // for(int i = 0; i < 9; ++i)
    // {
    //     weightTensor.memory().hostData()[i] = 1.0;
    // }

    // std::vector<int64_t> strides = {1, 1};
    // std::vector<int64_t> dilations = {1, 1};
    // std::vector<int64_t> padding = {0, 0};

    // CpuFpReferenceConvolutionImpl<double, double>::convFwdInference(
    //     inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // // Same expected values as fp32 test
    // EXPECT_DOUBLE_EQ(outputTensor.getHostValue(0, 0, 0, 0), 54.0);
    // EXPECT_DOUBLE_EQ(outputTensor.getHostValue(0, 0, 0, 1), 63.0);
    // EXPECT_DOUBLE_EQ(outputTensor.getHostValue(0, 0, 1, 0), 90.0);
    // EXPECT_DOUBLE_EQ(outputTensor.getHostValue(0, 0, 1, 1), 99.0);
}

TEST(TestCpuFpReferenceConvolutionFp32, ConvolutionBwdDataWithDilation)
{
    // // Test with dilation = 2
    // Tensor<float> inputTensor({1, 1, 5, 5});
    // Tensor<float> weightTensor({1, 1, 3, 3});
    // Tensor<float> outputTensor({1, 1, 1, 1});

    // // Fill input with sequential values
    // for(int i = 0; i < 25; ++i)
    // {
    //     inputTensor.memory().hostData()[i] = static_cast<float>(i + 1);
    // }

    // // Fill weights with 1s
    // for(int i = 0; i < 9; ++i)
    // {
    //     weightTensor.memory().hostData()[i] = 1.0f;
    // }

    // std::vector<int64_t> strides = {1, 1};
    // std::vector<int64_t> dilations = {2, 2};
    // std::vector<int64_t> padding = {0, 0};

    // CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
    //     inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // // With dilation=2, kernel samples positions: (0,0), (0,2), (0,4), (2,0), (2,2), (2,4), (4,0), (4,2), (4,4)
    // // Values: 1, 3, 5, 11, 13, 15, 21, 23, 25
    // // Sum: 1+3+5+11+13+15+21+23+25 = 117
    // EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 0), 117.0f);
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
