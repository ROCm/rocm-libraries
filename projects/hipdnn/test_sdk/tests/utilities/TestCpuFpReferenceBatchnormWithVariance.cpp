// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_data_sdk/utilities/UtilsFp16.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;

TEST(TestCpuFpReferenceBatchnormWithVarianceFp32, BatchnormFwdInferenceWithVarianceNchw)
{
    Tensor<float> inputTensor({1, 3, 224, 224});
    Tensor<float> outputTensor({1, 3, 224, 224});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceBfp16, BatchnormFwdInferenceWithVarianceNchw)
{
    Tensor<hip_bfloat16> inputTensor({1, 3, 224, 224});
    Tensor<hip_bfloat16> outputTensor({1, 3, 224, 224});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp16, BatchnormFwdInferenceWithVarianceNchw)
{
    Tensor<half> inputTensor({1, 3, 224, 224});
    Tensor<half> outputTensor({1, 3, 224, 224});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp64, BatchnormFwdInferenceWithVarianceNchw)
{
    Tensor<double> inputTensor({1, 3, 224, 224});
    Tensor<double> outputTensor({1, 3, 224, 224});
    Tensor<double> biasTensor({1, 3});
    Tensor<double> scaleTensor({1, 3});
    Tensor<double> meanTensor({1, 3});
    Tensor<double> varianceTensor({1, 3});

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp32, BatchnormFwdInferenceWithVarianceNhwc)
{
    Tensor<float> inputTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<float> outputTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp64, BatchnormFwdInferenceWithVarianceSanityNchw)
{
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor({1, 1});
    Tensor<double> biasTensor({1, 1});
    Tensor<double> meanTensor({1, 1});
    Tensor<double> varianceTensor({1, 1});

    // x = [1, 2, 3, 4]
    inputTensor.setHostValue(1.0, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0, 0, 0, 1, 1);

    // fixed scale and bias parameters (one channel)
    scaleTensor.setHostValue(2.0, 0, 0);
    biasTensor.setHostValue(0.5, 0, 0);

    // inference uses population statistics per channel:
    // mean = (1+2+3+4)/4 = 2.5
    // variance = [(-1.5)^2 + (-0.5)^2 + (0.5)^2 + (1.5)^2] / 4 = 5.0 / 4 = 1.25
    // inv_variance = 1 / sqrt(1.25 + 1e-5) = 0.894423613312618
    //
    // With variance input, we compute inv_variance from variance internally
    meanTensor.setHostValue(2.5, 0, 0);
    varianceTensor.setHostValue(1.25, 0, 0);

    // output is calculated via a pointwise linear transform on x:
    // y = scale * (x - mean) * inv_variance + bias = 2 * (x - 2.5) * inv_variance + 0.5
    const std::vector<double> expectedOutput = {-2.18327084, -0.39442361, 1.39442361, 3.18327084};

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);

    auto tolerance = 1e-6;

    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), expectedOutput[0], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), expectedOutput[1], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), expectedOutput[2], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), expectedOutput[3], tolerance);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp32, CompareVarianceVsInvVarianceImplementationsNchw)
{
    // This test verifies that fwdInferenceWithVariance produces the same results
    // as fwdInference when given variance vs inv_variance inputs
    const std::vector<int64_t> dims = {2, 3, 4, 4};

    Tensor<float> inputTensor(dims);
    Tensor<float> outputFromVariance(dims);
    Tensor<float> outputFromInvVariance(dims);
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});
    Tensor<float> invVarianceTensor({1, 3});

    // Initialize with random values
    inputTensor.fillWithRandomValues(-5.0f, 5.0f, 42);

    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(1.5f + (static_cast<float>(i) * 0.5f), 0, i);
        biasTensor.setHostValue(0.5f - (static_cast<float>(i) * 0.2f), 0, i);
        meanTensor.setHostValue(1.0f + (static_cast<float>(i) * 0.3f), 0, i);

        // Set variance and compute corresponding inv_variance
        auto var = 2.0f + (static_cast<float>(i) * 0.5f);
        varianceTensor.setHostValue(var, 0, i);
        auto invVar = 1.0f / std::sqrt(var + static_cast<float>(BATCHNORM_DEFAULT_EPSILON));
        invVarianceTensor.setHostValue(invVar, 0, i);
    }

    // Call both implementations
    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputFromVariance);

    CpuFpReferenceBatchnorm::fwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, invVarianceTensor, outputFromInvVariance);

    // Compare results
    auto tolerance = 1e-5f;
    for(int b = 0; b < 2; b++)
    {
        for(int c = 0; c < 3; c++)
        {
            for(int h = 0; h < 4; h++)
            {
                for(int w = 0; w < 4; w++)
                {
                    auto valFromVariance = outputFromVariance.getHostValue(b, c, h, w);
                    auto valFromInvVariance = outputFromInvVariance.getHostValue(b, c, h, w);
                    EXPECT_NEAR(valFromVariance, valFromInvVariance, tolerance)
                        << "Mismatch at [" << b << "," << c << "," << h << "," << w << "]";
                }
            }
        }
    }
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp32, BatchnormFwdInferenceWithVariance2D)
{
    // Test with 2D tensor (batch, channel)
    Tensor<float> inputTensor({4, 3});
    Tensor<float> outputTensor({4, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    inputTensor.fillWithValue(1.0f);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(1.0f, 0, i);
        biasTensor.setHostValue(0.0f, 0, i);
        meanTensor.setHostValue(1.0f, 0, i);
        varianceTensor.setHostValue(1.0f, 0, i);
    }

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp32, BatchnormFwdInferenceWithVariance3D)
{
    // Test with 3D tensor (batch, channel, length)
    Tensor<float> inputTensor({2, 3, 10});
    Tensor<float> outputTensor({2, 3, 10});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    inputTensor.fillWithValue(2.0f);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(2.0f, 0, i);
        biasTensor.setHostValue(1.0f, 0, i);
        meanTensor.setHostValue(2.0f, 0, i);
        varianceTensor.setHostValue(1.0f, 0, i);
    }

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp32, BatchnormFwdInferenceWithVarianceNcdhw)
{
    // Test with 5D tensor (batch, channel, depth, height, width)
    Tensor<float> inputTensor({2, 3, 4, 5, 6});
    Tensor<float> outputTensor({2, 3, 4, 5, 6});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    inputTensor.fillWithValue(1.5f);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(1.0f, 0, i);
        biasTensor.setHostValue(0.5f, 0, i);
        meanTensor.setHostValue(1.5f, 0, i);
        varianceTensor.setHostValue(0.5f, 0, i);
    }

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp32, ZeroVarianceHandling)
{
    // Test edge case: zero variance (relies on epsilon)
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<float> inputTensor(dims);
    Tensor<float> outputTensor(dims);
    Tensor<float> scaleTensor({1, 1});
    Tensor<float> biasTensor({1, 1});
    Tensor<float> meanTensor({1, 1});
    Tensor<float> varianceTensor({1, 1});

    // All input values identical
    inputTensor.setHostValue(3.0f, 0, 0, 0, 0);
    inputTensor.setHostValue(3.0f, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    inputTensor.setHostValue(3.0f, 0, 0, 1, 1);

    scaleTensor.setHostValue(2.0f, 0, 0);
    biasTensor.setHostValue(0.5f, 0, 0);
    meanTensor.setHostValue(3.0f, 0, 0);
    varianceTensor.setHostValue(0.0f, 0, 0); // Zero variance

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);

    // When variance is 0, inv_variance = 1/sqrt(epsilon)
    // For all elements: y = 2.0 * (3.0 - 3.0) * (1/sqrt(epsilon)) + 0.5 = 0.5
    auto tolerance = 1e-5f;
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), 0.5f, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), 0.5f, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), 0.5f, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), 0.5f, tolerance);
}
