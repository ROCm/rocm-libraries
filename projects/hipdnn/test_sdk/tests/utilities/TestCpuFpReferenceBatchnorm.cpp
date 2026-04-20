// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

#ifndef HIPDNN_DATA_SDK_SKIP_JSON_LIB
#include "GoldenReferenceCpu.hpp"
#endif

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using hipdnn_test_sdk::detail::safeTestTypeCast;

#ifndef HIPDNN_DATA_SDK_SKIP_JSON_LIB

template <class T>
class TestCpuBatchnormFwdInferenceGoldenReference : public TestGoldenReferenceCpu
{
public:
    void testSuite()
    {
        return goldenReferenceTestSuite(batchnorm::getToleranceInference<T>(),
                                        batchnorm::getToleranceInference<T>());
    }
};

class TestCpuBatchnormFwdInferenceGoldenReferenceNchwFp32
    : public TestCpuBatchnormFwdInferenceGoldenReference<float>
{
};

class TestCpuBatchnormFwdInferenceGoldenReferenceNchwFp16
    : public TestCpuBatchnormFwdInferenceGoldenReference<half>
{
};

class TestCpuBatchnormFwdInferenceGoldenReferenceNchwBfp16
    : public TestCpuBatchnormFwdInferenceGoldenReference<bfloat16>
{
};

class TestCpuBatchnormFwdInferenceGoldenReferenceNcdhwFp32
    : public TestCpuBatchnormFwdInferenceGoldenReference<float>
{
};

// Nchw Fp32------------
TEST_P(TestCpuBatchnormFwdInferenceGoldenReferenceNchwFp32, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestCpuBatchnormFwdInferenceGoldenReferenceNchwFp32,
                         getGoldenReferenceParams("BatchnormFwdInference/nchw/fp32"));

// Nchw Fp16------------
TEST_P(TestCpuBatchnormFwdInferenceGoldenReferenceNchwFp16, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestCpuBatchnormFwdInferenceGoldenReferenceNchwFp16,
                         getGoldenReferenceParams("BatchnormFwdInference/nchw/fp16"));

// Nchw Bfp16------------
TEST_P(TestCpuBatchnormFwdInferenceGoldenReferenceNchwBfp16, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestCpuBatchnormFwdInferenceGoldenReferenceNchwBfp16,
                         getGoldenReferenceParams("BatchnormFwdInference/nchw/bfp16"));

// Ncdhw Fp32------------
TEST_P(TestCpuBatchnormFwdInferenceGoldenReferenceNcdhwFp32, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestCpuBatchnormFwdInferenceGoldenReferenceNcdhwFp32,
                         getGoldenReferenceParams("BatchnormFwdInference/ncdhw/fp32"));

#endif // HIPDNN_DATA_SDK_SKIP_JSON_LIB

//--------------------------

template <typename T1, typename T2>
struct TypePair
{
    using First = T1;
    using Second = T2;
};

using TypesFwdInferenceNchw = ::testing::Types<TypePair<float, float>,
                                               TypePair<half, float>,
                                               TypePair<bfloat16, float>,
                                               TypePair<double, double>,
                                               TypePair<int8_t, float>,
                                               TypePair<fp8_e4m3, float>,
                                               TypePair<fp8_e5m2, float>>;

template <class T>
class CpuFpReferenceBatchnormFwdInferenceNchw : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceBatchnormFwdInferenceNchw, TypesFwdInferenceNchw, );

TYPED_TEST(CpuFpReferenceBatchnormFwdInferenceNchw, BatchnormFwdInferenceNchw)
{
    const Tensor<typename TypeParam::First> inputTensor({1, 3, 224, 224});
    Tensor<typename TypeParam::First> outputTensor({1, 3, 224, 224});
    const Tensor<typename TypeParam::Second> biasTensor({1, 3});
    const Tensor<typename TypeParam::Second> scaleTensor({1, 3});
    const Tensor<typename TypeParam::Second> meanTensor({1, 3});
    const Tensor<typename TypeParam::Second> varianceTensor({1, 3});

    CpuFpReferenceBatchnorm::fwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);
}

using TypesFwdInferenceNhwc = ::testing::Types<TypePair<float, float>, TypePair<half, bfloat16>>;

template <class T>
class CpuFpReferenceBatchnormFwdInferenceNhwc : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceBatchnormFwdInferenceNhwc, TypesFwdInferenceNhwc, );

TYPED_TEST(CpuFpReferenceBatchnormFwdInferenceNhwc, BatchnormFwdInferenceNhwc)
{
    const Tensor<typename TypeParam::First> inputTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<typename TypeParam::First> outputTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    const Tensor<typename TypeParam::Second> biasTensor({1, 3});
    const Tensor<typename TypeParam::Second> scaleTensor({1, 3});
    const Tensor<float> meanTensor({1, 3});
    const Tensor<float> varianceTensor({1, 3});

    CpuFpReferenceBatchnorm::fwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);
}

TEST(TestCpuFpReferenceBatchnormFp64, BatchnormFwdInferenceSanityValidationNchw)
{
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor({1, 1});
    Tensor<double> biasTensor({1, 1});
    Tensor<double> meanTensor({1, 1});
    Tensor<double> invVarianceTensor({1, 1});

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
    // where inv_variance = 1 / sqrt(1.25 + 1e-5) = 0.894423613312618
    // (in practice, computed during training)
    meanTensor.setHostValue(2.5, 0, 0);
    invVarianceTensor.setHostValue(0.894423613312618, 0, 0);

    // output is calculated via a pointwise linear transform on x:
    // y = scale * (x - mean) * inv_variance + bias = 2 * (x - 2.5) * inv_variance + 0.5
    const std::vector<double> expectedOutput = {-2.18327084, -0.39442361, 1.39442361, 3.18327084};

    CpuFpReferenceBatchnorm::fwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, invVarianceTensor, outputTensor);

    auto tolerance = 1e-6;

    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), expectedOutput[0], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), expectedOutput[1], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), expectedOutput[2], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), expectedOutput[3], tolerance);
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormFwdInference2D)
{
    // Test with 2D tensor (batch, channel)
    Tensor<float> inputTensor({4, 3});
    Tensor<float> outputTensor({4, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> invVarianceTensor({1, 3});

    inputTensor.fillWithValue(1.0f);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(1.0f, 0, i);
        biasTensor.setHostValue(0.0f, 0, i);
        meanTensor.setHostValue(1.0f, 0, i);
        invVarianceTensor.setHostValue(1.0f, 0, i);
    }

    CpuFpReferenceBatchnorm::fwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, invVarianceTensor, outputTensor);
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormFwdInference3D)
{
    // Test with 3D tensor (batch, channel, length)
    Tensor<float> inputTensor({2, 3, 10});
    Tensor<float> outputTensor({2, 3, 10});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    // Initialize tensors with test data
    inputTensor.fillWithValue(2.0f);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(2.0f, 0, i);
        biasTensor.setHostValue(1.0f, 0, i);
        meanTensor.setHostValue(2.0f, 0, i);
        varianceTensor.setHostValue(1.0f, 0, i);
    }

    CpuFpReferenceBatchnorm::fwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormFwdInferenceNcdhw)
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

    CpuFpReferenceBatchnorm::fwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);
}

TEST(TestCpuFpReferenceBatchnormBfp16, BatchnormFwdInferenceNdhwc)
{
    Tensor<float> inputTensor({2, 3, 4, 5, 6}, TensorLayout::NDHWC);
    Tensor<float> outputTensor({2, 3, 4, 5, 6}, TensorLayout::NDHWC);
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    inputTensor.fillWithValue(1.5f);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(2.0f, 0, i);
        biasTensor.setHostValue(0.5f, 0, i);
        meanTensor.setHostValue(1.5f, 0, i);
        varianceTensor.setHostValue(1.0f, 0, i);
    }

    CpuFpReferenceBatchnorm::fwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);
}

template <typename T1, typename T2, typename T3>
struct TypeTriplet
{
    using First = T1;
    using Second = T2;
    using Third = T3;
};

using TypesBackwardNchw = ::testing::Types<TypeTriplet<float, float, float>,
                                           TypeTriplet<bfloat16, float, float>,
                                           TypeTriplet<half, float, float>,
                                           TypeTriplet<int8_t, float, float>,
                                           TypeTriplet<fp8_e4m3, float, float>,
                                           TypeTriplet<fp8_e5m2, float, float>,
                                           TypeTriplet<half, bfloat16, float>,
                                           TypeTriplet<double, double, double>>;

template <typename T>
class CpuFpReferenceBatchnormBackwardNchw : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceBatchnormBackwardNchw, TypesBackwardNchw, );

TYPED_TEST(CpuFpReferenceBatchnormBackwardNchw, BatchnormBackwardNchw)
{
    const Tensor<typename TypeParam::First> xTensor({6, 3, 32, 32});
    const Tensor<typename TypeParam::First> dyTensor({6, 3, 32, 32});
    Tensor<typename TypeParam::First> dxTensor({6, 3, 32, 32});
    const Tensor<typename TypeParam::Second> scaleTensor({1, 3});
    const Tensor<typename TypeParam::Third> meanTensor({1, 3});
    const Tensor<typename TypeParam::Third> invVarianceTensor({1, 3});
    Tensor<typename TypeParam::Second> dscaleTensor({1, 3});
    Tensor<typename TypeParam::Second> dbiasTensor({1, 3});

    CpuFpReferenceBatchnorm::backward(dyTensor,
                                      xTensor,
                                      meanTensor,
                                      invVarianceTensor,
                                      scaleTensor,
                                      dxTensor,
                                      dscaleTensor,
                                      dbiasTensor);
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormBackwardNhwc)
{
    const Tensor<float> xTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    const Tensor<float> dyTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<float> dxTensor({6, 3, 32, 32});
    const Tensor<float> scaleTensor({1, 3});
    const Tensor<float> meanTensor({1, 3});
    const Tensor<float> invVarianceTensor({1, 3});
    Tensor<float> dscaleTensor({1, 3});
    Tensor<float> dbiasTensor({1, 3});

    CpuFpReferenceBatchnorm::backward(dyTensor,
                                      xTensor,
                                      meanTensor,
                                      invVarianceTensor,
                                      scaleTensor,
                                      dxTensor,
                                      dscaleTensor,
                                      dbiasTensor);
}

TEST(TestCpuFpReferenceBatchnormFp64, BatchnormBwdSanityValidationNchw)
{
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> xTensor(dims);
    Tensor<double> dyTensor(dims);
    Tensor<double> dxTensor(dims);
    Tensor<double> scaleTensor({1, 1});
    Tensor<double> meanTensor({1, 1});
    Tensor<double> invVarianceTensor({1, 1});
    Tensor<double> dscaleTensor({1, 1});
    Tensor<double> dbiasTensor({1, 1});

    // x = [1, 2, 3, 4]
    xTensor.setHostValue(1.0, 0, 0, 0, 0);
    xTensor.setHostValue(2.0, 0, 0, 0, 1);
    xTensor.setHostValue(3.0, 0, 0, 1, 0);
    xTensor.setHostValue(4.0, 0, 0, 1, 1);

    // gradient dy = [0.1, 0.2, 0.3, 0.4]
    dyTensor.setHostValue(0.1, 0, 0, 0, 0);
    dyTensor.setHostValue(0.2, 0, 0, 0, 1);
    dyTensor.setHostValue(0.3, 0, 0, 1, 0);
    dyTensor.setHostValue(0.4, 0, 0, 1, 1);

    // scale (one channel) = 2.0
    scaleTensor.setHostValue(2.0, 0, 0);

    // 1 batch, so compute mean and variance over all elements
    // mean = (1+2+3+4)/4 = 2.5
    // variance = [(-1.5)^2 + (-0.5)^2 + (0.5)^2 + (1.5)^2] / 4 = 5.0 / 4 = 1.25
    // inv_variance = 1 / sqrt(1.25 + 1e-5) = 0.894423613312618
    meanTensor.setHostValue(2.5, 0, 0);
    invVarianceTensor.setHostValue(0.894423613312618, 0, 0);

    // dbias = sum(dy) = 0.1 + 0.2 + 0.3 + 0.4 = 1.0
    auto expectedDbias = 1.0;

    // x_hat = (x - mean) * inv_variance = [-1.34163542 -0.44721181  0.44721181  1.34163542]
    // dscale = sum(dy * x_hat) = sum([-1.34163542 -0.44721181  0.44721181  1.34163542]) = 0.447211806656309
    auto expectedDscale = 0.447211806656309;

    // dx is calculated pointwise via the full backward formula
    // dx = scale * inv_variance * (dy - mean(dy) - x_hat * dscale / 4)
    std::vector<double> expectedDx
        = {-2.14659950e-06, -7.15533166e-07, 7.15533166e-07, 2.14659950e-06};

    CpuFpReferenceBatchnorm::backward(dyTensor,
                                      xTensor,
                                      scaleTensor,
                                      dxTensor,
                                      dscaleTensor,
                                      dbiasTensor,
                                      &meanTensor,
                                      &invVarianceTensor);

    auto tolerance = 1e-6;

    EXPECT_NEAR(dbiasTensor.getHostValue(0, 0), expectedDbias, tolerance);
    EXPECT_NEAR(dscaleTensor.getHostValue(0, 0), expectedDscale, tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 0, 0), expectedDx[0], tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 0, 1), expectedDx[1], tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 1, 0), expectedDx[2], tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 1, 1), expectedDx[3], tolerance);
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormBackward2D)
{
    // Test with 2D tensor (batch, channel)
    Tensor<float> xTensor({4, 3});
    Tensor<float> dyTensor({4, 3});
    Tensor<float> dxTensor({4, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> invVarianceTensor({1, 3});
    Tensor<float> dscaleTensor({1, 3});
    Tensor<float> dbiasTensor({1, 3});

    xTensor.fillWithValue(1.0f);
    dyTensor.fillWithValue(0.1f);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(1.0f, 0, i);
        meanTensor.setHostValue(1.0f, 0, i);
        invVarianceTensor.setHostValue(1.0f, 0, i);
    }

    CpuFpReferenceBatchnorm::backward(dyTensor,
                                      xTensor,
                                      meanTensor,
                                      invVarianceTensor,
                                      scaleTensor,
                                      dxTensor,
                                      dscaleTensor,
                                      dbiasTensor);
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormBackward3D)
{
    // Test with 3D tensor (batch, channel, length)
    Tensor<float> xTensor({2, 3, 10});
    Tensor<float> dyTensor({2, 3, 10});
    Tensor<float> dxTensor({2, 3, 10});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> invVarianceTensor({1, 3});
    Tensor<float> dscaleTensor({1, 3});
    Tensor<float> dbiasTensor({1, 3});

    xTensor.fillWithValue(2.0f);
    dyTensor.fillWithValue(0.2f);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(2.0f, 0, i);
        meanTensor.setHostValue(2.0f, 0, i);
        invVarianceTensor.setHostValue(0.5f, 0, i);
    }

    CpuFpReferenceBatchnorm::backward(dyTensor,
                                      xTensor,
                                      meanTensor,
                                      invVarianceTensor,
                                      scaleTensor,
                                      dxTensor,
                                      dscaleTensor,
                                      dbiasTensor);
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormBackwardNcdhw)
{
    // Test with 5D tensor (batch, channel, depth, height, width)
    Tensor<float> xTensor({2, 3, 4, 5, 6});
    Tensor<float> dyTensor({2, 3, 4, 5, 6});
    Tensor<float> dxTensor({2, 3, 4, 5, 6});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> invVarianceTensor({1, 3});
    Tensor<float> dscaleTensor({1, 3});
    Tensor<float> dbiasTensor({1, 3});

    xTensor.fillWithValue(1.5f);
    dyTensor.fillWithValue(0.1f);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(1.0f, 0, i);
        meanTensor.setHostValue(1.5f, 0, i);
        invVarianceTensor.setHostValue(0.7071f, 0, i); // 1/sqrt(2)
    }

    CpuFpReferenceBatchnorm::backward(dyTensor,
                                      xTensor,
                                      meanTensor,
                                      invVarianceTensor,
                                      scaleTensor,
                                      dxTensor,
                                      dscaleTensor,
                                      dbiasTensor);
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormBackwardNdhwc)
{
    Tensor<float> xTensor({2, 3, 4, 5, 6}, TensorLayout::NDHWC);
    Tensor<float> dyTensor({2, 3, 4, 5, 6}, TensorLayout::NDHWC);
    Tensor<float> dxTensor({2, 3, 4, 5, 6}, TensorLayout::NDHWC);
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> invVarianceTensor({1, 3});
    Tensor<float> dscaleTensor({1, 3});
    Tensor<float> dbiasTensor({1, 3});

    xTensor.fillWithValue(1.5f);
    dyTensor.fillWithValue(0.1f);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(1.0f, 0, i);
        meanTensor.setHostValue(1.5f, 0, i);
        invVarianceTensor.setHostValue(0.7071f, 0, i); // 1/sqrt(2)
    }

    CpuFpReferenceBatchnorm::backward(dyTensor,
                                      xTensor,
                                      meanTensor,
                                      invVarianceTensor,
                                      scaleTensor,
                                      dxTensor,
                                      dscaleTensor,
                                      dbiasTensor);
}

TEST(TestCpuFpReferenceBatchnormFp64, BatchnormBwdSanityValidationNchwWithoutSavedStats)
{
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> xTensor(dims);
    Tensor<double> dyTensor(dims);
    Tensor<double> dxTensor(dims);
    Tensor<double> scaleTensor({1, 1});
    Tensor<double> dscaleTensor({1, 1});
    Tensor<double> dbiasTensor({1, 1});

    // x = [1, 2, 3, 4]
    xTensor.setHostValue(1.0, 0, 0, 0, 0);
    xTensor.setHostValue(2.0, 0, 0, 0, 1);
    xTensor.setHostValue(3.0, 0, 0, 1, 0);
    xTensor.setHostValue(4.0, 0, 0, 1, 1);

    // gradient dy = [0.1, 0.2, 0.3, 0.4]
    dyTensor.setHostValue(0.1, 0, 0, 0, 0);
    dyTensor.setHostValue(0.2, 0, 0, 0, 1);
    dyTensor.setHostValue(0.3, 0, 0, 1, 0);
    dyTensor.setHostValue(0.4, 0, 0, 1, 1);

    // scale (one channel) = 2.0
    scaleTensor.setHostValue(2.0, 0, 0);

    // When mean and invVariance are not provided, they are computed from x:
    // mean = (1+2+3+4)/4 = 2.5
    // variance = [(-1.5)^2 + (-0.5)^2 + (0.5)^2 + (1.5)^2] / 4 = 1.25
    // inv_variance = 1 / sqrt(1.25 + 1e-5) = 0.894423613312618

    // Expected values should be identical to BatchnormBwdSanityValidationNchw
    auto expectedDbias = 1.0;
    auto expectedDscale = 0.447211806656309;
    std::vector<double> expectedDx
        = {-2.14659950e-06, -7.15533166e-07, 7.15533166e-07, 2.14659950e-06};

    CpuFpReferenceBatchnorm::backward<double, double, double, double, double, double>(
        dyTensor,
        xTensor,
        scaleTensor,
        dxTensor,
        dscaleTensor,
        dbiasTensor,
        nullptr,
        nullptr,
        BATCHNORM_DEFAULT_EPSILON);

    auto tolerance = 1e-6;

    EXPECT_NEAR(dbiasTensor.getHostValue(0, 0), expectedDbias, tolerance);
    EXPECT_NEAR(dscaleTensor.getHostValue(0, 0), expectedDscale, tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 0, 0), expectedDx[0], tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 0, 1), expectedDx[1], tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 1, 0), expectedDx[2], tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 1, 1), expectedDx[3], tolerance);
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormBackwardWithoutSavedStatsSingleElement)
{
    // Edge case: single element per channel so variance = 0
    const std::vector<int64_t> dims = {1, 1, 1, 1};

    Tensor<float> xTensor(dims);
    Tensor<float> dyTensor(dims);
    Tensor<float> dxTensor(dims);
    Tensor<float> scaleTensor({1, 1});
    Tensor<float> dscaleTensor({1, 1});
    Tensor<float> dbiasTensor({1, 1});

    // x = [5.0]
    xTensor.setHostValue(5.0f, 0, 0, 0, 0);
    dyTensor.setHostValue(0.3f, 0, 0, 0, 0);
    scaleTensor.setHostValue(1.5f, 0, 0);

    // Computed statistics:
    // mean = 5.0
    // variance = 0.0
    // inv_variance = 1 / sqrt(0.0 + 1e-5) = 316.227766...

    // dbias = sum(dy) = 0.3
    // x_hat = (5.0 - 5.0) * inv_variance = 0
    // dscale = sum(dy * x_hat) = 0.3 * 0 = 0
    // dx = scale * inv_variance * (dy - mean(dy) - x_hat * mean(dy * x_hat))
    //    = 1.5 * inv_variance * (0.3 - 0.3 - 0 * 0) = 0

    CpuFpReferenceBatchnorm::backward<float, float, float, float, float, float>(
        dyTensor, xTensor, scaleTensor, dxTensor, dscaleTensor, dbiasTensor, nullptr, nullptr);

    auto tolerance = 1e-5f;

    EXPECT_NEAR(dbiasTensor.getHostValue(0, 0), 0.3f, tolerance);
    EXPECT_NEAR(dscaleTensor.getHostValue(0, 0), 0.0f, tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 0, 0), 0.0f, tolerance);
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormBackwardWithoutSavedStatsZeroVariance)
{
    // Edge case: all identical values so variance = 0
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<float> xTensor(dims);
    Tensor<float> dyTensor(dims);
    Tensor<float> dxTensor(dims);
    Tensor<float> scaleTensor({1, 1});
    Tensor<float> dscaleTensor({1, 1});
    Tensor<float> dbiasTensor({1, 1});

    // x = [3.0, 3.0, 3.0, 3.0]
    xTensor.setHostValue(3.0f, 0, 0, 0, 0);
    xTensor.setHostValue(3.0f, 0, 0, 0, 1);
    xTensor.setHostValue(3.0f, 0, 0, 1, 0);
    xTensor.setHostValue(3.0f, 0, 0, 1, 1);

    dyTensor.setHostValue(0.1f, 0, 0, 0, 0);
    dyTensor.setHostValue(0.2f, 0, 0, 0, 1);
    dyTensor.setHostValue(0.3f, 0, 0, 1, 0);
    dyTensor.setHostValue(0.4f, 0, 0, 1, 1);

    scaleTensor.setHostValue(2.0f, 0, 0);

    // Computed statistics:
    // mean = 3.0
    // variance = 0.0
    // inv_variance = 1 / sqrt(epsilon) = 1 / sqrt(1e-5) = 316.227766...

    // dbias = sum(dy) = 1.0
    // x_hat = (x - 3.0) * inv_variance = 0 (all elements)
    // dscale = sum(dy * x_hat) = 0

    // dx calculation:
    // mean(dy) = 1.0 / 4 = 0.25
    // inv_variance = 1 / sqrt(1e-5) = 316.227766
    // dx = scale * inv_variance * (dy - mean(dy) - x_hat * mean(dy * x_hat))
    //    = 2.0 * 316.227766 * (dy - 0.25)
    //    = 632.455532 * (dy - 0.25)

    // dx[0] = 632.455532 * (0.1 - 0.25) = -94.86833
    // dx[1] = 632.455532 * (0.2 - 0.25) = -31.62278
    // dx[2] = 632.455532 * (0.3 - 0.25) = 31.62278
    // dx[3] = 632.455532 * (0.4 - 0.25) = 94.86833

    CpuFpReferenceBatchnorm::backward<float, float, float, float, float, float>(
        dyTensor, xTensor, scaleTensor, dxTensor, dscaleTensor, dbiasTensor, nullptr, nullptr);

    auto tolerance = 1e-4f;

    EXPECT_NEAR(dbiasTensor.getHostValue(0, 0), 1.0f, tolerance);
    EXPECT_NEAR(dscaleTensor.getHostValue(0, 0), 0.0f, tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 0, 0), -94.86833f, tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 0, 1), -31.62278f, tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 1, 0), 31.62278f, tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 1, 1), 94.86833f, tolerance);
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormBackwardWithoutSavedStats2DMinimal)
{
    // Edge case: minimum valid 2D tensor (batch, channel only - no spatial dims)
    const std::vector<int64_t> dims = {2, 1};

    Tensor<float> xTensor(dims);
    Tensor<float> dyTensor(dims);
    Tensor<float> dxTensor(dims);
    Tensor<float> scaleTensor({1, 1});
    Tensor<float> dscaleTensor({1, 1});
    Tensor<float> dbiasTensor({1, 1});

    // x = [1.0, 3.0]
    xTensor.setHostValue(1.0f, 0, 0);
    xTensor.setHostValue(3.0f, 1, 0);

    dyTensor.setHostValue(0.2f, 0, 0);
    dyTensor.setHostValue(0.4f, 1, 0);

    scaleTensor.setHostValue(1.0f, 0, 0);

    // Computed statistics:
    // mean = (1.0 + 3.0) / 2 = 2.0
    // variance = [(1.0-2.0)^2 + (3.0-2.0)^2] / 2 = (1.0 + 1.0) / 2 = 1.0
    // inv_variance = 1 / sqrt(1.0 + 1e-5) ≈ 0.9999950000374998

    // dbias = sum(dy) = 0.6
    // x_hat = [(1.0-2.0)*inv_var, (3.0-2.0)*inv_var] ≈ [-0.999995, 0.999995]
    // dscale = sum(dy * x_hat) ≈ 0.2*(-0.999995) + 0.4*(0.999995) ≈ 0.199999

    CpuFpReferenceBatchnorm::backward<float, float, float, float, float, float>(
        dyTensor, xTensor, scaleTensor, dxTensor, dscaleTensor, dbiasTensor, nullptr, nullptr);

    auto tolerance = 1e-4f;

    EXPECT_NEAR(dbiasTensor.getHostValue(0, 0), 0.6f, tolerance);
    EXPECT_NEAR(dscaleTensor.getHostValue(0, 0), 0.199999f, tolerance);
    // dx values are small due to mean subtraction in backward formula
    EXPECT_TRUE(std::isfinite(dxTensor.getHostValue(0, 0)));
    EXPECT_TRUE(std::isfinite(dxTensor.getHostValue(1, 0)));
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormBackwardWithoutSavedStatsLargeEpsilon)
{
    // Edge case: test with large epsilon to ensure it affects invVariance computation
    const std::vector<int64_t> dims = {1, 1, 1, 1};

    Tensor<float> xTensor(dims);
    Tensor<float> dyTensor(dims);
    Tensor<float> dxTensor(dims);
    Tensor<float> scaleTensor({1, 1});
    Tensor<float> dscaleTensor({1, 1});
    Tensor<float> dbiasTensor({1, 1});

    xTensor.setHostValue(5.0f, 0, 0, 0, 0);
    dyTensor.setHostValue(0.5f, 0, 0, 0, 0);
    scaleTensor.setHostValue(1.0f, 0, 0);

    const double largeEpsilon = 1.0;

    // Computed statistics with large epsilon:
    // mean = 5.0
    // variance = 0.0
    // inv_variance = 1 / sqrt(0.0 + 1.0) = 1.0

    CpuFpReferenceBatchnorm::backward<float, float, float, float, float, float>(dyTensor,
                                                                                xTensor,
                                                                                scaleTensor,
                                                                                dxTensor,
                                                                                dscaleTensor,
                                                                                dbiasTensor,
                                                                                nullptr,
                                                                                nullptr,
                                                                                largeEpsilon);

    auto tolerance = 1e-5f;

    EXPECT_NEAR(dbiasTensor.getHostValue(0, 0), 0.5f, tolerance);
    EXPECT_NEAR(dscaleTensor.getHostValue(0, 0), 0.0f, tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 0, 0), 0.0f, tolerance);
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormFwdTrainingNchwBasic)
{
    Tensor<float> inputTensor({2, 3, 4, 4});
    Tensor<float> outputTensor({2, 3, 4, 4});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> biasTensor({1, 3});

    inputTensor.fillWithValue(1.0f);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(1.0f, 0, i);
        biasTensor.setHostValue(0.0f, 0, i);
    }

    // Call without optional tensors
    CpuFpReferenceBatchnorm::fwdTraining(
        inputTensor, scaleTensor, biasTensor, outputTensor, BATCHNORM_DEFAULT_EPSILON, 0.1);
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormFwdTrainingNchwWithSavedStats)
{
    Tensor<float> inputTensor({2, 3, 4, 4});
    Tensor<float> outputTensor({2, 3, 4, 4});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> savedMean({1, 3});
    Tensor<float> savedInvVariance({1, 3});

    inputTensor.fillWithValue(1.0f);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(1.0f, 0, i);
        biasTensor.setHostValue(0.0f, 0, i);
    }

    // Call with saved statistics for backward pass
    CpuFpReferenceBatchnorm::fwdTraining(inputTensor,
                                         scaleTensor,
                                         biasTensor,
                                         outputTensor,
                                         BATCHNORM_DEFAULT_EPSILON,
                                         0.1,
                                         &savedMean,
                                         &savedInvVariance);

    // Verify saved statistics were populated
    for(int i = 0; i < 3; i++)
    {
        EXPECT_NE(savedMean.getHostValue(0, i), 0.0f);
        EXPECT_GT(savedInvVariance.getHostValue(0, i), 0.0f);
    }
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormFwdTrainingNchwWithRunningStats)
{
    Tensor<float> inputTensor({2, 3, 4, 4});
    Tensor<float> outputTensor({2, 3, 4, 4});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> prevRunningMean({1, 3});
    Tensor<float> prevRunningVariance({1, 3});
    Tensor<float> nextRunningMean({1, 3});
    Tensor<float> nextRunningVariance({1, 3});

    inputTensor.fillWithValue(1.0f);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(1.0f, 0, i);
        biasTensor.setHostValue(0.0f, 0, i);
        prevRunningMean.setHostValue(0.5f, 0, i);
        prevRunningVariance.setHostValue(1.0f, 0, i);
    }

    const double momentum = 0.1;

    // Cannot deduce template with partial nullptr params
    CpuFpReferenceBatchnorm::fwdTraining<float, float, float, float, float>(
        inputTensor,
        scaleTensor,
        biasTensor,
        outputTensor,
        BATCHNORM_DEFAULT_EPSILON,
        momentum,
        nullptr,
        nullptr,
        &prevRunningMean,
        &prevRunningVariance,
        &nextRunningMean,
        &nextRunningVariance);

    // Verify running statistics were updated
    for(int i = 0; i < 3; i++)
    {
        // Next running mean should be different from previous
        EXPECT_NE(nextRunningMean.getHostValue(0, i), prevRunningMean.getHostValue(0, i));
        // Next running variance should be different from previous
        EXPECT_NE(nextRunningVariance.getHostValue(0, i), prevRunningVariance.getHostValue(0, i));
    }
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormFwdTrainingNchwFullFeatures)
{
    Tensor<float> inputTensor({2, 3, 4, 4});
    Tensor<float> outputTensor({2, 3, 4, 4});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> savedMean({1, 3});
    Tensor<float> savedInvVariance({1, 3});
    Tensor<float> prevRunningMean({1, 3});
    Tensor<float> prevRunningVariance({1, 3});
    Tensor<float> nextRunningMean({1, 3});
    Tensor<float> nextRunningVariance({1, 3});

    outputTensor.fillWithValue(-10.0f);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(2.0f, 0, i);
        biasTensor.setHostValue(0.5f, 0, i);
        prevRunningMean.setHostValue(0.0f, 0, i);
        prevRunningVariance.setHostValue(1.0f, 0, i);
    }

    // Initialize input with known values
    for(int b = 0; b < 2; b++)
    {
        for(int c = 0; c < 3; c++)
        {
            for(int h = 0; h < 4; h++)
            {
                for(int w = 0; w < 4; w++)
                {
                    inputTensor.setHostValue(static_cast<float>(b + c + h + w), b, c, h, w);
                }
            }
        }
    }

    // Call with all optional parameters
    CpuFpReferenceBatchnorm::fwdTraining(inputTensor,
                                         scaleTensor,
                                         biasTensor,
                                         outputTensor,
                                         BATCHNORM_DEFAULT_EPSILON,
                                         0.1,
                                         &savedMean,
                                         &savedInvVariance,
                                         &prevRunningMean,
                                         &prevRunningVariance,
                                         &nextRunningMean,
                                         &nextRunningVariance);

    // Verify all outputs were populated
    for(int i = 0; i < 3; i++)
    {
        EXPECT_GT(savedInvVariance.getHostValue(0, i), 0.0f);
        EXPECT_NE(nextRunningMean.getHostValue(0, i), prevRunningMean.getHostValue(0, i));
        EXPECT_NE(nextRunningVariance.getHostValue(0, i), prevRunningVariance.getHostValue(0, i));
    }

    for(int b = 0; b < 2; b++)
    {
        for(int c = 0; c < 3; c++)
        {
            for(int h = 0; h < 4; h++)
            {
                for(int w = 0; w < 4; w++)
                {
                    EXPECT_NE(outputTensor.getHostValue(b, c, h, w), -10.0f);
                }
            }
        }
    }
}

using TypesFwdTrainingNhwc = ::testing::Types<TypeTriplet<bfloat16, float, float>,
                                              TypeTriplet<half, float, float>,
                                              TypeTriplet<double, double, double>,
                                              TypeTriplet<int8_t, double, double>,
                                              TypeTriplet<fp8_e4m3, float, float>,
                                              TypeTriplet<fp8_e5m2, float, float>,
                                              TypeTriplet<half, bfloat16, float>>;

template <typename T>
class CpuFpReferenceBatchnromFwdTrainingNchw : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceBatchnromFwdTrainingNchw, TypesFwdTrainingNhwc, );

TYPED_TEST(CpuFpReferenceBatchnromFwdTrainingNchw, BatchnormFwdTrainingNchw)
{
    Tensor<typename TypeParam::First> inputTensor({2, 3, 4, 4});
    Tensor<typename TypeParam::First> outputTensor({2, 3, 4, 4});
    Tensor<typename TypeParam::Second> scaleTensor({1, 3});
    Tensor<typename TypeParam::Second> biasTensor({1, 3});
    Tensor<typename TypeParam::Third> savedMean({1, 3});
    Tensor<typename TypeParam::Third> savedInvVariance({1, 3});

    inputTensor.fillWithValue(safeTestTypeCast<typename TypeParam::First>(1.0));
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(safeTestTypeCast<typename TypeParam::Second>(1.0), 0, i);
        biasTensor.setHostValue(safeTestTypeCast<typename TypeParam::Second>(0.0), 0, i);
    }

    CpuFpReferenceBatchnorm::fwdTraining(inputTensor,
                                         scaleTensor,
                                         biasTensor,
                                         outputTensor,
                                         BATCHNORM_DEFAULT_EPSILON,
                                         0.1,
                                         &savedMean,
                                         &savedInvVariance);
}

TEST(TestCpuFpReferenceBatchnormFp64, BatchnormFwdTrainingSanityValidationNchw)
{
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor({1, 1});
    Tensor<double> biasTensor({1, 1});
    Tensor<double> savedMean({1, 1});
    Tensor<double> savedInvVariance({1, 1});
    Tensor<double> prevRunningMean({1, 1});
    Tensor<double> prevRunningVariance({1, 1});
    Tensor<double> nextRunningMean({1, 1});
    Tensor<double> nextRunningVariance({1, 1});

    // x = [1, 2, 3, 4]
    inputTensor.setHostValue(1.0, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0, 0, 0, 1, 1);

    // fixed scale and bias parameters
    scaleTensor.setHostValue(2.0, 0, 0);
    biasTensor.setHostValue(0.5, 0, 0);

    // Initialize running statistics
    prevRunningMean.setHostValue(0.0, 0, 0);
    prevRunningVariance.setHostValue(1.0, 0, 0);

    const double epsilon = BATCHNORM_DEFAULT_EPSILON;
    const double momentum = 0.1;

    // During training, batch statistics are calculated:
    // mean = (1+2+3+4)/4 = 2.5
    // variance = [(-1.5)^2 + (-0.5)^2 + (0.5)^2 + (1.5)^2] / 4 = 1.25
    // invVariance = 1 / sqrt(1.25 + 1e-5) = 0.894423613312618
    // y = scale * (x - mean) * invVariance + bias
    const std::vector<double> expectedOutput = {-2.18327084, -0.39442361, 1.39442361, 3.18327084};

    CpuFpReferenceBatchnorm::fwdTraining(inputTensor,
                                         scaleTensor,
                                         biasTensor,
                                         outputTensor,
                                         epsilon,
                                         momentum,
                                         &savedMean,
                                         &savedInvVariance,
                                         &prevRunningMean,
                                         &prevRunningVariance,
                                         &nextRunningMean,
                                         &nextRunningVariance);

    auto tolerance = 1e-6;

    // Verify output values
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), expectedOutput[0], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), expectedOutput[1], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), expectedOutput[2], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), expectedOutput[3], tolerance);

    // Verify saved statistics
    EXPECT_NEAR(savedMean.getHostValue(0, 0), 2.5, tolerance);
    EXPECT_NEAR(savedInvVariance.getHostValue(0, 0), 0.894423613312618, tolerance);

    // Verify running statistics update
    // newRunningMean = (1 - 0.1) * 0.0 + 0.1 * 2.5 = 0.25
    EXPECT_NEAR(nextRunningMean.getHostValue(0, 0), 0.25, tolerance);

    // Bessel's correction: adjustedVariance = 1.25 * (4 / 3) = 1.66666...
    // newRunningVariance = (1 - 0.1) * 1.0 + 0.1 * 1.66666... = 0.9 + 0.166666... = 1.066666...
    EXPECT_NEAR(nextRunningVariance.getHostValue(0, 0), 1.06666666666667, tolerance);
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormFwdTraining2D)
{
    // Test with 2D tensor (batch, channel)
    Tensor<float> inputTensor({4, 3});
    Tensor<float> outputTensor({4, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> biasTensor({1, 3});

    inputTensor.fillWithValue(1.0);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(1.0f, 0, i);
        biasTensor.setHostValue(0.0f, 0, i);
    }

    CpuFpReferenceBatchnorm::fwdTraining(
        inputTensor, scaleTensor, biasTensor, outputTensor, BATCHNORM_DEFAULT_EPSILON, 0.1);
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormFwdTraining3D)
{
    // Test with 3D tensor (batch, channel, length)
    Tensor<float> inputTensor({2, 3, 10});
    Tensor<float> outputTensor({2, 3, 10});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> biasTensor({1, 3});

    inputTensor.fillWithValue(1.0);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(1.0f, 0, i);
        biasTensor.setHostValue(0.0f, 0, i);
    }

    CpuFpReferenceBatchnorm::fwdTraining(
        inputTensor, scaleTensor, biasTensor, outputTensor, BATCHNORM_DEFAULT_EPSILON, 0.1);
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormFwdTrainingNcdhw)
{
    // Test with 5D tensor (batch, channel, depth, height, width)
    Tensor<float> inputTensor({2, 3, 4, 5, 6});
    Tensor<float> outputTensor({2, 3, 4, 5, 6});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> savedMean({1, 3});
    Tensor<float> savedInvVariance({1, 3});

    inputTensor.fillWithValue(1.0);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(1.0f, 0, i);
        biasTensor.setHostValue(0.0f, 0, i);
    }

    CpuFpReferenceBatchnorm::fwdTraining(inputTensor,
                                         scaleTensor,
                                         biasTensor,
                                         outputTensor,
                                         BATCHNORM_DEFAULT_EPSILON,
                                         0.1,
                                         &savedMean,
                                         &savedInvVariance);
}

TEST(TestCpuFpReferenceBatchnormFp32, BatchnormFwdTrainingNdhwc)
{
    // Test with 5D tensor in NDHWC layout
    Tensor<float> inputTensor({2, 3, 4, 5, 6}, TensorLayout::NDHWC);
    Tensor<float> outputTensor({2, 3, 4, 5, 6}, TensorLayout::NDHWC);
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> savedMean({1, 3});
    Tensor<float> savedInvVariance({1, 3});

    inputTensor.fillWithRandomValues(-1.0f, 1.0f, 123);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(1.0f, 0, i);
        biasTensor.setHostValue(0.0f, 0, i);
    }

    CpuFpReferenceBatchnorm::fwdTraining(inputTensor,
                                         scaleTensor,
                                         biasTensor,
                                         outputTensor,
                                         BATCHNORM_DEFAULT_EPSILON,
                                         0.1,
                                         &savedMean,
                                         &savedInvVariance);
}
