/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt-ext-op.h>
#include <hipblaslt/host_validation/HipblasltDataInitialization.hpp>
#include <hipblaslt/host_validation/Types.hpp>
#include <roc/host_validation/validation.hpp>
#include <vector>

#include "hipblaslt_arguments.hpp"

enum class amaxInitMethod
{
    hpl = 111,
    nan = 222,
    max = 333,
    min = 444
};

struct AMaxTestData
{
    hipDataType type;
    hipDataType dtype;
    uint32_t    m;
    uint32_t    n;
};

class ExtOpSoftmaxTest : public testing::TestWithParam<uint32_t>
{
};
class ExtOpSoftmaxUnsupportedDatatypeTest : public testing::TestWithParam<hipDataType>
{
};

class ExtOpLayerNormTest : public testing::TestWithParam<uint32_t>
{
};
class ExtOpLayerNormUnsupportedDatatypeTest : public testing::TestWithParam<hipDataType>
{
};

class ExtOpAMaxTest : public testing::TestWithParam<AMaxTestData>
{
};
class ExtOpAMaxUnsupportedDatatypeTest : public testing::TestWithParam<hipDataType>
{
};

TEST_P(ExtOpSoftmaxTest, softmaxSuccess)
{
    uint32_t           m = GetParam();
    uint32_t           n = 16;
    std::vector<float> input(m * n, 0.f);
    std::vector<float> output(m * n, 0.f);
    hipblaslt::host_validation::initialize(
        input.data(), input.size(), hipblaslt_initialization::rand_int);
    float* gpuInput{};
    float* gpuOutput{};

    auto err          = hipMalloc(&gpuInput, m * n * sizeof(float));
    err               = hipMalloc(&gpuOutput, m * n * sizeof(float));
    err               = hipMemcpyHtoD(gpuInput, input.data(), m * n * sizeof(float));
    auto hipblasltErr = hipblasltExtSoftmax(HIP_R_32F, m, n, 1, gpuOutput, gpuInput, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_SUCCESS);
    err = hipDeviceSynchronize();
    ASSERT_EQ(err, hipSuccess);
    err = hipMemcpyDtoH(output.data(), gpuOutput, m * n * sizeof(float));
    ASSERT_EQ(err, hipSuccess);

    using namespace roc::host_validation;
    using namespace hipblaslt::host_validation;
    Tensor expected(ScalarType::Float32, Shape{m, n});
    referenceSoftmax(SoftmaxProblem(
        tensorFromStorage(input.data(), input.size(), Layout::contiguous(Shape{m, n})),
        expected,
        1,
        ScalarType::Float32));
    const ComparisonResult comparison
        = compare(tensorFromStorage(output.data(), output.size(), Layout::contiguous(Shape{m, n})),
                  expected,
                  nearComparisonOptions(1e-5));
    EXPECT_TRUE(comparison.passed());

    err = hipFree(gpuInput);
    err = hipFree(gpuOutput);
}

TEST_P(ExtOpLayerNormTest, layernormSuccess)
{
    uint32_t m = GetParam();
    uint32_t n = 16;

    std::vector<float> output(m * n, 0.f);
    std::vector<float> mean(m, 0.f);
    std::vector<float> invvar(m, 0.f);
    std::vector<float> input(m * n, 0.f);
    std::vector<float> gamma(n, 1.f);
    std::vector<float> beta(n, 0.f);

    hipblaslt::host_validation::initialize(
        input.data(), input.size(), hipblaslt_initialization::hpl);
    hipblaslt::host_validation::initialize(
        gamma.data(), gamma.size(), hipblaslt_initialization::hpl);
    hipblaslt::host_validation::initialize(beta.data(), beta.size(), hipblaslt_initialization::hpl);

    float* gpuOutput{};
    float* gpuMean{};
    float* gpuInvvar{};
    float* gpuInput{};
    float* gpuGamma{};
    float* gpuBeta{};

    auto err = hipMalloc(&gpuOutput, m * n * sizeof(float));
    err      = hipMalloc(&gpuMean, m * sizeof(float));
    err      = hipMalloc(&gpuInvvar, m * sizeof(float));
    err      = hipMalloc(&gpuInput, m * n * sizeof(float));
    err      = hipMalloc(&gpuGamma, n * sizeof(float));
    err      = hipMalloc(&gpuBeta, n * sizeof(float));

    err = hipMemcpyHtoD(gpuInput, input.data(), m * n * sizeof(float));
    err = hipMemcpyHtoD(gpuGamma, gamma.data(), n * sizeof(float));
    err = hipMemcpyHtoD(gpuBeta, beta.data(), n * sizeof(float));

    auto hipblasltErr = hipblasltExtLayerNorm(HIP_R_32F,
                                              gpuOutput,
                                              gpuMean,
                                              gpuInvvar,
                                              gpuInput,
                                              m,
                                              n,
                                              1e-05,
                                              gpuGamma,
                                              gpuBeta,
                                              nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_SUCCESS);
    err = hipDeviceSynchronize();
    ASSERT_EQ(err, hipSuccess);

    err = hipMemcpyDtoH(output.data(), gpuOutput, m * n * sizeof(float));
    err = hipMemcpyDtoH(mean.data(), gpuMean, m * sizeof(float));
    err = hipMemcpyDtoH(invvar.data(), gpuInvvar, m * sizeof(float));

    using namespace roc::host_validation;
    using namespace hipblaslt::host_validation;
    const Layout tensorLayout     = Layout::contiguous(Shape{m, n});
    const Layout statisticsLayout = Layout::contiguous(Shape{m});
    const Layout affineLayout     = Layout::contiguous(Shape{n});
    const Tensor referenceOutputTensor(ScalarType::Float32, tensorLayout);
    const Tensor referenceMeanTensor(ScalarType::Float32, statisticsLayout);
    const Tensor referenceInverseVarianceTensor(ScalarType::Float32, statisticsLayout);

    LayerNormProblem problem(tensorFromStorage(input.data(), input.size(), tensorLayout),
                             referenceOutputTensor,
                             1,
                             ScalarType::Float32);
    problem.mean            = referenceMeanTensor;
    problem.inverseVariance = referenceInverseVarianceTensor;
    problem.gamma           = tensorFromStorage(gamma.data(), gamma.size(), affineLayout);
    problem.beta            = tensorFromStorage(beta.data(), beta.size(), affineLayout);
    problem.epsilon         = 1e-5;
    referenceLayerNorm(problem);

    const ComparisonOptions comparisonOptions = nearComparisonOptions(1e-5);
    const ComparisonResult  outputComparison
        = compare(tensorFromStorage(output.data(), output.size(), tensorLayout),
                  referenceOutputTensor,
                  comparisonOptions);
    EXPECT_TRUE(outputComparison.passed())
        << "LayerNorm output mismatches: " << outputComparison.mismatches
        << ", max absolute difference: " << outputComparison.maxAbsoluteDifference;

    const ComparisonResult meanComparison
        = compare(tensorFromStorage(mean.data(), mean.size(), statisticsLayout),
                  referenceMeanTensor,
                  comparisonOptions);
    EXPECT_TRUE(meanComparison.passed())
        << "LayerNorm mean mismatches: " << meanComparison.mismatches
        << ", max absolute difference: " << meanComparison.maxAbsoluteDifference;

    const ComparisonResult inverseVarianceComparison
        = compare(tensorFromStorage(invvar.data(), invvar.size(), statisticsLayout),
                  referenceInverseVarianceTensor,
                  comparisonOptions);
    EXPECT_TRUE(inverseVarianceComparison.passed())
        << "LayerNorm inverse-variance mismatches: " << inverseVarianceComparison.mismatches
        << ", max absolute difference: " << inverseVarianceComparison.maxAbsoluteDifference;

    err = hipFree(gpuOutput);
    err = hipFree(gpuMean);
    err = hipFree(gpuInvvar);
    err = hipFree(gpuInput);
    err = hipFree(gpuGamma);
    err = hipFree(gpuBeta);
}

template <typename Ti, typename To>
void AMaxTest(hipDataType type, hipDataType dtype, std::size_t m, std::size_t n)
{
    std::size_t numElements = m * n;
    std::size_t inNumBytes  = sizeof(Ti);
    std::size_t outNumBytes = sizeof(To);

    To* gpuOutput{nullptr};
    Ti* gpuInput{nullptr};

    auto hipErr = hipMalloc(&gpuOutput, outNumBytes);
    hipErr      = hipMalloc(&gpuInput, m * n * inNumBytes);

    std::vector<To> cpuOutput(1, 0.f);
    std::vector<Ti> cpuInput(m * n, 0.f);
    std::vector<To> refOutput(1, 0.f);

    hipblaslt::host_validation::initialize(
        cpuInput.data(), cpuInput.size(), hipblaslt_initialization::hpl);

    hipErr = hipMemcpyHtoD(gpuInput, cpuInput.data(), m * n * inNumBytes);

    hipStream_t stream{};
    hipErr            = hipStreamCreate(&stream);
    auto hipblasltErr = hipblasltExtAMax(type, dtype, gpuOutput, gpuInput, m, n, stream);

    hipErr = hipMemcpyDtoH(cpuOutput.data(), gpuOutput, outNumBytes);

    using namespace roc::host_validation;
    Tensor referenceOutput = hipblaslt::host_validation::tensorFromMutableStorage(
        refOutput.data(), refOutput.size(), Layout::contiguous(Shape{}));
    referenceMaximumAbsolute(
        hipblaslt::host_validation::tensorFromStorage(
            cpuInput.data(), cpuInput.size(), Layout::contiguous(Shape{numElements})),
        referenceOutput,
        ScalarType::Float32);
    hipblaslt::host_validation::copyTensorStorageTo(
        refOutput.data(), refOutput.size(), referenceOutput);

    EXPECT_NEAR(float(refOutput[0]), float(cpuOutput[0]), 1e-5);

    hipErr = hipStreamDestroy(stream);
    hipErr = hipFree(gpuOutput);
    hipErr = hipFree(gpuInput);
}

TEST_P(ExtOpAMaxTest, amaxSuccess)
{
    AMaxTestData testdata = GetParam();

    if(testdata.type == HIP_R_32F && testdata.dtype == HIP_R_32F)
    {
        AMaxTest<float, float>(testdata.type, testdata.dtype, testdata.m, testdata.n);
    }
    else if(testdata.type == HIP_R_32F && testdata.dtype == HIP_R_16F)
    {
        AMaxTest<float, hipblasLtHalf>(testdata.type, testdata.dtype, testdata.m, testdata.n);
    }
    else if(testdata.type == HIP_R_16F && testdata.dtype == HIP_R_32F)
    {
        AMaxTest<hipblasLtHalf, float>(testdata.type, testdata.dtype, testdata.m, testdata.n);
    }
    else if(testdata.type == HIP_R_16F && testdata.dtype == HIP_R_16F)
    {
        AMaxTest<hipblasLtHalf, hipblasLtHalf>(
            testdata.type, testdata.dtype, testdata.m, testdata.n);
    }
}

TEST_P(ExtOpSoftmaxUnsupportedDatatypeTest, softmaxFailureUnsupportedDatatype)
{
    auto hipblasltErr = hipblasltExtSoftmax(GetParam(), 16, 16, 1, nullptr, nullptr, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_NOT_SUPPORTED);
}

TEST(ExtOpTest, softmaxFailureUnsupportedShapeOrReductionDim)
{
    auto hipblasltErr = hipblasltExtSoftmax(HIP_R_32F, 16, 512, 1, nullptr, nullptr, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_INVALID_VALUE);
    hipblasltErr = hipblasltExtSoftmax(HIP_R_32F, 16, 16, 0, nullptr, nullptr, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_NOT_SUPPORTED);
}

TEST_P(ExtOpLayerNormUnsupportedDatatypeTest, layernormFailureUnsupportedDatatype)
{
    auto hipblasltErr = hipblasltExtLayerNorm(
        GetParam(), nullptr, nullptr, nullptr, nullptr, 16, 1024, 1e-05, nullptr, nullptr, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_NOT_SUPPORTED);
}

TEST(ExtOpTest, layernormFailureInvalidValue)
{
    auto hipblasltErr = hipblasltExtLayerNorm(
        HIP_R_32F, nullptr, nullptr, nullptr, nullptr, 16, 1024, 1e-05, nullptr, nullptr, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_P(ExtOpAMaxUnsupportedDatatypeTest, amaxFailureUnsupportedDatatype)
{
    auto hipblasltErr = hipblasltExtAMax(GetParam(), GetParam(), nullptr, nullptr, 0, 0, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_NOT_SUPPORTED);
}

TEST(ExtOpTest, amaxFailureInvalidValue)
{
    auto hipblasltErr = hipblasltExtAMax(HIP_R_32F, HIP_R_32F, nullptr, nullptr, 0, 0, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_INVALID_VALUE);
}

INSTANTIATE_TEST_SUITE_P(ExtOpTest, ExtOpSoftmaxTest, testing::Values<uint32_t>(1, 16, 1335));
INSTANTIATE_TEST_SUITE_P(ExtOpTest,
                         ExtOpSoftmaxUnsupportedDatatypeTest,
                         testing::Values<hipDataType>(HIP_R_16F, HIP_R_16BF));

INSTANTIATE_TEST_SUITE_P(ExtOpTest,
                         ExtOpLayerNormTest,
                         testing::Values<uint32_t>(1, 16, 1335, 6666));
INSTANTIATE_TEST_SUITE_P(ExtOpTest,
                         ExtOpLayerNormUnsupportedDatatypeTest,
                         testing::Values<hipDataType>(HIP_R_16F, HIP_R_16BF));

INSTANTIATE_TEST_SUITE_P(
    ExtOpTest,
    ExtOpAMaxTest,
    testing::Values<AMaxTestData>(AMaxTestData{HIP_R_32F, HIP_R_32F, 1, 1},
                                  AMaxTestData{HIP_R_32F, HIP_R_32F, 16, 16},
                                  AMaxTestData{HIP_R_32F, HIP_R_32F, 1335, 666},
                                  AMaxTestData{HIP_R_32F, HIP_R_16F, 1, 1},
                                  AMaxTestData{HIP_R_32F, HIP_R_16F, 16, 16},
                                  AMaxTestData{HIP_R_32F, HIP_R_16F, 1335, 666},
                                  AMaxTestData{HIP_R_16F, HIP_R_32F, 1, 1},
                                  AMaxTestData{HIP_R_16F, HIP_R_32F, 16, 16},
                                  AMaxTestData{HIP_R_16F, HIP_R_32F, 1335, 666},
                                  AMaxTestData{HIP_R_16F, HIP_R_16F, 1, 1},
                                  AMaxTestData{HIP_R_16F, HIP_R_16F, 16, 16},
                                  AMaxTestData{HIP_R_16F, HIP_R_16F, 1335, 666}));
INSTANTIATE_TEST_SUITE_P(ExtOpTest,
                         ExtOpAMaxUnsupportedDatatypeTest,
                         testing::Values<hipDataType>(HIP_R_16BF));
