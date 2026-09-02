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
#include <hipblaslt/host_numerics/HipblasltDataInitialization.hpp>
#include <hipblaslt/host_numerics/Types.hpp>
#include <roc/host_numerics/validation.hpp>
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
    hipblaslt::host_numerics::initialize(
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

    using namespace roc::host_numerics;
    using namespace hipblaslt::host_numerics;
    Tensor expected(ScalarType::Float32, Shape{m, n});
    referenceSoftmaxInto(
        copyTensorFromEncodedStorage(
            input.data(), input.size(), Layout::contiguousLastDimensionFastest(Shape{m, n})),
        expected,
        1,
        ScalarType::Float32);
    const ComparisonReport comparison = compare(
        copyTensorFromEncodedStorage(
            output.data(), output.size(), Layout::contiguousLastDimensionFastest(Shape{m, n})),
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

    hipblaslt::host_numerics::initialize(
        input.data(), input.size(), hipblaslt_initialization::hpl);
    hipblaslt::host_numerics::initialize(
        gamma.data(), gamma.size(), hipblaslt_initialization::hpl);
    hipblaslt::host_numerics::initialize(beta.data(), beta.size(), hipblaslt_initialization::hpl);

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

    using namespace roc::host_numerics;
    using namespace hipblaslt::host_numerics;
    const Layout tensorLayout     = Layout::contiguousLastDimensionFastest(Shape{m, n});
    const Layout statisticsLayout = Layout::contiguousLastDimensionFastest(Shape{m});
    const Layout affineLayout     = Layout::contiguousLastDimensionFastest(Shape{n});

    LayerNormOptions options;
    options.axis    = 1;
    options.gamma   = copyTensorFromEncodedStorage(gamma.data(), gamma.size(), affineLayout);
    options.beta    = copyTensorFromEncodedStorage(beta.data(), beta.size(), affineLayout);
    options.epsilon = 1e-5;
    const LayerNormOutputs reference
        = referenceLayerNorm(copyTensorFromEncodedStorage(input.data(), input.size(), tensorLayout),
                             {.output          = ScalarType::Float32,
                              .mean            = ScalarType::Float32,
                              .inverseVariance = ScalarType::Float32},
                             options);

    const ComparisonOptions comparisonOptions = nearComparisonOptions(1e-5);
    const ComparisonReport  outputComparison
        = compare(copyTensorFromEncodedStorage(output.data(), output.size(), tensorLayout),
                  reference.output,
                  comparisonOptions);
    EXPECT_TRUE(outputComparison.passed())
        << "LayerNorm output mismatches: " << outputComparison.mismatches
        << ", max absolute difference: " << outputComparison.maxAbsoluteDifference;

    const ComparisonReport meanComparison
        = compare(copyTensorFromEncodedStorage(mean.data(), mean.size(), statisticsLayout),
                  *reference.mean,
                  comparisonOptions);
    EXPECT_TRUE(meanComparison.passed())
        << "LayerNorm mean mismatches: " << meanComparison.mismatches
        << ", max absolute difference: " << meanComparison.maxAbsoluteDifference;

    const ComparisonReport inverseVarianceComparison
        = compare(copyTensorFromEncodedStorage(invvar.data(), invvar.size(), statisticsLayout),
                  *reference.inverseVariance,
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

    ASSERT_EQ(hipMalloc(&gpuOutput, outNumBytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&gpuInput, m * n * inNumBytes), hipSuccess);

    std::vector<To> cpuOutput(1, 0.f);
    std::vector<Ti> cpuInput(m * n, 0.f);
    std::vector<To> refOutput(1, 0.f);

    hipblaslt::host_numerics::initialize(
        cpuInput.data(), cpuInput.size(), hipblaslt_initialization::hpl);

    ASSERT_EQ(hipMemcpyHtoD(gpuInput, cpuInput.data(), m * n * inNumBytes), hipSuccess);

    hipStream_t stream{};
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    auto hipblasltErr = hipblasltExtAMax(type, dtype, gpuOutput, gpuInput, m, n, stream);
    ASSERT_EQ(hipblasltErr, HIPBLAS_STATUS_SUCCESS);
    // The call is asynchronous on stream, and the allocator may reuse the preceding parameter's
    // output storage. Wait before copying the result to the host.
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipMemcpyDtoH(cpuOutput.data(), gpuOutput, outNumBytes), hipSuccess);

    using namespace roc::host_numerics;
    Tensor referenceOutput = hipblaslt::host_numerics::copyTensorFromEncodedStorage(
        refOutput.data(), refOutput.size(), Layout::contiguousLastDimensionFastest(Shape{}));
    referenceMaximumAbsoluteInto(hipblaslt::host_numerics::copyTensorFromEncodedStorage(
                                     cpuInput.data(),
                                     cpuInput.size(),
                                     Layout::contiguousLastDimensionFastest(Shape{numElements})),
                                 referenceOutput,
                                 ScalarType::Float32);
    hipblaslt::host_numerics::copyTensorEncodedBackingStorageToBuffer(
        refOutput.data(), refOutput.size(), referenceOutput);

    EXPECT_NEAR(float(refOutput[0]), float(cpuOutput[0]), 1e-5);

    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
    EXPECT_EQ(hipFree(gpuOutput), hipSuccess);
    EXPECT_EQ(hipFree(gpuInput), hipSuccess);
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
