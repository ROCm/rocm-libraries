/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt-ext-op.h>
#include <vector>

#include "../include/hipblaslt_random.hpp"

namespace
{
    template <typename DType>
    void cpuSoftmax(DType* m, DType* a, std::uint32_t numRows, std::uint32_t numCols)
    {
        for(std::uint32_t i = 0; i < numRows; ++i)
        {
            const auto rowMax = *std::max_element(a + i * numCols, a + i * numCols + numCols);
            auto       rowSum = 0.f;
            std::transform(a + i * numCols,
                           a + i * numCols + numCols,
                           m + i * numCols,
                           [&rowSum, rowMax](auto v) {
                               const auto u = std::exp(v - rowMax);
                               rowSum += u;
                               return u;
                           });

            std::transform(m + i * numCols,
                           m + i * numCols + numCols,
                           m + i * numCols,
                           [rowSum](auto v) { return v / rowSum; });
        }
    }

}

class ExtOpSoftmaxTest : public testing::TestWithParam<uint32_t>
{
};
class ExtOpSoftmaxUnsupportedDatatypeTest : public testing::TestWithParam<hipDataType>
{
};

TEST_P(ExtOpSoftmaxTest, softmaxSuccess)
{
    uint32_t           m = GetParam();
    uint32_t           n = 16;
    std::vector<float> input(m * n, 0.f);
    std::vector<float> output(m * n, 0.f);
    hipblaslt_uniform_int_1_10_run_float(input.data(), input.size());
    float* gpuInput{};
    float* gpuOutput{};

    auto err          = hipMalloc(&gpuInput, m * n * sizeof(float));
    err               = hipMalloc(&gpuOutput, m * n * sizeof(float));
    err               = hipMemcpyHtoD(gpuInput, input.data(), m * n * sizeof(float));
    auto hipblasltErr = hipblasltExtSoftmax(HIP_R_32F, m, n, 1, gpuOutput, gpuInput, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_SUCCESS);
    err = hipDeviceSynchronize();
    ASSERT_EQ(err, hipSuccess);
    std::vector<float> cpuRef(m * n, 0.f);
    cpuSoftmax(cpuRef.data(), input.data(), m, n);
    err = hipMemcpyDtoH(output.data(), gpuOutput, m * n * sizeof(float));

    for(std::size_t i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(output[i], cpuRef[i], 1e-5);
    }

    err = hipFree(gpuInput);
    err = hipFree(gpuOutput);
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

INSTANTIATE_TEST_SUITE_P(ExtOpTest, ExtOpSoftmaxTest, testing::Values<uint32_t>(1, 16, 1335));
INSTANTIATE_TEST_SUITE_P(ExtOpTest,
                         ExtOpSoftmaxUnsupportedDatatypeTest,
                         testing::Values<hipDataType>(HIP_R_16F, HIP_R_16BF));
