// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipblaslt/host_numerics/MatmulValidation.hpp>

#include "utility.hpp"
#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <utility>
#include <vector>

namespace
{
    std::pair<double, bool>
        validatePointerArrayNorm(const std::array<float, 2>& observed, bool assertNorm)
    {
        using namespace hipblaslt::host_numerics;

        const std::array<float, 2> expected{1.0f, 1.0f};
        MatmulValidationCase       testCase;
        for(size_t batch = 0; batch < expected.size(); ++batch)
        {
            HostComparisonRequest output;
            output.rows = output.columns = output.leadingDimension = output.batchCount = 1;
            output.expected = &expected[batch];
            output.observed = &observed[batch];
            output.type     = HIP_R_32F;
            testCase.outputs.push_back(output);
        }

        double error = 0.0, absolute = 0.0, relative = 0.0, maximumUlp = 0.0, averageUlp = 0.0;
        const bool passed
            = validateMatmulOutputs({.compareNorm = true, .assertNorm = assertNorm},
                                    std::span(&testCase, 1),
                                    {error, absolute, relative, maximumUlp, averageUlp});
        return {error, passed};
    }

    hipblaslt::host_numerics::HostComparisonRequest scalarComparison(const float* expected,
                                                                     const float* observed)
    {
        hipblaslt::host_numerics::HostComparisonRequest output;
        output.rows = output.columns = output.leadingDimension = output.batchCount = 1;
        output.expected                                                            = expected;
        output.observed                                                            = observed;
        output.type                                                                = HIP_R_32F;
        return output;
    }

    std::pair<double, double>
        validateAllClose(std::span<const hipblaslt::host_numerics::MatmulValidationCase> cases)
    {
        using namespace hipblaslt::host_numerics;

        double error = 0.0, absolute = 0.0, relative = 0.0, maximumUlp = 0.0, averageUlp = 0.0;
        validateMatmulOutputs(
            {.searchAllClose = true}, cases, {error, absolute, relative, maximumUlp, averageUlp});
        return {absolute, relative};
    }
}

TEST(HostNumericsMatmulValidation, PointerArrayNormCountsEachBatchOnce)
{
    const auto [error, passed] = validatePointerArrayNorm({2.0f, 3.0f}, false);
    EXPECT_DOUBLE_EQ(error, 3.0);
    EXPECT_TRUE(passed);
}

TEST(HostNumericsMatmulValidation, PointerArrayNormChecksEachBatch)
{
    const auto [error, passed] = validatePointerArrayNorm({100.0f, 1.0f}, true);
    EXPECT_DOUBLE_EQ(error, 99.0);
    EXPECT_FALSE(passed);
}

TEST(HostNumericsMatmulValidation, PointerArrayOutputsKeepCombinedAllCloseTolerance)
{
    using namespace hipblaslt::host_numerics;

    const std::array<float, 3> expected{0.0f, 1.0f, 1.0f};
    const std::array<float, 3> observed{0.005f, 1.005f, 1.0f};
    MatmulValidationCase       testCase;
    for(size_t output = 0; output < expected.size(); ++output)
        testCase.outputs.push_back(scalarComparison(&expected[output], &observed[output]));

    const auto [absolute, relative]
        = validateAllClose(std::span<const MatmulValidationCase>(&testCase, 1));
    EXPECT_DOUBLE_EQ(absolute, 1e-2);
    EXPECT_DOUBLE_EQ(relative, 1e-2);
}

TEST(HostNumericsMatmulValidation, GroupedCasesKeepEarlierAllCloseFailure)
{
    using namespace hipblaslt::host_numerics;

    const std::array<float, 2>          expected{0.0f, 1.0f};
    const std::array<float, 2>          observed{2.0f, 1.0f};
    std::array<MatmulValidationCase, 2> cases;
    for(size_t problem = 0; problem < cases.size(); ++problem)
        cases[problem].outputs.push_back(scalarComparison(&expected[problem], &observed[problem]));

    const auto [absolute, relative] = validateAllClose(cases);
    EXPECT_DOUBLE_EQ(absolute, 1.0);
    EXPECT_DOUBLE_EQ(relative, 1.0);
}

TEST(MatmulAlgoIndex, MixedValidityContract)
{
    hipblaslt_local_handle handle;
    std::vector<hipblasLtMatmulHeuristicResult_t> allAlgorithms;
    const auto status = hipblaslt_ext::getAllAlgos(handle,
                                                    hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                                    HIPBLAS_OP_N,
                                                    HIPBLAS_OP_N,
                                                    HIP_R_32F,
                                                    HIP_R_32F,
                                                    HIP_R_32F,
                                                    HIP_R_32F,
                                                    HIPBLAS_COMPUTE_32F,
                                                    allAlgorithms);
    if(status != HIPBLAS_STATUS_SUCCESS || allAlgorithms.empty())
        GTEST_SKIP() << "No F32 algorithms available";

    std::vector<int> validIndices;
    validIndices.reserve(allAlgorithms.size());
    for(auto& algorithm : allAlgorithms)
        validIndices.push_back(hipblaslt_ext::getIndexFromAlgo(algorithm.algo));

    auto verify = [&](std::vector<int> indices, hipblasStatus_t expectedStatus, size_t count) {
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        EXPECT_EQ(hipblaslt_ext::getAlgosFromIndex(handle, indices, results), expectedStatus);
        EXPECT_EQ(results.size(), count);
    };

    auto indices = validIndices;
    indices.push_back(std::numeric_limits<int>::max());
    verify(indices, HIPBLAS_STATUS_INVALID_VALUE, validIndices.size());
    indices = validIndices;
    indices.insert(indices.begin() + indices.size() / 2, std::numeric_limits<int>::max());
    verify(indices, HIPBLAS_STATUS_INVALID_VALUE, validIndices.size());
    indices = validIndices;
    std::reverse(indices.begin(), indices.end());
    verify(indices, HIPBLAS_STATUS_SUCCESS, validIndices.size());
    indices.clear();
    for(int index : validIndices)
        indices.insert(indices.end(), {index, index});
    verify(indices, HIPBLAS_STATUS_SUCCESS, 2 * validIndices.size());
    verify({}, HIPBLAS_STATUS_SUCCESS, 0);
    verify({std::numeric_limits<int>::max(), std::numeric_limits<int>::max() - 1},
           HIPBLAS_STATUS_INVALID_VALUE,
           0);
}
