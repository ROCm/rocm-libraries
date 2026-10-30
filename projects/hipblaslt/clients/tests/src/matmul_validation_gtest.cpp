// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipblaslt/hostnumerics/MatmulValidation.hpp>
#include <hipblaslt/hostnumerics/norm.hpp>

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
    std::pair<double, bool> validatePointerArrayNorm(const std::array<float, 2>& observed,
                                                     bool                        assertNorm)
    {
        using namespace hipblaslt::hostnumerics;

        const std::array<float, 2> expected{1.0f, 1.0f};
        MatmulValidationCase       testCase;
        for(size_t batch = 0; batch < expected.size(); ++batch)
        {
            testCase.outputs.emplace_back(roc::hostnumerics::Tensor(expected[batch]),
                                          roc::hostnumerics::Tensor(observed[batch]));
        }

        const MatmulValidationSummary summary = validateMatmulOutputs(
            {.compareNorm = true, .assertNorm = assertNorm}, std::span(&testCase, 1));
        return {summary.relativeFrobeniusError, summary.passed};
    }

    hipblaslt::hostnumerics::MatmulValidationCase::TensorPair scalarComparison(float expected,
                                                                               float observed)
    {
        return {roc::hostnumerics::Tensor(expected), roc::hostnumerics::Tensor(observed)};
    }

    std::pair<double, double>
        validateAllClose(std::span<const hipblaslt::hostnumerics::MatmulValidationCase> cases)
    {
        using namespace hipblaslt::hostnumerics;

        const MatmulValidationSummary summary
            = validateMatmulOutputs({.searchAllClose = true}, cases);
        return {summary.absoluteTolerance, summary.relativeTolerance};
    }

    // M x N x batch outputs whose expected values repeat 1, 2, 3; scale changes one batch.
    struct BatchedOutputs
    {
        roc::hostnumerics::Tensor expected;
        roc::hostnumerics::Tensor observed;

        BatchedOutputs(roc::hostnumerics::ScalarType type, size_t m, size_t n, size_t batches)
            : expected(type, roc::hostnumerics::Shape{m, n, batches})
            , observed(type, roc::hostnumerics::Shape{m, n, batches})
        {
            for(size_t b = 0; b < batches; ++b)
                for(size_t i = 0; i < m; ++i)
                    for(size_t j = 0; j < n; ++j)
                    {
                        const double value = 1.0 + double((i + j * m + b) % 3);
                        expected.storeFrom({i, j, b}, value);
                        observed.storeFrom({i, j, b}, value);
                    }
        }

        void scaleBatch(size_t batch, double factor) const
        {
            for(size_t i = 0; i < observed.shape()[0]; ++i)
                for(size_t j = 0; j < observed.shape()[1]; ++j)
                    observed.storeFrom({i, j, batch},
                                       observed.loadAs<double>({i, j, batch}) * factor);
        }

        hipblaslt::hostnumerics::MatmulValidationSummary
            validate(roc::hostnumerics::ScalarType inputType) const
        {
            hipblaslt::hostnumerics::MatmulValidationCase testCase;
            testCase.outputs.emplace_back(expected, observed);
            return hipblaslt::hostnumerics::validateMatmulOutputs({.compareNorm    = true,
                                                                   .searchAllClose = true,
                                                                   .inputTypeA     = inputType,
                                                                   .inputTypeB     = inputType},
                                                                  std::span(&testCase, 1));
        }
    };
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
    using namespace hipblaslt::hostnumerics;

    const std::array<float, 3> expected{0.0f, 1.0f, 1.0f};
    const std::array<float, 3> observed{0.005f, 1.005f, 1.0f};
    MatmulValidationCase       testCase;
    for(size_t output = 0; output < expected.size(); ++output)
        testCase.outputs.push_back(scalarComparison(expected[output], observed[output]));

    const auto [absolute, relative]
        = validateAllClose(std::span<const MatmulValidationCase>(&testCase, 1));
    EXPECT_DOUBLE_EQ(absolute, 1e-2);
    EXPECT_DOUBLE_EQ(relative, 1e-2);
}

TEST(HostNumericsMatmulValidation, GroupedCasesKeepEarlierAllCloseFailure)
{
    using namespace hipblaslt::hostnumerics;

    const std::array<float, 2>          expected{0.0f, 1.0f};
    const std::array<float, 2>          observed{2.0f, 1.0f};
    std::array<MatmulValidationCase, 2> cases;
    for(size_t problem = 0; problem < cases.size(); ++problem)
        cases[problem].outputs.push_back(scalarComparison(expected[problem], observed[problem]));

    const auto [absolute, relative] = validateAllClose(cases);
    EXPECT_DOUBLE_EQ(absolute, 1.0);
    EXPECT_DOUBLE_EQ(relative, 1.0);
}

TEST(HostNumericsMatmulValidation, EmptyOutputsAreNoOps)
{
    using namespace hipblaslt::hostnumerics;
    using namespace roc::hostnumerics;

    MatmulValidationCase testCase;
    testCase.outputs.emplace_back(Tensor(ScalarType::Float32, Shape{0, 3, 2}),
                                  Tensor(ScalarType::Float32, Shape{0, 3, 2}));

    const MatmulValidationSummary summary = validateMatmulOutputs({.compareAllClose = true,
                                                                   .compareNorm     = true,
                                                                   .searchAllClose  = true,
                                                                   .computeUlp      = true,
                                                                   .assertNorm      = true},
                                                                  std::span(&testCase, 1));
    EXPECT_TRUE(summary.passed);
    EXPECT_EQ(summary.relativeFrobeniusError, 0.0);
    EXPECT_EQ(summary.absoluteTolerance, 0.0);
    EXPECT_EQ(summary.relativeTolerance, 0.0);
    EXPECT_EQ(summary.maximumUlp, 0.0);
    EXPECT_EQ(summary.averageUlp, 0.0);
}

TEST(HostNumericsMatmulValidation, UnitCheckPreservesNonFiniteSemantics)
{
    using namespace hipblaslt::hostnumerics;

    const float          nan = std::numeric_limits<float>::quiet_NaN();
    MatmulValidationCase matching;
    MatmulValidationCase mismatching;
    matching.outputs.push_back(scalarComparison(nan, nan));
    mismatching.outputs.push_back(scalarComparison(nan, 1.0f));
    const MatmulValidationOptions options{.compareAllClose = true};

    EXPECT_TRUE(validateMatmulOutputs(options, std::span(&matching, 1)).passed);
    EXPECT_FALSE(validateMatmulOutputs(options, std::span(&mismatching, 1)).passed);
}

TEST(HostNumericsMatmulValidation, NormOnlyPreservesNonFiniteConsistencyCheck)
{
    using namespace hipblaslt::hostnumerics;

    const float          infinity = std::numeric_limits<float>::infinity();
    MatmulValidationCase mismatching;
    mismatching.outputs.push_back(scalarComparison(infinity, 1.0f));

    EXPECT_FALSE(validateMatmulOutputs({.compareNorm = true}, std::span(&mismatching, 1)).passed);
}

// Norm and all-close failures are recorded without assertNorm, so a benchmark can count them.
TEST(HostNumericsMatmulValidation, WrongLaterBatchRecordsNormAndAllCloseFailures)
{
    using Kind = hipblaslt::hostnumerics::MatmulOutputCheckFailure::Kind;
    using roc::hostnumerics::ScalarType;

    BatchedOutputs outputs(ScalarType::Float32, 8, 4, 3);
    outputs.observed.storeFrom({5, 0, 2}, 7.0);
    const auto summary = outputs.validate(ScalarType::Float32);
    EXPECT_TRUE(summary.passed);
    EXPECT_GT(summary.relativeFrobeniusError, 0.1);
    EXPECT_EQ(summary.absoluteTolerance, 1.0);
    ASSERT_EQ(summary.outputCheckFailures.size(), 2u);
    EXPECT_EQ(summary.outputCheckFailures[0].kind, Kind::Norm);
    EXPECT_EQ(summary.outputCheckFailures[0].batch, 2u);
    EXPECT_DOUBLE_EQ(summary.outputCheckFailures[0].limit, 1e-5);
    EXPECT_EQ(summary.outputCheckFailures[1].kind, Kind::AllClose);
}

TEST(HostNumericsMatmulValidation, MatchingPointerArrayRecordsNoFailure)
{
    using namespace hipblaslt::hostnumerics;

    MatmulValidationCase testCase;
    for(size_t batch = 0; batch < 3; ++batch)
        testCase.outputs.push_back(scalarComparison(1.0f + batch, 1.0f + batch));
    const auto summary = validateMatmulOutputs({.compareNorm = true, .searchAllClose = true},
                                               std::span(&testCase, 1));
    EXPECT_TRUE(summary.outputCheckFailures.empty());
    EXPECT_EQ(summary.relativeFrobeniusError, 0.0);
    EXPECT_EQ(summary.absoluteTolerance, 1e-6);
}

// Each batch is off by about 4e-6 relative, under the f32 limit of 1e-5. The reported
// error sums the eight batches past the limit, but each batch passes on its own.
TEST(HostNumericsMatmulValidation, AccurateBatchesDoNotAddUpToNormFailure)
{
    using roc::hostnumerics::ScalarType;

    BatchedOutputs outputs(ScalarType::Float32, 8, 4, 8);
    for(size_t batch = 0; batch < 8; ++batch)
        outputs.scaleBatch(batch, 1.000004);
    const auto summary = outputs.validate(ScalarType::Float32);
    EXPECT_TRUE(summary.outputCheckFailures.empty());
    EXPECT_GT(summary.relativeFrobeniusError, 1e-5);
}

TEST(HostNumericsMatmulValidation, PointerArrayNormFailureNamesItsBatch)
{
    using namespace hipblaslt::hostnumerics;

    MatmulValidationCase testCase;
    testCase.outputs.push_back(scalarComparison(1.0f, 1.0f));
    testCase.outputs.push_back(scalarComparison(1.0f, 1.5f));
    testCase.outputs.push_back(scalarComparison(1.0f, 1.0f));
    const auto summary = validateMatmulOutputs({.compareNorm = true}, std::span(&testCase, 1));
    ASSERT_EQ(summary.outputCheckFailures.size(), 1u);
    EXPECT_EQ(summary.outputCheckFailures[0].batch, 1u);
    EXPECT_DOUBLE_EQ(summary.outputCheckFailures[0].error, 0.5);
}

TEST(HostNumericsMatmulValidation, MixedPrecisionNormUsesGemmTolerance)
{
    using roc::hostnumerics::ScalarType;

    BatchedOutputs outputs(ScalarType::Float32, 8, 4, 1);
    outputs.scaleBatch(0, 1.001);
    const auto summary = outputs.validate(ScalarType::Float16);
    EXPECT_TRUE(summary.outputCheckFailures.empty());
    EXPECT_GT(summary.relativeFrobeniusError, norm_tolerance(ScalarType::Float32));
    EXPECT_DOUBLE_EQ(
        norm_tolerance(
            ScalarType::Float32, HIPBLAS_COMPUTE_32F, ScalarType::Float16, ScalarType::Float16),
        0.01);
    EXPECT_DOUBLE_EQ(
        norm_tolerance(
            ScalarType::Float16, HIPBLAS_COMPUTE_32F, ScalarType::BFloat16, ScalarType::BFloat16),
        0.1);
    EXPECT_DOUBLE_EQ(norm_tolerance(ScalarType::Float32,
                                    HIPBLAS_COMPUTE_32F_FAST_16BF,
                                    ScalarType::Float32,
                                    ScalarType::Float32),
                     0.5);
}

// F16 output uses the f16 limit of 0.01: a 0.4% error passes, which the f32 limit would
// fail, and a 2% error fails, which the bf16 limit would pass. All-close passes both.
TEST(HostNumericsMatmulValidation, HalfOutputNormUsesHalfTolerance)
{
    using Kind = hipblaslt::hostnumerics::MatmulOutputCheckFailure::Kind;
    using roc::hostnumerics::ScalarType;

    BatchedOutputs withinLimit(ScalarType::Float16, 8, 4, 2);
    BatchedOutputs beyondLimit(ScalarType::Float16, 8, 4, 2);
    for(size_t batch = 0; batch < 2; ++batch)
    {
        withinLimit.scaleBatch(batch, 1.004);
        beyondLimit.scaleBatch(batch, 1.02);
    }
    const auto within = withinLimit.validate(ScalarType::Float16);
    EXPECT_TRUE(within.outputCheckFailures.empty());
    EXPECT_GT(within.relativeFrobeniusError, norm_tolerance(ScalarType::Float32));
    const auto beyond = beyondLimit.validate(ScalarType::Float16);
    ASSERT_EQ(beyond.outputCheckFailures.size(), 1u);
    EXPECT_EQ(beyond.outputCheckFailures[0].kind, Kind::Norm);
}

TEST(MatmulAlgoIndex, MixedValidityContract)
{
    hipblaslt_local_handle                        handle;
    std::vector<hipblasLtMatmulHeuristicResult_t> allAlgorithms;
    const auto                                    status = hipblaslt_ext::getAllAlgos(handle,
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
